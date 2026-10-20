"""Embed data/chunks.json and store the vectors in a persistent Chroma collection.

    chunks.json --(embedding model)--> 1024-d vectors --> data/index/chroma/

Full build (~10 min on CPU): run after changing the model or the chunking strategy.
Incremental update (--update, seconds): re-embeds only chunks that are new or whose text
changed since the last build (e.g. after adding an amendment in data/amendments.json),
and deletes chunks that no longer exist.

The BM25 index is not stored: it takes well under a second to rebuild at query time.

Usage:
    python src/build_index.py [--model BAAI/bge-m3] [--batch-size 16]
    python src/build_index.py --update
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer

from retriever import CHROMA_DIR, CHUNKS, DEFAULT_MODEL, collection_name, passage_prefix

# Written only when the index content really changes. The web app watches this file to know when
# to reload; it can't watch Chroma's own files, whose mtimes change on every (read-only) query.
BUILD_ID = CHROMA_DIR.parent / "BUILD_ID"


def mark_built() -> None:
    BUILD_ID.write_text(str(time.time()), encoding="utf-8")


def embed(model_name: str, chunks: list[dict], batch_size: int):
    print(f"Loading {model_name} (first run downloads it to the Hugging Face cache)...")
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 512  # our longest chunk is ~450 tokens; shorter = faster
    texts = [passage_prefix(model_name) + c["embed_text"] for c in chunks]
    start = time.time()
    # normalize_embeddings=True -> unit vectors, so cosine similarity == dot product
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 16, convert_to_numpy=True)
    print(f"Embedded {len(texts)} chunks in {time.time() - start:.0f}s")
    return vectors.tolist()


def records(chunks: list[dict], vectors: list) -> dict:
    return dict(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        # Chroma metadata values must be str/int/float/bool (no None)
        metadatas=[{"article_number": c["article_number"], "part": c["part"],
                    "chapitre": c["chapitre"] or ""} for c in chunks],
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--update", action="store_true",
                        help="only re-embed new/changed chunks instead of rebuilding everything")
    args = parser.parse_args()

    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = collection_name(args.model)
    exists = name in [c.name for c in client.list_collections()]

    if args.update and exists:
        collection = client.get_collection(name, embedding_function=None)
        stored = collection.get(include=["documents"])
        stored_text = dict(zip(stored["ids"], stored["documents"]))
        changed = [c for c in chunks if stored_text.get(c["chunk_id"]) != c["text"]]
        removed = sorted(set(stored_text) - {c["chunk_id"] for c in chunks})
        if removed:
            collection.delete(ids=removed)
        if changed:
            collection.upsert(**records(changed, embed(args.model, changed, args.batch_size)))
        if changed or removed:
            mark_built()
        print(f"Updated '{name}': {len(changed)} re-embedded {[c['chunk_id'] for c in changed]}, "
              f"{len(removed)} removed {removed} -> {collection.count()} vectors")
        return

    vectors = embed(args.model, chunks, args.batch_size)
    if exists:
        client.delete_collection(name)
    collection = client.create_collection(
        name, metadata={"hnsw:space": "cosine", "model": args.model}, embedding_function=None)
    collection.add(**records(chunks, vectors))
    mark_built()
    print(f"Stored {collection.count()} vectors in collection '{name}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
