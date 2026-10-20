"""Retriever over the Code du travail: direct article lookup + dense embeddings (+ optional BM25).

    question ──► "article 39" mentioned? ──► direct lookup ─────────────┐
             ├─► bge-m3 embedding ──► Chroma ──► ranked chunks ────────┤ (weighted) Reciprocal
             └─► BM25 (French stemmed tokens) ──► ranked chunks ───────┘ Rank Fusion ──► articles
                 (only in --mode hybrid / bm25)

Results are returned per *article* (the citation unit): when several chunks of a long article
match, the article keeps its best-ranked chunk.

Usage:
    python src/retriever.py "Quelle est la durée du congé de maternité ?" [-k 5] [--mode dense]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from rank_bm25 import BM25Okapi

from text_utils import tokenize

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "chunks.json"
CHROMA_DIR = ROOT / "data" / "index" / "chroma"

DEFAULT_MODEL = "BAAI/bge-m3"
RRF_K = 60            # constant from Cormack et al. (2009); dampens the weight of top ranks
# Weight of the BM25 vote in "hybrid" mode (dense = 1.0). On our 50-question eval set every
# BM25 weight > 0 lowered recall (R@5: 0.92 at 0 -> 0.78 at 1.0), so the default mode is
# "dense"; hybrid is kept, lightly weighted, for exact-term queries. See docs/02_retrieval.md.
BM25_WEIGHT = 0.1
CANDIDATES = 50       # how many chunks each retriever contributes to the fusion
MODES = ("bm25", "dense", "hybrid")
DEFAULT_MODE = "dense"  # dense + direct article lookup

ARTICLE_REF_RE = re.compile(r"\bart(?:icle)?s?\.?\s*(\d{1,3}|premier)\b", re.IGNORECASE)

# Some embedding models are trained with instruction prefixes and perform worse without them.
_PREFIXES = {
    "intfloat/multilingual-e5": ("query: ", "passage: "),
}


def query_prefix(model: str) -> str:
    return next((q for k, (q, _) in _PREFIXES.items() if model.startswith(k)), "")


def passage_prefix(model: str) -> str:
    return next((p for k, (_, p) in _PREFIXES.items() if model.startswith(k)), "")


def collection_name(model: str) -> str:
    """One Chroma collection per embedding model, so models can be compared side by side."""
    return "code_travail_" + re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


@dataclass
class Hit:
    article_number: int
    score: float
    chunk: dict
    sources: list[str]  # which retrievers found it: "lookup", "bm25", "dense"

    @property
    def citation(self) -> str:
        return f"Article {self.article_number}"


class Retriever:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self.chunks: list[dict] = json.loads(CHUNKS.read_text(encoding="utf-8"))
        self.index_of = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.first_chunk_of = {}
        for i, c in enumerate(self.chunks):
            self.first_chunk_of.setdefault(c["article_number"], i)
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in self.chunks])

    # Dense components are loaded lazily: BM25-only use doesn't need PyTorch or Chroma.
    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name, device="cpu")
        model.max_seq_length = 512
        return model

    @cached_property
    def collection(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            return client.get_collection(collection_name(self.model_name), embedding_function=None)
        except Exception as e:  # chromadb raises different types across versions
            raise SystemExit(f"No index for {self.model_name}: run `python src/build_index.py "
                             f"--model {self.model_name}` first ({e})")

    # ---- individual retrievers: each returns chunk indices, best first
    def lookup(self, query: str) -> list[int]:
        numbers = []
        for m in ARTICLE_REF_RE.finditer(query):
            n = 1 if m.group(1).lower() == "premier" else int(m.group(1))
            if n in self.first_chunk_of:
                numbers.append(n)
        return [i for n in numbers for i, c in enumerate(self.chunks) if c["article_number"] == n]

    def bm25_search(self, query: str, n: int = CANDIDATES) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in ranked[:n] if scores[i] > 0]

    def dense_search(self, query: str, n: int = CANDIDATES) -> list[int]:
        vector = self.encoder.encode(query_prefix(self.model_name) + query, normalize_embeddings=True)
        result = self.collection.query(query_embeddings=[vector.tolist()], n_results=n)
        return [self.index_of[cid] for cid in result["ids"][0]]

    # ---- fusion
    def search(self, query: str, k: int = 5, mode: str = DEFAULT_MODE, bm25_weight: float = BM25_WEIGHT) -> list[Hit]:
        rankings: dict[str, list[int]] = {"lookup": self.lookup(query)}
        if mode in ("bm25", "hybrid"):
            rankings["bm25"] = self.bm25_search(query)
        if mode in ("dense", "hybrid"):
            rankings["dense"] = self.dense_search(query)

        scores: dict[int, float] = {}
        sources: dict[int, list[str]] = {}
        weights = {"lookup": 1.0, "bm25": bm25_weight if mode == "hybrid" else 1.0, "dense": 1.0}
        for name, ranked in rankings.items():
            for rank, idx in enumerate(ranked, start=1):
                # lookup adds a flat 1.0, far above any RRF score (< 0.02): cited articles come first
                rrf = 1.0 if name == "lookup" else 1.0 / (RRF_K + rank)
                scores[idx] = scores.get(idx, 0.0) + weights[name] * rrf
                sources.setdefault(idx, []).append(name)

        # collapse chunks -> articles, keeping each article's best chunk
        best: dict[int, Hit] = {}
        for idx in sorted(scores, key=scores.get, reverse=True):
            num = self.chunks[idx]["article_number"]
            if num not in best:
                best[num] = Hit(num, scores[idx], self.chunks[idx], sources[idx])
            if len(best) == k:
                break
        return list(best.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    retriever = Retriever(args.model)
    for rank, hit in enumerate(retriever.search(args.question, args.k, args.mode), start=1):
        c = hit.chunk
        where = " > ".join(x for x in (c["titre"], c["chapitre"], c["section"]) if x)
        print(f"{rank}. {hit.citation}  [score {hit.score:.4f} via {'+'.join(hit.sources)}]")
        print(f"   {where}")
        print(f"   {c['text'][:220].replace(chr(10), ' ')}…\n")


if __name__ == "__main__":
    main()
