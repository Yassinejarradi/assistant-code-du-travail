"""Turn data/articles.json into retrieval chunks: data/chunks.json.

Strategy (see docs/02_retrieval.md):
    - one article = one chunk, because the article is also the unit we cite;
    - articles longer than MAX_CHARS (12 of 589) are split at paragraph boundaries,
      and every part repeats the article's first paragraph (often a lead-in such as
      "Sont punis d'une amende de ... :" that the following list items depend on);
    - each chunk has two texts:
        `text`        the raw article (part), shown to the user and given to the LLM
        `embed_text`  hierarchy path + article number + text, used for embedding/BM25,
                      so that chapter titles ("De la protection de la maternité")
                      contribute to retrieval even when the body doesn't repeat them;
    - amendments published after 2011 (data/amendments.json) are appended to their article
      as a labelled note with the amending law and its Bulletin Officiel reference.

Usage:
    python src/chunking.py [--max-chars 1500]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES = ROOT / "data" / "articles.json"
CHUNKS = ROOT / "data" / "chunks.json"
AMENDMENTS = ROOT / "data" / "amendments.json"  # changes published after the 2011 consolidation

MAX_CHARS = 1500  # ~400 tokens; only the ~2% longest articles exceed it
HIERARCHY = ["livre", "titre", "chapitre", "section", "sous_section"]


def hierarchy_path(article: dict) -> str:
    return " > ".join(article[k] for k in HIERARCHY if article.get(k))


def split_article(text: str, max_chars: int) -> list[str]:
    """Greedy grouping of paragraphs into parts of at most ~max_chars.
    Parts after the first are prefixed with the article's first paragraph."""
    paragraphs = text.split("\n")
    if len(text) <= max_chars or len(paragraphs) == 1:
        return [text]
    lead, rest = paragraphs[0], paragraphs[1:]
    parts, current = [], [lead]
    for p in rest:
        size = sum(len(x) + 1 for x in current) + len(p)
        if size > max_chars and len(current) > 1:
            parts.append("\n".join(current))
            current = [lead, p] if len(lead) < max_chars // 3 else [p]
        else:
            current.append(p)
    parts.append("\n".join(current))
    return parts


def load_amendments() -> dict[int, list[dict]]:
    if not AMENDMENTS.exists():
        return {}
    by_article: dict[int, list[dict]] = {}
    for a in json.loads(AMENDMENTS.read_text(encoding="utf-8"))["amendments"]:
        by_article.setdefault(a["article_number"], []).append(a)
    return by_article


def amendment_note(a: dict) -> str:
    """Appended to the article text, so the LLM sees (and cites) the change with its legal source."""
    if a.get("official_text"):
        return f"[Modifié par la {a['law']} — {a['bulletin_officiel']}] Nouvelle rédaction : {a['official_text']}"
    return (f"[Mise à jour — {a['law']}, {a['bulletin_officiel']}, en vigueur depuis le "
            f"{a['in_force']}. Résumé, texte officiel non reproduit] {a['summary']}")


def build_chunks(articles: list[dict], max_chars: int) -> list[dict]:
    amendments = load_amendments()
    chunks = []
    for art in articles:
        notes = "".join("\n" + amendment_note(a) for a in amendments.get(art["article_number"], []))
        # Notes go on EVERY part, so no chunk can show the old rule without its update.
        parts = [p + notes for p in split_article(art["text"], max_chars)]
        path = hierarchy_path(art)
        for i, part in enumerate(parts, start=1):
            suffix = f" (partie {i}/{len(parts)})" if len(parts) > 1 else ""
            chunks.append({
                "chunk_id": f"{art['id']}" + (f"-p{i}" if len(parts) > 1 else ""),
                "article_id": art["id"],
                "article_number": art["article_number"],
                "part": i,
                "n_parts": len(parts),
                "text": part,
                "embed_text": f"{path}\nArticle {art['article_number']}{suffix}\n{part}",
                "livre": art["livre"],
                "titre": art["titre"],
                "chapitre": art["chapitre"],
                "section": art["section"],
                "sous_section": art.get("sous_section"),
                "abrogated": art.get("abrogated", False),
                "amended_by": [a["law"] for a in amendments.get(art["article_number"], [])],
                "page": art.get("page"),
            })
    return chunks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = parser.parse_args()

    articles = json.loads(ARTICLES.read_text(encoding="utf-8"))
    chunks = build_chunks(articles, args.max_chars)
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    split = sorted({c["article_number"] for c in chunks if c["n_parts"] > 1})
    longest = max(len(c["text"]) for c in chunks)
    print(f"{len(articles)} articles -> {len(chunks)} chunks ({len(split)} articles split: {split})")
    print(f"longest chunk text: {longest} chars -> {CHUNKS}")


if __name__ == "__main__":
    main()
