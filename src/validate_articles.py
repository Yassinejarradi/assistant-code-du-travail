"""Validate data/articles.json produced by extract_articles.py.

Checks:
    - article numbers are continuous from 1 to the last article (missing / duplicates)
    - no empty articles (repealed articles marked `abrogated` are allowed)
    - every record has the required fields
    - leftover extraction artefacts (glued footnote markers, page numbers, "Article N" inside text)
Prints stats: total, average/median length, shortest and longest articles, counts per livre.

Exit code is 1 if any error is found, so it can be used in CI.

Usage:
    python src/validate_articles.py [--file PATH] [--expected-last 589]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "data" / "articles.json"

REQUIRED_FIELDS = ["id", "article_number", "livre", "titre", "chapitre", "section",
                   "text", "source", "version_date"]
SUSPICIOUS = {
    "glued footnote marker": re.compile(r"[^\W\d]\d{1,3}(?=[\s.,;:)]|$)"),  # "section36", not "2.000"
    "page-number residue": re.compile(r"(^|\n)\s*-\s*\d{1,3}\s*-\s*($|\n)"),
    "article heading inside text": re.compile(r"(^|\n)Article\s+\d+\s*($|\n)"),
}


def preview(text: str, width: int = 90) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--expected-last", type=int, default=589)
    args = parser.parse_args()

    records = json.loads(args.file.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    # --- schema
    for r in records:
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            errors.append(f"{r.get('id', '?')}: missing fields {missing}")
    ids = Counter(r["id"] for r in records)
    errors += [f"duplicate id {i} (x{c})" for i, c in ids.items() if c > 1]

    # --- continuity
    numbers = [r["article_number"] for r in records]
    counts = Counter(numbers)
    expected = set(range(1, max(args.expected_last, max(numbers, default=0)) + 1))
    missing_numbers = sorted(expected - counts.keys())
    duplicates = sorted(n for n, c in counts.items() if c > 1)
    out_of_order = [b for a, b in zip(numbers, numbers[1:]) if b <= a]
    if missing_numbers:
        errors.append(f"missing articles: {missing_numbers}")
    if duplicates:
        errors.append(f"duplicated articles: {duplicates}")
    if out_of_order:
        errors.append(f"articles out of order at: {out_of_order[:20]}")

    # --- empty / repealed
    abrogated = [r["article_number"] for r in records if r.get("abrogated")]
    empty = [r["article_number"] for r in records if not r.get("abrogated") and not r["text"].strip()]
    if empty:
        errors.append(f"empty articles: {empty}")
    no_hierarchy = [r["article_number"] for r in records if not r["livre"]]
    if no_hierarchy:
        errors.append(f"articles without livre: {no_hierarchy}")

    # --- artefacts (warnings: may be legitimate, worth a look)
    for r in records:
        for label, pattern in SUSPICIOUS.items():
            m = pattern.search(r["text"])
            if m:
                ctx = r["text"][max(0, m.start() - 30): m.end() + 30]
                warnings.append(f"art. {r['article_number']}: possible {label}: …{preview(ctx, 80)}…")

    # --- stats
    live = [r for r in records if not r.get("abrogated")]
    lengths = [len(r["text"]) for r in live]
    words = [len(r["text"].split()) for r in live]
    by_len = sorted(live, key=lambda r: len(r["text"]))

    print(f"File: {args.file}")
    print("=" * 72)
    print(f"Total articles        : {len(records)}  (range {min(numbers)}–{max(numbers)})")
    print(f"Repealed (abrogés)    : {len(abrogated)} {abrogated}")
    print(f"With footnotes        : {sum(1 for r in records if r.get('notes'))}")
    if lengths:
        print(f"Length (chars)        : avg {statistics.mean(lengths):.0f} | median "
              f"{statistics.median(lengths):.0f} | min {min(lengths)} | max {max(lengths)}")
        print(f"Length (words)        : avg {statistics.mean(words):.0f} | median "
              f"{statistics.median(words):.0f} | total {sum(words)}")
        print("\nShortest articles:")
        for r in by_len[:5]:
            print(f"  art. {r['article_number']:>3} ({len(r['text']):>4} chars): {preview(r['text'])}")
        print("Longest articles:")
        for r in reversed(by_len[-5:]):
            print(f"  art. {r['article_number']:>3} ({len(r['text']):>4} chars): {preview(r['text'])}")

    print("\nArticles per livre:")
    for livre, n in Counter(r["livre"] for r in records).items():
        print(f"  {n:>4}  {preview(livre or '(none)', 80)}")

    print("\n" + "=" * 72)
    if warnings:
        print(f"{len(warnings)} warning(s) (check manually):")
        for w in warnings:
            print(f"  ! {w}")
    if errors:
        print(f"{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  ✗ {e}")
        return 1
    print("✓ OK: numbering continuous, no duplicates, no empty articles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
