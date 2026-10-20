"""Measure retrieval quality on data/eval/questions.json.

Metrics (computed per article, the citation unit):
    Recall@k  share of questions where a relevant article is in the top k
    MRR       mean of 1/rank of the first relevant article (0 if not in top 10)

Usage:
    python src/evaluate_retrieval.py [--modes bm25 dense hybrid] [--model BAAI/bge-m3] [--show-misses]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from retriever import DEFAULT_MODEL, MODES, Retriever

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "eval" / "questions.json"
KS = (1, 3, 5, 10)


def first_relevant_rank(ranked: list[int], relevant: list[int]) -> int | None:
    return next((r for r, n in enumerate(ranked, start=1) if n in relevant), None)


def summarize(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    out = {f"R@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    out["MRR"] = sum(1 / r for r in ranks if r) / n
    return out


def fmt(row: dict[str, float]) -> str:
    return "  ".join(f"{v:5.2f}" for v in row.values())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--show-misses", action="store_true", help="list questions not answered in top 5")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    retriever = Retriever(args.model)
    header = "  ".join(f"{h:>5}" for h in [*(f"R@{k}" for k in KS), "MRR"])

    print(f"{len(questions)} questions | model: {args.model}\n")
    print(f"{'mode':<8} {'style':<11} {'n':>3}  {header}")
    for mode in args.modes:
        ranks_by_style: dict[str, list] = defaultdict(list)
        misses = []
        for q in questions:
            ranked = [h.article_number for h in retriever.search(q["question"], k=max(KS), mode=mode)]
            rank = first_relevant_rank(ranked, q["relevant"])
            ranks_by_style["ALL"].append(rank)
            ranks_by_style[q["style"]].append(rank)
            if not rank or rank > 5:
                misses.append((q, rank, ranked[:5]))
        for style, ranks in sorted(ranks_by_style.items(), key=lambda kv: kv[0] != "ALL"):
            print(f"{mode:<8} {style:<11} {len(ranks):>3}  {fmt(summarize(ranks))}")
        if args.show_misses and misses:
            print(f"  misses (not in top 5) for {mode}:")
            for q, rank, top in misses:
                print(f"   {q['id']} want {q['relevant']} got {top} (rank {rank or '>10'}): {q['question']}")
        print()


if __name__ == "__main__":
    main()
