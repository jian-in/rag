"""Retrieval evaluation harness for the local RAG knowledge base.

Usage:
    1. Copy eval_golden.example.jsonl to eval_golden.jsonl and fill in
       50-100 question/expect pairs from YOUR OWN notes (one JSON per line):
       {"question": "...", "expect": "substring of the expected source path"}
    2. Run:  RAG/Scripts/python.exe eval_retrieval.py [--k 1 3 5 10] [--rewrite none|expand|decompose|hyde]

Metrics:
    recall@k  - fraction of questions whose expected source appears in top-k
    MRR       - mean reciprocal rank of the first correct hit

Golden-set discipline (from 2026 production consensus):
    - Questions must come from real usage or real notes, not LLM fantasies.
    - Change ONE variable at a time between runs and keep the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mcp_server import KnowledgeRuntime  # noqa: E402

GOLDEN_PATH = ROOT / "eval_golden.jsonl"


def load_golden(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            item = json.loads(line)
            if "question" not in item or "expect" not in item:
                raise ValueError(f"line {line_no}: needs 'question' and 'expect'")
            items.append(item)
    return items


def evaluate(
    runtime: KnowledgeRuntime,
    golden: list[dict],
    ks: list[int],
    rewrite_strategy: str | None,
    window: int,
) -> None:
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    rr_sum = 0.0

    for idx, item in enumerate(golden, start=1):
        result = runtime.search(
            query=item["question"],
            top_k=max_k,
            use_hybrid=True,
            rewrite_strategy=rewrite_strategy,
            window=window,
        )
        # Normalize path separators so golden "expect" works with / or \ on any OS.
        sources = [r.get("source", "").replace("\\", "/") for r in result["results"]]
        expect = item["expect"].replace("\\", "/")

        rank = None
        for pos, src in enumerate(sources, start=1):
            if expect in src:
                rank = pos
                break

        if rank is not None:
            rr_sum += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits_at[k] += 1
        tag = f"hit@{rank}" if rank else "miss"
        print(f"[{idx:3}/{len(golden)}] {tag:>6}  {item['question'][:60]}")

    n = len(golden)
    print("\n===== summary =====")
    print(f"questions: {n}  rewrite: {rewrite_strategy or 'none'}  window: {window}")
    for k in ks:
        print(f"recall@{k:<2}: {hits_at[k]/n:.3f}  ({hits_at[k]}/{n})")
    print(f"MRR      : {rr_sum/n:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against a golden set.")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--rewrite",
        choices=["none", "expand", "decompose", "hyde"],
        default="none",
        help="Query rewriting strategy to test (default: none).",
    )
    parser.add_argument("--window", type=int, default=0, help="Parent-window size (0 = off).")
    args = parser.parse_args()

    if not GOLDEN_PATH.exists():
        print(f"Golden set not found: {GOLDEN_PATH}")
        print("Copy eval_golden.example.jsonl to eval_golden.jsonl and fill it in first.")
        return 1

    golden = load_golden(GOLDEN_PATH)
    if not golden:
        print("Golden set is empty.")
        return 1

    runtime = KnowledgeRuntime()
    runtime.ensure_loaded()
    print(f"loaded {runtime.doc_count} chunks; evaluating {len(golden)} questions...\n")

    evaluate(
        runtime,
        golden,
        ks=sorted(args.k),
        rewrite_strategy=None if args.rewrite == "none" else args.rewrite,
        window=args.window,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
