#!/usr/bin/env python
"""Fail the build when detection quality regresses.

This is the point of the eval CI job. Unit tests catch code that throws; they do
not catch a refactor that quietly costs eight points of average precision. A
model change that makes the tool worse should be as loud as a syntax error.

    python scripts/check_regression.py \
        --current artifacts/reports/evaluation.json \
        --baseline benchmarks/baseline.json \
        --tolerance 0.05

Exit codes: 0 pass, 1 regression, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GUARDED = [
    ("headline.pr_auc", "PR-AUC"),
    ("headline.precision_at.50", "Precision@50"),
    ("headline.precision_at.100", "Precision@100"),
    ("clustering.ari", "Cluster ARI"),
]


def dig(obj: dict, path: str):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument(
        "--tolerance", type=float, default=0.05, help="Absolute drop tolerated per metric."
    )
    ap.add_argument("--markdown", type=Path, default=None, help="Write a PR-comment table here.")
    args = ap.parse_args()

    if not args.current.exists():
        print(f"::error::no evaluation output at {args.current}", file=sys.stderr)
        return 2
    current = json.loads(args.current.read_text(encoding="utf-8"))

    if not args.baseline.exists():
        print(f"::warning::no baseline at {args.baseline}; recording current run as baseline")
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return 0
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    rows, failed = [], []
    for path, label in GUARDED:
        cur, base = dig(current, path), dig(baseline, path)
        if cur is None or base is None:
            continue
        delta = cur - base
        ok = delta >= -args.tolerance
        rows.append((label, base, cur, delta, ok))
        if not ok:
            failed.append((label, base, cur, delta))

    width = max(len(r[0]) for r in rows) if rows else 10
    print(f"\n{'metric'.ljust(width)}  {'baseline':>9}  {'current':>9}  {'delta':>8}  status")
    print("-" * (width + 42))
    for label, base, cur, delta, ok in rows:
        print(
            f"{label.ljust(width)}  {base:>9.4f}  {cur:>9.4f}  {delta:>+8.4f}  "
            f"{'ok' if ok else 'REGRESSION'}"
        )

    if args.markdown:
        lines = [
            "### Detection benchmark\n",
            "| Metric | Baseline | This PR | Δ | |",
            "|---|---:|---:|---:|:--|",
        ]
        for label, base, cur, delta, ok in rows:
            lines.append(
                f"| {label} | {base:.4f} | {cur:.4f} | {delta:+.4f} | {'✅' if ok else '❌'} |"
            )
        lines.append(f"\n_Tolerance: {args.tolerance:+.3f} absolute per metric._")
        if failed:
            lines.append("\n**Detection quality regressed — this build is blocked.**")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines), encoding="utf-8")

    if failed:
        for label, base, cur, delta in failed:
            print(f"::error::{label} regressed {base:.4f} -> {cur:.4f} ({delta:+.4f})")
        return 1

    print("\nno regression detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
