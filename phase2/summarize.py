#!/usr/bin/env python3
"""Aggregate isolated per-group specialist accuracy into one table.

"Isolated" = each specialist scored on its group's *true* test members (assumes
perfect routing). This is the upper bound; the cascade evaluator (later) will
report the end-to-end number that also pays for phase-1 routing errors.

Usage:  python phase2/summarize.py --model lgbm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from remake import taxonomy  # noqa: E402

RUNS = Path("runs")


def latest(prefix: str):
    cands = sorted(RUNS.glob(f"{prefix}_2*"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lgbm")
    args = ap.parse_args()

    print(f"\nPhase-2 specialist isolated accuracy  (model={args.model}, "
          f"assumes perfect routing)")
    print(f"  {'group':12s} {'leaves':>6} {'test_acc':>9} {'support':>9}  run")
    print("  " + "-" * 64)
    tw = tn = ts = tsa = 0.0
    for g in taxonomy.GROUP_NAMES:
        nL = len(taxonomy.GROUP_LEAVES[g])
        rd = latest(f"spec_{g}_{args.model}")
        mp = rd / "metrics_test.json" if rd else None
        if not mp or not mp.exists():
            print(f"  {g:12s} {nL:6d}    (not trained)")
            continue
        m = json.loads(mp.read_text())
        acc, n = m["accuracy"], m["n"]
        print(f"  {g:12s} {nL:6d} {acc:9.4f} {n:9d}  {rd.name}")
        tw += nL * acc; tn += nL; ts += n; tsa += n * acc
    if tn:
        print("  " + "-" * 64)
        print(f"  {'TOTAL':12s} {int(tn):6d} {tw/tn:9.4f}   (mean weighted by #leaves)")
        print(f"  {'':12s} {'':6s} {tsa/ts:9.4f}   (weighted by test support)")


if __name__ == "__main__":
    main()
