#!/usr/bin/env python3
"""analyze_jev_traces.py — audited trace analysis for the Jev configs.

Does the gated config actually defer to its formula when confidence is low, and
what do the *ungated* decisions contribute? This is a WITHIN-RUN analysis: each
decision trace carries its own `anchor` value (the game is closed-loop, so
comparing order sequences across two different runs is invalid — different orders
change every later week's state).

Reports per config:
  - gate rate + confidence distribution
  - gated decisions where order != anchor  (should be 0; anything else is a bug)
  - ungated decisions: how many differ from the anchor, and the delta distribution
    (bounded by the margin clamp, so this is the model's whole contribution)

Usage: python3 analyze_jev_traces.py --dir results/jev_controls
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from statistics import mean

ROLES = ("retailer", "wholesaler", "distributor", "factory")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/jev_controls")
    args = ap.parse_args()

    for f in sorted(glob.glob(f"{args.dir}/*.jsonl")):
        for line in open(f):
            rec = json.loads(line)
            if not rec.get("orders"):
                continue
            traces = [t for role in ROLES for t in (rec.get("traces") or {}).get(role, [])]
            if not traces or "anchor" not in (traces[0] or {}):
                print(f"{rec['config']:>18} run {rec['run']}: cost {rec['total_cost']:.0f}  "
                      f"(no anchor in traces — pre-patch record)")
                continue
            gated = [t for t in traces if t.get("gate_fired")]
            open_ = [t for t in traces if not t.get("gate_fired")]
            confs = sorted(float(t["confidence"]) for t in traces if t.get("confidence") is not None)
            violations = [t for t in gated if t.get("order") != t.get("anchor")]
            diffs = [t for t in open_ if t.get("order") != t.get("anchor")]
            deltas = sorted(t["order"] - t["anchor"] for t in diffs)
            print(f"{rec['config']:>18} run {rec['run']}: cost {rec['total_cost']:.0f}  "
                  f"decisions {len(traces)}  gate {len(gated)}/{len(traces)} ({100*len(gated)/len(traces):.0f}%)")
            if confs:
                print(f"{'':>18}   confidence min {confs[0]:.2f} / median {confs[len(confs)//2]:.2f} / max {confs[-1]:.2f}")
            print(f"{'':>18}   gated decisions not equal to the anchor: {len(violations)}"
                  + ("  <-- BUG" if violations else "  (clean)"))
            print(f"{'':>18}   ungated decisions: {len(open_)}  differ from anchor: {len(diffs)}  "
                  + (f"deltas [{deltas[0]:+d}..{deltas[-1]:+d}] mean {mean(deltas):+.1f}  "
                     f"signs {dict(Counter('+' if d > 0 else '-' if d < 0 else '0' for d in deltas))}" if deltas else ""))


if __name__ == "__main__":
    main()
