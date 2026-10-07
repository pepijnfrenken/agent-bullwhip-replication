#!/usr/bin/env python3
"""extract_deltas.py — pull the model's own (order - anchor) deltas out of a run set.

Writes results/jev_deltas.json = {pass_str: {role: [delta per week]}} from the
jev_gate_never traces. Used by the delta_replay control: replaying pass k's deltas
on a different pass's game gives the SAME delta distribution with WRONG state
alignment — if that scores like the model, the model's deltas carry no information
beyond their distribution.

Usage: python3 extract_deltas.py [--file results/jev_protocol/<noisy>.jsonl] [--config jev_gate_never]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROLES = ("retailer", "wholesaler", "distributor", "factory")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="run set (default: the noisy set, by mtime order)")
    ap.add_argument("--config", default="jev_gate_never")
    ap.add_argument("--out", default="results/jev_deltas.json")
    args = ap.parse_args()
    f = args.file
    if f is None:
        files = sorted(glob.glob("results/jev_protocol/*.jsonl"), key=os.path.getmtime)
        f = files[1]  # noisy
    out: dict[str, dict[str, list[int]]] = {}
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("config") != args.config or not r.get("orders"):
            continue
        per_role = {}
        for role in ROLES:
            seq = []
            for t in r["traces"].get(role, []):
                if t.get("anchor") is None:
                    seq.append(0)
                else:
                    seq.append(int(t["order"] - t["anchor"]))
            per_role[role] = seq
        out[str(r["run"])] = per_role
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    n = sum(len(v) for ro in out.values() for v in ro.values())
    print(f"wrote {args.out}: {len(out)} passes, {n} deltas")
    for p in sorted(out)[:3]:
        print(f"  pass {p}: retailer[:12] = {out[p]['retailer'][:12]}")


if __name__ == "__main__":
    main()
