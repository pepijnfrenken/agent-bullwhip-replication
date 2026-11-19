#!/usr/bin/env python3
"""Verify a re-frozen wave-8 rung gate file against the pre-registered values.

Why: the rebuild of the 1.7B and 4B rungs must not train on a gate that drifted.
The expected values below were written into docs/waves/W8-RERUN-LOG.md BEFORE the
preflight was submitted (commit history is the timestamp). This script is the
enforcement: it exits non-zero on any mismatch so a dependent Slurm job never
starts.

Stdlib only, so it runs on the cluster's module python with no venv.

Usage:
    python3 scripts/verify_rung_gates.py --gates results/wave8/gates_4b.json --rung 4b
    python3 scripts/verify_rung_gates.py --gates results/wave8/gates_1.7b.json --gates2 ... --json out.json

Tolerance: 1e-3 relative. Values come from RECONCILIATION-2026-10-22.md section 7,
which states them to gate-file precision; a real seed or readout drift moves a
control statistic by far more than 0.1%.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

REL_TOL = 1e-3

EXPECTED = {
    "1.7b": {"H_lo": 9.6527e-05, "H_hi": 1.6313, "I_hi0": 1.4858, "m_hi": 0.1487},
    "4b": {"H_lo": 0.038026, "H_hi": 1.910513, "I_hi0": 0.676551, "m_hi": 0.1487},
}

# gate files have carried these names for the four control statistics
ALIASES = {
    "H_lo": ("H_lo", "H_LO", "h_lo"),
    "H_hi": ("H_hi", "H_HI", "h_hi"),
    "I_hi0": ("I_hi0", "I_HI0", "i_hi0"),
    "m_hi": ("m_hi", "M_HI"),
}


def find_value(doc: dict, key: str):
    """Search the gate document (nested dicts) for the first alias that exists."""
    for alias in ALIASES[key]:
        stack = [doc]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if alias in cur and isinstance(cur[alias], (int, float)):
                    return cur[alias], alias
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True, action="append", help="gate json path (repeatable)")
    ap.add_argument("--rung", required=True, action="append", help="rung name per --gates (repeatable)")
    ap.add_argument("--json", help="write a machine-readable report here")
    ap.add_argument("--tol", type=float, default=REL_TOL)
    args = ap.parse_args()

    if len(args.gates) != len(args.rung):
        print("FATAL: give one --rung per --gates", file=sys.stderr)
        return 2

    report = {"tolerance_rel": args.tol, "checks": [], "ok": True}
    for path, rung in zip(args.gates, args.rung):
        key = rung.lower()
        if key not in EXPECTED:
            print(f"FATAL: no registered expectation for rung '{rung}'", file=sys.stderr)
            return 2
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except FileNotFoundError:
            print(f"FAIL {rung}: gate file missing at {path}", file=sys.stderr)
            report["ok"] = False
            report["checks"].append({"rung": rung, "path": path, "error": "missing"})
            continue
        except json.JSONDecodeError as exc:
            print(f"FAIL {rung}: {path} is not valid json ({exc})", file=sys.stderr)
            report["ok"] = False
            report["checks"].append({"rung": rung, "path": path, "error": f"json: {exc}"})
            continue

        for field, exp in EXPECTED[key].items():
            got, alias = find_value(doc, field)
            if got is None:
                ok = False
                rel = None
            else:
                rel = abs(got - exp) / abs(exp) if exp else math.inf
                ok = rel <= args.tol
            report["ok"] = report["ok"] and ok
            report["checks"].append({
                "rung": rung, "path": path, "field": field, "alias": alias,
                "expected": exp, "found": got, "rel_diff": rel, "ok": ok,
            })
            status = "ok  " if ok else "FAIL"
            print(f"{status} {rung:5s} {field:6s} expected {exp!r:>12} found {got!r:>12} "
                  f"rel {('%.2e' % rel) if rel is not None else 'n/a'}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"report -> {args.json}")

    print("RESULT:", "PREDICTION HOLDS" if report["ok"] else "MISMATCH — do not train")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
