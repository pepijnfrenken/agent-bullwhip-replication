#!/usr/bin/env python3
"""Diagnose interleaved run failures."""
import json, glob
from collections import Counter

path = glob.glob("results/interleaved_val/*.jsonl")[0]
lines = [json.loads(l) for l in open(path) if l.strip()]
failed = [r for r in lines if r.get("total_cost") is None]
completed = [r for r in lines if r.get("total_cost") is not None]

print(f"total: {len(lines)} completed: {len(completed)} failed: {len(failed)}")
print("\n=== failed by config ===")
print(Counter(r["config"] for r in failed))
print("\n=== failed by run (pass) ===")
print(Counter(r["run"] for r in failed))
print("\n=== sample errors ===")
for r in failed[:8]:
    print(f"  run {r['run']} {r['config']}: {str(r.get('error'))[:120]}")

print("\n=== completed by config ===")
print(Counter(r["config"] for r in completed))

print("\n=== failure rate over time (first 20 records) ===")
for i, r in enumerate(lines[:20]):
    mark = "FAIL" if r.get("total_cost") is None else "ok  "
    print(f"  {i:>2} {mark} run={r['run']} {r['config']}")
