#!/usr/bin/env python3
"""Live progress on the interleaved validation run."""
import json, glob, os, time

files = glob.glob("results/interleaved_val/*.jsonl")
if not files:
    print("no jsonl yet")
    raise SystemExit

path = files[0]
lines = [json.loads(l) for l in open(path) if l.strip()]
completed = [r for r in lines if r.get("total_cost") is not None]
failed = [r for r in lines if r.get("total_cost") is None]
print(f"file: {os.path.basename(path)}")
print(f"total records: {len(lines)} | completed: {len(completed)} | failed: {len(failed)}")

if completed:
    from collections import defaultdict
    per = defaultdict(list)
    for r in completed:
        per[r["config"]].append(r["total_cost"])
    print(f"\n{'config':<24}{'n':>3}{'mean':>12}{'cv':>8}")
    for c, costs in sorted(per.items()):
        m = sum(costs)/len(costs)
        sd = (sum((x-m)**2 for x in costs)/len(costs))**0.5
        print(f"{c:<24}{len(costs):>3}{m:>12,.0f}{sd/m:>8.3f}")
    # per-pass window drift: cost by run index (pass number)
    print("\n=== cost by pass (run index) ===")
    by_pass = defaultdict(list)
    for r in completed:
        by_pass[r["run"]].append((r["config"], r["total_cost"]))
    for p in sorted(by_pass):
        items = by_pass[p]
        costs = [c for _, c in items]
        print(f"pass {p:>2}: {len(items):>2} configs | cost range {min(costs):>10,.0f} - {max(costs):>10,.0f}")
