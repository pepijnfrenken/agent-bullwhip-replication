#!/usr/bin/env python3
"""Progress check for the crossover run (chaotic + wild arms)."""
import json, glob, os, time
from collections import defaultdict

for arm in ["chaotic_leakfree", "wild_leakfree"]:
    files = glob.glob(f"results/{arm}/*.jsonl")
    if not files:
        print(f"{arm}: no jsonl yet")
        continue
    path = files[0]
    age = (time.time() - os.path.getmtime(path)) / 60
    lines = [json.loads(l) for l in open(path) if l.strip()]
    comp = [r for r in lines if r.get("total_cost") is not None]
    fail = [r for r in lines if r.get("total_cost") is None]
    print(f"=== {arm} | mtime age {age:.1f} min | records {len(lines)} completed {len(comp)} failed {len(fail)} ===")
    per = defaultdict(list)
    for r in comp:
        per[r["config"]].append(r["total_cost"])
    for c, costs in sorted(per.items(), key=lambda kv: sum(kv[1])/len(kv[1])):
        m = sum(costs)/len(costs)
        print(f"  {c:<22} n={len(costs):>2} mean={m:>14,.0f}")
    if fail:
        from collections import Counter
        print("  failures:", dict(Counter(r["config"] for r in fail)))
