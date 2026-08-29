#!/usr/bin/env python3
"""Check the NEW interleaved_val2 run progress."""
import json, glob, os, time

files = glob.glob("results/interleaved_val2/*.jsonl")
if not files:
    print("no jsonl in interleaved_val2 yet")
    raise SystemExit
path = files[0]
size = os.path.getsize(path)
mtime = os.path.getmtime(path)
print(f"file: {os.path.basename(path)} | size: {size} | mtime age: {(time.time()-mtime)/60:.1f} min")

lines = [json.loads(l) for l in open(path) if l.strip()]
comp = [r for r in lines if r.get("total_cost") is not None]
fail = [r for r in lines if r.get("total_cost") is None]
print(f"records: {len(lines)} | completed: {len(comp)} | failed: {len(fail)}")

if comp:
    from collections import defaultdict
    per = defaultdict(list)
    for r in comp:
        per[r["config"]].append(r["total_cost"])
    print(f"\n{'config':<24}{'n':>3}{'mean':>14}{'cv':>8}")
    for c, costs in sorted(per.items()):
        m = sum(costs)/len(costs)
        sd = (sum((x-m)**2 for x in costs)/len(costs))**0.5
        print(f"{c:<24}{len(costs):>3}{m:>14,.0f}{sd/m:>8.3f}")

print("\nlast 3 records:")
for r in lines[-3:]:
    print(f"  run={r.get('run')} {r.get('config'):<22} cost={r.get('total_cost')} err={str(r.get('error'))[:60]}")
