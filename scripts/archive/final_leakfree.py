#!/usr/bin/env python3
"""Leak-free noisy results: the final fair LLM-vs-formula verdict."""
import json, glob
from collections import defaultdict

files = glob.glob("results/noisy_leakfree/*.jsonl")
if not files:
    print("no jsonl")
    raise SystemExit
path = files[0]
lines = [json.loads(l) for l in open(path) if l.strip()]
comp = [r for r in lines if r.get("total_cost") is not None]
fail = [r for r in lines if r.get("total_cost") is None]
print(f"records: {len(lines)} | completed: {len(comp)} | failed: {len(fail)}")
if fail:
    from collections import Counter
    print("failures by config:", dict(Counter(r["config"] for r in fail)))

per = defaultdict(list)
for r in comp:
    per[r["config"]].append(r["total_cost"])
print(f"\n{'config':<24}{'n':>3}{'mean':>14}{'std':>12}{'cv':>8}{'min':>12}{'max':>14}")
for c in sorted(per, key=lambda c: sum(per[c])/len(per[c])):
    costs = per[c]
    m = sum(costs)/len(costs)
    sd = (sum((x-m)**2 for x in costs)/len(costs))**0.5
    print(f"{c:<24}{len(costs):>3}{m:>14,.0f}{sd:>12,.0f}{sd/m:>8.3f}{min(costs):>12,.0f}{max(costs):>14,.0f}")

print("\n=== per-pass: kb_system_gated vs order_up_to (fair fight) ===")
by_pass = defaultdict(dict)
for r in comp:
    by_pass[r["run"]][r["config"]] = r["total_cost"]
for p in sorted(by_pass):
    row = by_pass[p]
    if "kb_system_gated" in row and "order_up_to" in row:
        d = row["kb_system_gated"] - row["order_up_to"]
        print(f"  pass {p:>2}: gated={row['kb_system_gated']:>9,.0f} floor={row['order_up_to']:>9,.0f} delta={d:>+9,.0f} ({(d/row['order_up_to'])*100:+.1f}%)")
print("\n=== per-pass: kb_pointer_verbal vs order_up_to ===")
for p in sorted(by_pass):
    row = by_pass[p]
    if "kb_pointer_verbal" in row and "order_up_to" in row:
        d = row["kb_pointer_verbal"] - row["order_up_to"]
        print(f"  pass {p:>2}: verbal={row['kb_pointer_verbal']:>9,.0f} floor={row['order_up_to']:>9,.0f} delta={d:>+9,.0f} ({(d/row['order_up_to'])*100:+.1f}%)")
