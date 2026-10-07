#!/usr/bin/env python3
"""Inspect Wave 2 introspect traces + verify costs."""
import json
import statistics

costs = []
for line in open("results/deepseek-v4-flash-introspect.jsonl"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("total_cost"):
        costs.append(d["total_cost"])
mean = sum(costs) / len(costs)
print(f"introspect n={len(costs)} mean={mean:,.0f} cv={statistics.stdev(costs)/mean:.3f}")

with open("results/deepseek-v4-flash-introspect.jsonl") as f:
    first = json.loads(f.readline())
r = first.get("traces", {}).get("retailer", [])
print(f"run0 retailer traces: {len(r)}")
if r:
    t = r[3] if len(r) > 3 else r[0]
    print("ctx:", {k: t["ctx"][k] for k in ("t", "on_hand", "backlog", "outstanding", "incoming_last", "incoming_now")})
    print("order:", t.get("order"), "conf:", t.get("confidence"), "used:", t.get("order_used"))
    print("thinking:", (t.get("thinking") or "")[:180])
