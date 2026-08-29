#!/usr/bin/env python3
"""Final trace pass: where do the blowups cluster, and what state triggers them?"""
import json, statistics, collections

CONFIGS = ["kb_introspect_gated", "kb_system_gated", "kb_pointer_gated", "kb_introspect", "kb_general_introspect"]

def load(cfg):
    runs = []
    for line in open(f"results/deepseek-v4-flash-{cfg}.jsonl"):
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    return runs

def order_up_to_order(ctx):
    last = ctx.get("last_orders") or []
    if not last:
        return 0
    smoothed = last[0]
    for v in last[1:]:
        smoothed = 0.5 * smoothed + 0.5 * v
    target = 3.0 * smoothed + ctx.get("backlog", 0)
    position = ctx.get("on_hand", 0) + ctx.get("outstanding", 0) - ctx.get("backlog", 0)
    return max(0, int(round(target - position)))

print("=" * 80)
print("A. Blowup concentration: how many runs carry the |dev|>=50 events?")
print("=" * 80)
for cfg in CONFIGS:
    runs = load(cfg)
    per_run = []
    for d in runs:
        n = 0
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                dev = t.get("order_used") - order_up_to_order(t["ctx"])
                if abs(dev) >= 50:
                    n += 1
        per_run.append(n)
    n_with = sum(1 for n in per_run if n > 0)
    worst_run = max(per_run)
    print(f"{cfg:<24} runs_with_blowups={n_with:>3}/{len(runs)}  worst_run={worst_run:>3}  "
          f"median={statistics.median(per_run):.0f}")

print()
print("=" * 80)
print("B. State at under-order blowups (order_used=0, dev<=-50): what does the model see?")
print("=" * 80)
for cfg in ["kb_introspect_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    states = []
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                dev = t.get("order_used") - order_up_to_order(t["ctx"])
                if t.get("order_used") == 0 and dev <= -50:
                    ctx = t["ctx"]
                    states.append((role, ctx.get("backlog", 0), ctx.get("outstanding", 0),
                                   ctx.get("incoming_now", 0), ctx.get("incoming_last", 0),
                                   ctx.get("on_hand", 0)))
    n = len(states)
    if not n:
        continue
    print(f"--- {cfg}: n={n} under-order blowups (order=0) ---")
    roles = collections.Counter(s[0] for s in states)
    print("  by role:", dict(roles))
    print(f"  median backlog={statistics.median(s[1] for s in states):.0f}  "
          f"median outstanding={statistics.median(s[2] for s in states):.0f}  "
          f"median incoming_now={statistics.median(s[3] for s in states):.0f}  "
          f"median incoming_last={statistics.median(s[4] for s in states):.0f}")
    demand_drop = sum(1 for s in states if s[3] < s[4])
    print(f"  incoming_now < incoming_last (demand step-down): {demand_drop}/{n} ({100*demand_drop/n:.0f}%)")
    out_over_bl = sum(1 for s in states if s[2] >= 2 * s[1])
    print(f"  outstanding >= 2x backlog: {out_over_bl}/{n} ({100*out_over_bl/n:.0f}%)")

print()
print("=" * 80)
print("C. What the model says at the exact moment of a catastrophic under-order")
print("=" * 80)
shown = 0
for cfg in ["kb_introspect_gated", "kb_pointer_gated", "kb_system_gated"]:
    runs = load(cfg)
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                dev = t.get("order_used") - order_up_to_order(t["ctx"])
                if t.get("order_used") == 0 and dev <= -100 and shown < 4:
                    th = (t.get("thinking") or t.get("reasoning") or "")[:300]
                    print(f"[{cfg} {role} t={t['ctx'].get('t')}] dev={dev} conf={t.get('confidence')}")
                    print(f"   {th}")
                    print()
                    shown += 1
