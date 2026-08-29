#!/usr/bin/env python3
"""Post-mortem on kb_pointer_verbal: did the gate fire, on what, and did it help?"""
import json, statistics, collections

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

for cfg in ["kb_pointer_gated", "kb_pointer_verbal"]:
    runs = load(cfg)
    costs = [d["total_cost"] for d in runs]
    n_verbal = n_conf_gate = n_order0_bl = 0
    n_good_overrides = 0   # gate fired, anchor order > 0 (probably helped)
    n_bad_overrides = 0    # gate fired, anchor order == 0 too (no change) or model was right
    override_devs = []
    failures_total = 0
    calls_total = 0
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                if t.get("gated_verbal"):
                    n_verbal += 1
                    dev = t.get("order_used") - order_up_to_order(t["ctx"])
                    override_devs.append(dev)
                    if t.get("order_used", 0) > 0:
                        n_good_overrides += 1
                    else:
                        n_bad_overrides += 1
                if t.get("gated") and not t.get("gated_verbal"):
                    n_conf_gate += 1
                if t.get("order_used") == 0 and (t.get("ctx") or {}).get("backlog", 0) > 0:
                    n_order0_bl += 1
        failures_total += sum(d.get("failures", {}).values())
        calls_total += sum(d.get("calls", {}).values())
    print(f"=== {cfg} ===")
    print(f"  mean_cost={statistics.mean(costs):,.0f}  cv={statistics.stdev(costs)/statistics.mean(costs):.3f}")
    print(f"  verbal-gate fires: {n_verbal} | conf-gate fires (non-verbal): {n_conf_gate}")
    print(f"  order-0-with-backlog (after all gates): {n_order0_bl}")
    print(f"  verbal overrides that produced order>0: {n_good_overrides} | order==0 anyway: {n_bad_overrides}")
    if override_devs:
        print(f"  mean |dev from floor| of overridden decisions: {statistics.mean(abs(x) for x in override_devs):.1f}")
    print(f"  failures={failures_total} calls={calls_total} failure_rate={failures_total/max(calls_total,1):.3f}")
    print()

# What does the model say in the overridden cases? Sample 5 verbal-gated traces
print("=== SAMPLE verbal-gated traces (kb_pointer_verbal) ===")
runs = load("kb_pointer_verbal")
shown = 0
for d in runs:
    for role, trs in (d.get("traces") or {}).items():
        for t in trs:
            if t.get("gated_verbal") and shown < 6:
                ctx = t["ctx"]
                opt = order_up_to_order(ctx)
                print(f"[{role} t={ctx.get('t')}] backlog={ctx.get('backlog')} outstanding={ctx.get('outstanding')} "
                      f"incoming={ctx.get('incoming_now')} | model_order={t.get('order')} used={t.get('order_used')} floor={opt}")
                print(f"   think: {(t.get('thinking') or '')[:150]}")
                print()
                shown += 1
