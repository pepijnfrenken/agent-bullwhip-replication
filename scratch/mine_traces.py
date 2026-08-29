#!/usr/bin/env python3
"""Mine thinking/reasoning traces across all introspect-family configs.

Questions:
1. How often do traces carry thinking/reasoning? (per config, per role)
2. When does the gate fire (gated=True)? What does confidence look like?
3. When does order != order_used (output-format failures)?
4. Qualitative: best vs worst decisions with their reasoning.
"""
import json, statistics, collections, re

CONFIGS = [
    "introspect", "kb_introspect", "kb_system_introspect", "kb_pointer_introspect",
    "kb_general_introspect", "kb_introspect_gated", "kb_system_gated", "kb_pointer_gated",
]

def load(cfg):
    runs = []
    try:
        for line in open(f"results/deepseek-v4-flash-{cfg}.jsonl"):
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    except FileNotFoundError:
        return None
    return runs

print("=" * 80)
print("1. TRACE PRESENCE + GATE + FORMAT FAILURES per config")
print("=" * 80)
hdr = f"{'config':<24}{'runs':>5}{'traces':>8}{'think%':>7}{'reasons%':>9}{'apiR%':>7}{'gated%':>7}{'mismatch%':>10}"
print(hdr)
for cfg in CONFIGS:
    runs = load(cfg)
    if not runs:
        continue
    n_traces = n_think = n_reason = n_apir = n_gated = n_mismatch = 0
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                n_traces += 1
                if (t.get("thinking") or "").strip():
                    n_think += 1
                if (t.get("reasoning") or "").strip():
                    n_reason += 1
                if (t.get("api_reasoning") or "").strip():
                    n_apir += 1
                if t.get("gated"):
                    n_gated += 1
                if t.get("order") != t.get("order_used"):
                    n_mismatch += 1
    pct = lambda n: f"{100*n/n_traces:5.1f}%" if n_traces else "  n/a"
    print(f"{cfg:<24}{len(runs):>5}{n_traces:>8}{pct(n_think):>7}{pct(n_reason):>9}{pct(n_apir):>7}{pct(n_gated):>7}{pct(n_mismatch):>10}")

print()
print("=" * 80)
print("2. GATED EVENTS: when does the gate fire, and does it save cost?")
print("=" * 80)
for cfg in ["kb_introspect_gated", "kb_system_gated", "kb_pointer_gated"]:
    runs = load(cfg)
    if not runs:
        continue
    gate_costs = []
    n_gate = 0
    n_gate_save = 0
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                if t.get("gated"):
                    n_gate += 1
                    # was the gated (fallback) order different from what the model wanted?
                    if t.get("order") != t.get("order_used"):
                        n_gate_save += 1
    print(f"{cfg}: gated events {n_gate}, of which fallback changed the order: {n_gate_save}")

print()
print("=" * 80)
print("3. ORDER=0 / BACKLOG>0 pathological decisions (model forgets to order)")
print("=" * 80)
for cfg in ["kb_pointer_gated", "kb_system_gated", "kb_introspect_gated", "kb_introspect", "kb_general_introspect"]:
    runs = load(cfg)
    if not runs:
        continue
    n_bad = 0
    examples = []
    for d in runs:
        for role, trs in (d.get("traces") or {}).items():
            for t in trs:
                ctx = t.get("ctx") or {}
                if t.get("order_used") == 0 and ctx.get("backlog", 0) > 0:
                    n_bad += 1
                    if len(examples) < 3:
                        examples.append((role, ctx.get("t"), ctx.get("backlog"), ctx.get("on_hand"),
                                         ctx.get("outstanding"), (t.get("thinking") or "")[:140]))
    print(f"{cfg}: order=0 with backlog>0: {n_bad} occurrences")
    for role, tt, bl, oh, out, th in examples:
        print(f"   [{role} t={tt}] backlog={bl} on_hand={oh} outstanding={out} :: {th}")
