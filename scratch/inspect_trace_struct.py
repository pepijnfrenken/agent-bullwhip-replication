#!/usr/bin/env python3
"""Inspect trace structure across configs."""
import json

for cfg in ["kb_system_gated", "kb_introspect_gated", "kb_pointer_gated", "kb_general_introspect"]:
    path = f"results/deepseek-v4-flash-{cfg}.jsonl"
    try:
        d = json.loads(open(path).readline())
    except FileNotFoundError:
        print(f"{cfg}: MISSING")
        continue
    roles = list((d.get("traces") or {}).keys())
    print(f"=== {cfg} ===")
    print("top keys:", list(d.keys()))
    print("roles:", roles)
    tr = d["traces"].get("retailer", [])
    print(f"retailer traces: {len(tr)}")
    if tr:
        t = tr[5] if len(tr) > 5 else tr[0]
        print("trace keys:", list(t.keys()))
        for k, v in t.items():
            if k != "ctx":
                print(f"  {k}: {str(v)[:120]}")
        print("ctx:", {k: v for k, v in t["ctx"].items()})
    print()
