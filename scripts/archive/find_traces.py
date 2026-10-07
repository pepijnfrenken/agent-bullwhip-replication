#!/usr/bin/env python3
"""Find a trace with non-empty thinking/reasoning across the introspect runs."""
import json

found = 0
for line in open("results/deepseek-v4-flash-introspect.jsonl"):
    d = json.loads(line)
    for role, traces in (d.get("traces") or {}).items():
        for t in traces:
            th = t.get("thinking") or ""
            re_ = t.get("reasoning") or ""
            if th or re_:
                print(f"--- run {d['run']} {role} t={t['ctx'].get('t')} conf={t.get('confidence')} order={t.get('order')} used={t.get('order_used')}")
                if th:
                    print(f"  THINKING: {th[:200]}")
                if re_:
                    print(f"  REASONING: {re_[:200]}")
                found += 1
                if found >= 5:
                    raise SystemExit
print("total with traces found:", found)
