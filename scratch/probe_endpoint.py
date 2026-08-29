#!/usr/bin/env python3
"""Probe the FreeInference endpoint directly (no client retries)."""
import os, time, requests

KEY = os.environ.get("FREEINFERENCE_API_KEY", "")
if not KEY:
    p = "/home/pino/projects/agent-bullwhip-replication/.env.local"
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("FREEINFERENCE_API_KEY="):
                KEY = line.split("=", 1)[1].strip()
                break

BASE = os.environ.get("FREEINFERENCE_BASE", "https://freeinference.org/v1")
print("base:", BASE, "key set:", bool(KEY))

# 1) models endpoint (light)
try:
    r = requests.get(f"{BASE}/models", headers={"Authorization": f"Bearer {KEY}"}, timeout=15)
    print("GET /models:", r.status_code, str(r.text)[:80])
except Exception as e:
    print("GET /models FAIL:", str(e)[:120])

# 2) a tiny chat completion (the real test)
try:
    t0 = time.time()
    r = requests.post(
        f"{BASE}/chat/completions",
        headers={"Authorization": f"Bearer {KEY}"},
        json={"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Reply with just 42"}],
              "temperature": 0.0, "max_tokens": 8},
        timeout=20,
    )
    print(f"POST chat: {r.status_code} in {time.time()-t0:.1f}s | {str(r.text)[:150]}")
except Exception as e:
    print("POST chat FAIL:", str(e)[:150])
