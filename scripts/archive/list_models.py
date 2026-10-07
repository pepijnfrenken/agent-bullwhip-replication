#!/usr/bin/env python3
"""List models from FreeInference and CommandCode using the repo's env keys."""
import os, sys
from pathlib import Path

env = {}
for line in (Path(__file__).resolve().parent / ".env.local").read_text().splitlines():
    if line.startswith(("FREEINFERENCE_API_KEY=", "COMMANDCODE_API_KEY=")):
        k, v = line.split("=", 1)
        env[k] = v.strip()

import requests

def list_models(base, key, label):
    print(f"\n=== {label} ===")
    if not key:
        print("  (no key)")
        return []
    try:
        r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
        for i in sorted(ids):
            print(f"  {i}")
        return ids
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

list_models("https://freeinference.org/v1", env.get("FREEINFERENCE_API_KEY", ""), "FreeInference")
list_models("https://api.commandcode.ai/provider/v1", env.get("COMMANDCODE_API_KEY", ""), "CommandCode")
