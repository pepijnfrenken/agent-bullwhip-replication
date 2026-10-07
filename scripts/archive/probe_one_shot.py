#!/usr/bin/env python3
"""Ultra-light single-decision tool probe — 1 call per model, long gaps.

FreeInference free tier throttles with "Invalid or expired API key" when the
account is exhausted (NOT an auth problem). This probe does ONE decision per
model and sleeps 45s between models so the quota recovers.
"""
import json, os, sys, time
from pathlib import Path

_env = Path(__file__).resolve().parent / ".env.local"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.startswith("FREEINFERENCE_API_KEY=") and not os.environ.get("FREEINFERENCE_API_KEY"):
            os.environ["FREEINFERENCE_API_KEY"] = _line.split("=", 1)[1].strip()

from agent_bullwhip.agents import ToolAgent, LLMAgentConfig
from agent_bullwhip.client import chat_with_tools, parse_order_prefer_label

MODELS = ["glm-5.3-flash", "minimax-m3", "qwen3.6-35b", "deepseek-v4-flash"]
PROMPT = (
    "You are the RETAILER in a four-stage beer supply chain. Current week: 10. "
    "Your state: on-hand 2, backlog 9, outstanding 4, incoming last week 18, "
    "incoming now 20, recent orders [16,18,17]. Holding cost $1/unit/week, backlog "
    "$2/unit/week, supplier lead time 2 weeks. You may call the run_python tool to "
    "help compute your order. You must actually call the tool before answering. "
    "After you have the result, reply with ONLY an integer: units to order."
)

def probe(model: str) -> dict:
    tools = ToolAgent._tools()
    try:
        text, trace = chat_with_tools(
            [{"role": "user", "content": PROMPT}],
            tools=tools, model=model, temperature=0.3,
            max_tool_rounds=3, force_tool=True, retries=5, timeout=120,
        )
    except Exception as e:
        return {"model": model, "error": f"{type(e).__name__}: {str(e)[:150]}"}
    calls = [t for t in trace if isinstance(t, dict)]
    n_calls = len(calls)
    n_err = sum(1 for t in calls if str(t.get("result", "")).startswith("ERROR"))
    retry = False
    seen_err = False
    for t in calls:
        if str(t.get("result", "")).startswith("ERROR"):
            seen_err = True
        elif seen_err:
            retry = True
    order = parse_order_prefer_label(text)
    return {
        "model": model, "text": text[:100], "order": order,
        "n_calls": n_calls, "n_err": n_err, "retry_after_error": retry,
        "has_tool_call": n_calls > 0,
        "first_code": (calls[0].get("arguments", {}).get("code", "")[:120] if calls else ""),
        "first_result": (calls[0].get("result", "")[:120] if calls else ""),
    }

def main():
    results = []
    for i, m in enumerate(MODELS):
        print(f"\n=== {m} (call {i+1}/{len(MODELS)}) ===", flush=True)
        r = probe(m)
        results.append(r)
        print(json.dumps(r, indent=2))
        if i < len(MODELS) - 1:
            print("  ... sleeping 45s for quota recovery ...", flush=True)
            time.sleep(45)
    out = Path("results/toolagent_probe/one_shot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
