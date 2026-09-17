#!/usr/bin/env python3
"""Quota-friendly tool-calling probe for candidate models (FreeInference free tier).

Runs a SMALL number of decisions per model (default 3) with a LONG delay between
calls, because the free tier throttles rapid calls with a misleading
"Invalid or expired API key" error (it's actually a rate limit — observed
2026-08-31).

For each model, measures:
  - tool_call_rate      : fraction of decisions where the model called the tool
  - code_error_rate     : fraction of tool calls whose exec returned an ERROR
  - retry_after_error   : fraction of errored decisions where a SECOND call was made
  - parse_fail_rate     : fraction with no parseable integer
  - cost (approx, from a mini horizon)

Usage:
    python3 probe_tool_models_light.py [--models glm-5.3-flash,minimax-m3] [--n 3]
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

# ensure the API key is available to the client even when not in env
_env_local = Path(__file__).resolve().parent / ".env.local"
if _env_local.exists():
    for _line in _env_local.read_text().splitlines():
        if _line.startswith("FREEINFERENCE_API_KEY=") and not os.environ.get("FREEINFERENCE_API_KEY"):
            os.environ["FREEINFERENCE_API_KEY"] = _line.split("=", 1)[1].strip()

from agent_bullwhip.agents import ToolAgent, LLMAgentConfig
from agent_bullwhip.client import chat_with_tools
from agent_bullwhip.client import parse_order_prefer_label

OUT = Path("results/toolagent_probe")
DEFAULT_MODELS = ["glm-5.3-flash", "minimax-m3", "qwen3.6-35b", "deepseek-v4-flash"]
# long delay between calls to stay under the free-tier quota
CALL_DELAY = float(os.environ.get("PROBE_CALL_DELAY", "12"))

def build_prompt_for(t: int, ctx: dict) -> str:
    """Minimal single-decision prompt (same shape as ToolAgent.decide)."""
    return (
        f"You are the RETAILER in a four-stage beer supply chain. "
        f"Current week: {t}. Your state: on-hand {ctx['on_hand']}, backlog {ctx['backlog']}, "
        f"outstanding {ctx['outstanding']}, incoming last week {ctx['incoming_last']}, "
        f"incoming now {ctx.get('incoming_now', 'not yet known')}, recent orders {ctx.get('last_orders', [])}. "
        f"Holding cost $1/unit/week, backlog $2/unit/week, supplier lead time 2 weeks. "
        f"You may call the run_python tool to help compute your order (fit a forecast, "
        f"simulate, safety stock — whatever you think is best). You must actually call "
        f"the tool before answering. After you have the result, reply with ONLY an "
        f"integer: the number of units to order this week."
    )

def probe_model(model: str, n: int) -> dict:
    agent = ToolAgent("retailer", LLMAgentConfig(model=model, tag="probe"))
    tools = agent._tools()
    # 3 representative states: normal, backlog-heavy, high-pipeline
    states = [
        {"t": 10, "on_hand": 20, "backlog": 0, "outstanding": 8, "incoming_last": 12, "incoming_now": 14, "last_orders": [10, 12, 11]},
        {"t": 15, "on_hand": 2, "backlog": 9, "outstanding": 4, "incoming_last": 18, "incoming_now": 20, "last_orders": [16, 18, 17]},
        {"t": 20, "on_hand": 30, "backlog": 0, "outstanding": 15, "incoming_last": 22, "incoming_now": 25, "last_orders": [20, 22, 24]},
    ][:n]
    stats = []
    for ctx in states:
        time.sleep(CALL_DELAY)  # quota guard
        try:
            text, trace = chat_with_tools(
                [{"role": "user", "content": build_prompt_for(ctx["t"], ctx)}],
                tools=tools,
                model=model,
                temperature=0.3,
                max_tool_rounds=3,
                exec_globals={},
                force_tool=True,
                retries=5,
                timeout=120,
            )
        except Exception as e:
            stats.append({"state": ctx["t"], "error": str(e)[:120]})
            continue
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
        stats.append({
            "state": ctx["t"], "text": text[:80], "order": order,
            "n_calls": n_calls, "n_err": n_err, "retry_after_error": retry,
            "has_tool_call": n_calls > 0,
        })
        print(f"    [t={ctx['t']}] calls={n_calls} err={n_err} retry={retry} order={order} text={text[:50]!r}", flush=True)
    n_ok = len([s for s in stats if "error" not in s])
    if n_ok == 0:
        return {"model": model, "error": "all decisions failed", "detail": stats}
    ok_stats = [s for s in stats if "error" not in s]
    return {
        "model": model,
        "decisions": n_ok,
        "tool_call_rate": round(sum(s["has_tool_call"] for s in ok_stats) / n_ok, 3),
        "avg_calls_per_decision": round(sum(s["n_calls"] for s in ok_stats) / n_ok, 2),
        "code_error_rate": round(sum(s["n_err"] for s in ok_stats) / max(1, sum(s["n_calls"] for s in ok_stats)), 3),
        "retry_after_error_rate": round(sum(s["retry_after_error"] for s in ok_stats) / max(1, sum(1 for s in ok_stats if s["n_err"] > 0)), 3),
        "parse_fail_rate": round(sum(1 for s in ok_stats if s["order"] is None) / n_ok, 3),
        "detail": stats,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=str(OUT / "probe_results.json"))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for m in models:
        print(f"\n=== probing {m} (n={args.n}) ===", flush=True)
        try:
            r = probe_model(m, args.n)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "detail"}, indent=2))
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({"model": m, "error": str(e)})
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.out}")
    ok = [r for r in results if "code_error_rate" in r]
    if ok:
        ok.sort(key=lambda r: (r.get("code_error_rate") or 1, r.get("parse_fail_rate") or 1))
        print("\n=== RANKING (by code-error rate, then parse-fail) ===")
        for r in ok:
            print(f"  {r['model']:<22} tool_call={r.get('tool_call_rate')}  "
                  f"err={r.get('code_error_rate')}  retry={r.get('retry_after_error_rate')}  "
                  f"parse_fail={r.get('parse_fail_rate')}")
    else:
        print("\nNo successful probes to rank (all errored).")

if __name__ == "__main__":
    main()
