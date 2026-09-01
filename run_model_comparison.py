#!/usr/bin/env python3
"""Resumable, quota-aware tool-calling model comparison (FreeInference free tier).

The free tier throttles with a misleading "Invalid or expired API key" 200-error
when the account quota is exhausted. This runner:
  - processes ONE decision at a time per model,
  - waits LONG between decisions (PROBE_CALL_DELAY, default 30s),
  - treats the quota-lie as retryable (the client now does),
  - checkpoints each decision to JSONL immediately (resume-friendly),
  - interleaves models so a quota recovery benefits all of them fairly.

Usage:
    python3 run_model_comparison.py [--models glm-5.3-flash,minimax-m3] [--decisions 6]
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

_env = Path(__file__).resolve().parent / ".env.local"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if _line.startswith("FREEINFERENCE_API_KEY=") and not os.environ.get("FREEINFERENCE_API_KEY"):
            os.environ["FREEINFERENCE_API_KEY"] = _line.split("=", 1)[1].strip()

from agent_bullwhip.agents import ToolAgent, LLMAgentConfig
from agent_bullwhip.client import chat_with_tools, parse_order_prefer_label

OUT = Path("results/toolagent_compare")
CALL_DELAY = float(os.environ.get("PROBE_CALL_DELAY", "30"))
DEFAULT_MODELS = ["glm-5.3-flash", "minimax-m3", "qwen3.6-35b", "deepseek-v4-flash"]

# model IDs may contain '/' (e.g. z-ai/glm-5.3-flash) — sanitize for filenames
def _fname(m: str) -> str:
    return m.replace("/", "__") + ".jsonl"

# 6 representative states spanning the game's regime space
STATES = [
    {"t": 2,  "on_hand": 24, "backlog": 0, "outstanding": 6,  "incoming_last": 8,  "incoming_now": 10, "last_orders": [8, 8, 9]},
    {"t": 6,  "on_hand": 12, "backlog": 3, "outstanding": 10, "incoming_last": 14, "incoming_now": 16, "last_orders": [12, 14, 13]},
    {"t": 10, "on_hand": 2,  "backlog": 9, "outstanding": 4,  "incoming_last": 18, "incoming_now": 20, "last_orders": [16, 18, 17]},
    {"t": 14, "on_hand": 30, "backlog": 0, "outstanding": 15, "incoming_last": 22, "incoming_now": 25, "last_orders": [20, 22, 24]},
    {"t": 20, "on_hand": 6,  "backlog": 6, "outstanding": 18, "incoming_last": 26, "incoming_now": 28, "last_orders": [24, 26, 25]},
    {"t": 30, "on_hand": 40, "backlog": 0, "outstanding": 8,  "incoming_last": 16, "incoming_now": 14, "last_orders": [18, 16, 17]},
]

def build_prompt(ctx: dict) -> str:
    return (
        f"You are the RETAILER in a four-stage beer supply chain. Current week: {ctx['t']}. "
        f"Your state: on-hand {ctx['on_hand']}, backlog {ctx['backlog']}, outstanding {ctx['outstanding']}, "
        f"incoming last week {ctx['incoming_last']}, incoming now {ctx.get('incoming_now', 'not yet known')}, "
        f"recent orders {ctx.get('last_orders', [])}. Holding cost $1/unit/week, backlog $2/unit/week, "
        f"supplier lead time 2 weeks. You may call the run_python tool to help compute your order "
        f"(fit a forecast, simulate, safety stock — whatever you think is best). The tool has "
        f"math, numpy (np), pandas (pd), and statistics (stats) available; scipy is NOT installed. "
        f"You must actually "
        f"call the tool before answering. After you have the result, reply with ONLY an integer: "
        f"the number of units to order this week."
    )

def decision_stats(ctx: dict, text: str, trace: list) -> dict:
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
    return {
        "state": ctx["t"], "text": text[:80], "order": parse_order_prefer_label(text),
        "n_calls": n_calls, "n_err": n_err, "retry_after_error": retry,
        "has_tool_call": n_calls > 0,
        "first_code": (calls[0].get("arguments", {}).get("code", "")[:100] if calls else ""),
        "first_result": (calls[0].get("result", "")[:100] if calls else ""),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--decisions", type=int, default=len(STATES))
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    # load existing checkpoints (resume support)
    done: dict[str, set] = {}
    for m in models:
        f = OUT / _fname(m)
        done[m] = set()
        if f.exists():
            for line in f.read_text().splitlines():
                try:
                    done[m].add(json.loads(line)["state"])
                except Exception:
                    pass
        print(f"{m}: {len(done[m])}/{args.decisions} decisions already done", flush=True)

    # interleave: round-robin over (model, state) pairs not yet done
    pending = [(m, s) for m in models for s in STATES[:args.decisions] if s["t"] not in done[m]]
    print(f"Total pending: {len(pending)}\n", flush=True)
    for idx, (m, s) in enumerate(pending):
        print(f"[{idx+1}/{len(pending)}] {m} t={s['t']} ...", flush=True)
        agent = ToolAgent("retailer", LLMAgentConfig(model=m, tag="cmp"))
        try:
            text, trace = chat_with_tools(
                [{"role": "user", "content": build_prompt(s)}],
                tools=agent._tools(), model=m, temperature=0.3,
                max_tool_rounds=3, force_tool=True, retries=5, timeout=120,
            )
            rec = decision_stats(s, text, trace)
        except Exception as e:
            rec = {"state": s["t"], "error": f"{type(e).__name__}: {str(e)[:120]}"}
        with open(OUT / _fname(m), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"    -> {json.dumps(rec)[:160]}", flush=True)
        if idx < len(pending) - 1:
            time.sleep(CALL_DELAY)  # quota guard between calls

    # summary + ranking
    print("\n=== SUMMARY ===")
    results = []
    for m in models:
        f = OUT / _fname(m)
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        ok = [r for r in recs if "error" not in r]
        if not ok:
            print(f"  {m}: ALL FAILED ({len(recs)} errors)")
            continue
        n = len(ok)
        n_calls = sum(r["n_calls"] for r in ok)
        n_err = sum(r["n_err"] for r in ok)
        n_retry = sum(1 for r in ok if r["retry_after_error"])
        n_err_dec = sum(1 for r in ok if r["n_err"] > 0)
        summary = {
            "model": m, "decisions": n,
            "tool_call_rate": round(sum(r["has_tool_call"] for r in ok) / n, 3),
            "avg_calls_per_decision": round(n_calls / n, 2),
            "code_error_rate": round(n_err / max(1, n_calls), 3),
            "retry_after_error_rate": round(n_retry / max(1, n_err_dec), 3),
            "parse_fail_rate": round(sum(1 for r in ok if r["order"] is None) / n, 3),
            "errors": len(recs) - n,
        }
        results.append(summary)
        print(json.dumps(summary, indent=2))
    if results:
        results.sort(key=lambda r: (r["code_error_rate"], r["parse_fail_rate"]))
        print("\n=== RANKING (by code-error, then parse-fail) ===")
        for r in results:
            print(f"  {r['model']:<22} tool={r['tool_call_rate']} err={r['code_error_rate']} "
                  f"retry={r['retry_after_error_rate']} parse_fail={r['parse_fail_rate']}")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
