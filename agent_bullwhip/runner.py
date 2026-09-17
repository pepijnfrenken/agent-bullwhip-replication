"""Run N replications of a config and dump JSONL + summary.

Usage:
    python -m agent_bullwhip.runner --config baseline --runs 30 --model deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from .agents import LLMAgent, LLMAgentConfig, MirrorAgent, OrderUpToAgent, ToolAgent
from .client import list_models
from .engine import ROLES, SimConfig, make_demand, run_game
from .metrics import compute_metrics

CONFIGS: dict[str, dict] = {
    # ---- the paper's own levers (Wave 1) ----
    "baseline": {},  # default decentralized LLM (replication anchor)
    "budget": {"guardrail_ratio": 2.0},  # paper Table 1 "budget" policy (order cap)
    "prompt_weighted": {"prompt_variant": "weighted"},  # paper §3.4 reframe
    "voting5": {"voting": 5},  # paper §4.3 repeated sampling (negative result check)
    # ---- our extension (Wave 2) ----
    "anchor": {"anchor_margin": 6},
    "combined": {"guardrail_ratio": 2.0, "anchor_margin": 6, "prompt_variant": "weighted"},
    # ---- Wave 2: knowledge base + introspection (the iteration loop) ----
    "kb": {"kb": True},                                    # playbook injected, no trace
    "introspect": {"introspect": True},                    # reason + confidence, logged
    "kb_introspect": {"kb": True, "introspect": True},     # both (the full upgrade)
    "kb_introspect_gated": {"kb": True, "introspect": True, "conf_threshold": 0.5,
                            "anchor_margin": 6},           # self-gate: low conf -> anchor
    # ---- Wave 2b: KB placement A/B (system message vs inline user message) ----
    "kb_system": {"kb": True, "kb_placement": "system"},                 # KB in system msg
    "kb_system_introspect": {"kb": True, "kb_placement": "system",
                             "introspect": True},                        # system + trace
    "kb_system_gated": {"kb": True, "kb_placement": "system",
                        "introspect": True, "conf_threshold": 0.5,
                        "anchor_margin": 6},              # system + self-gate
    # ---- Wave 2c: pointer-only KB (no contents injected — self-directed) ----
    "kb_pointer": {"kb": True, "kb_placement": "pointer"},               # guide-only, no contents
    "kb_pointer_introspect": {"kb": True, "kb_placement": "pointer",
                              "introspect": True},                       # pointer + trace
    "kb_pointer_gated": {"kb": True, "kb_placement": "pointer",
                         "introspect": True, "conf_threshold": 0.5,
                         "anchor_margin": 6},            # pointer + self-gate
    # ---- Wave 2e: verbal-consistency gate (trace-driven ablation) ----
    # pointer + self-gate + verbal-consistency gate: if the model says "cover the
    # backlog / order-up-to" but emits order=0 with a backlog, override to anchor.
    # Directly targets the dominant failure mode found in trace mining.
    "kb_pointer_verbal": {"kb": True, "kb_placement": "pointer",
                          "introspect": True, "conf_threshold": 0.5,
                          "anchor_margin": 6, "consistency_gate": True},
    # ---- Wave 2d: generalizable (domain-agnostic) KB — transfer test ----
    "kb_general": {"kb": True, "kb_file": "GENERAL_KB.md"},              # general principles only
    "kb_general_introspect": {"kb": True, "kb_file": "GENERAL_KB.md",
                              "introspect": True},                       # general + trace
    "kb_general_gated": {"kb": True, "kb_file": "GENERAL_KB.md",
                         "introspect": True, "conf_threshold": 0.5,
                         "anchor_margin": 6},            # general + self-gate
    # ---- deterministic baselines ----
    "mirror": {"_baseline": "mirror"},
    "order_up_to": {"_baseline": "order_up_to"},
    # ---- Wave 4: tool-calling operator (run_python per decision) ----
    "toolagent": {"_tool": True},   # LLM can call run_python each week before ordering
    # ---- Wave 5: TypeSafe Jev (System One decision model) — see agent_bullwhip/jev_agent.py
    # The model only *chooses* (typed answers + calibrated confidence); the arithmetic
    # stays deterministic in code. Gated variants test the confidence-gate idea that
    # was inert for the text LLM (dead claim #1).
    "jev_mult": {"_jev": True, "mode": "choice_mult"},
    "jev_mult_expect": {"_jev": True, "mode": "expectation"},
    "jev_mult_gated": {"_jev": True, "mode": "choice_mult", "conf_threshold": 0.5, "anchor_margin": 6},
    "jev_grid": {"_jev": True, "mode": "choice_grid"},
    "jev_grid_gated": {"_jev": True, "mode": "choice_grid", "conf_threshold": 0.5, "anchor_margin": 6},
    "jev_reads": {"_jev": True, "mode": "reads"},
    # wrapper-only ablation: this agent's anchor, zero API calls (isolates the formula)
    "jev_anchor_only": {"_jev": True, "mode": "anchor_only"},
    # gate controls: threshold 1.0 = gate always fires (pure anchor, model still asked);
    # threshold 0.0 = gate never fires (the model's own judgment only, clamped).
    "jev_gate_always": {"_jev": True, "mode": "choice_grid", "conf_threshold": 1.0, "anchor_margin": 6},
    "jev_gate_never": {"_jev": True, "mode": "choice_grid", "conf_threshold": 0.0, "anchor_margin": 6},
    # decisive control: the SAME ±6 perturbation, from a seeded RNG instead of the model
    "jev_jitter6_a": {"_jev": True, "mode": "jitter_only", "anchor_margin": 6, "jitter_seed": 1},
    "jev_jitter6_b": {"_jev": True, "mode": "jitter_only", "anchor_margin": 6, "jitter_seed": 7},
    "jev_jitter6_c": {"_jev": True, "mode": "jitter_only", "anchor_margin": 6, "jitter_seed": 13},
    # the bar that matters: the OOS-tuned formula (fixed-demand 3,148 in the repo's own measurement)
    "anchor_tuned": {"_jev": True, "mode": "anchor_only", "theta": 3.0, "lam": 0.35},
}


def make_agents(config: dict, model: str, tag: str | None = None) -> dict:
    if config.get("_baseline") == "mirror":
        return {r: MirrorAgent() for r in ROLES}
    if config.get("_baseline") == "order_up_to":
        return {r: OrderUpToAgent() for r in ROLES}
    lcfg = {k: v for k, v in config.items() if not k.startswith("_")}
    lcfg["tag"] = tag or model
    if config.get("_jev"):
        # TypeSafe Jev: the runner's --model is a FreeInference id, so only pass it
        # through when it is actually a Jev model (agent defaults to jev-latest).
        from .jev_agent import JevAgent, JevAgentConfig
        jcfg = dict(lcfg)
        if not str(jcfg.get("model") or "").startswith("jev"):
            jcfg["model"] = None
        return {r: JevAgent(r, JevAgentConfig(**jcfg)) for r in ROLES}
    if config.get("_tool"):
        return {r: ToolAgent(r, LLMAgentConfig(**lcfg, model=model)) for r in ROLES}
    return {r: LLMAgent(r, LLMAgentConfig(**lcfg, model=model)) for r in ROLES}


def run_config(name: str, runs: int, model: str, horizon: int, pattern: str, outdir: Path) -> dict:
    cfg = CONFIGS[name]
    tag = f"{model}-{name}".replace("/", "__")  # model IDs contain / — safe for filenames
    sim_cfg = SimConfig(horizon=horizon)
    results: list[dict] = []
    run_logs = []
    t0 = time.time()
    total_prompt_tokens = total_completion_tokens = 0
    collision_total = 0
    collision_runs = 0
    for i in range(runs):
        run_error = None
        for run_attempt in range(5):  # retry transient endpoint failures per run
            try:
                # 'noisy' demand: fresh seeded realization per run (reproducible).
                # Deterministic patterns stay identical across runs (CV isolates
                # agent instability — that's the paper's design).
                if pattern == "noisy":
                    demand = make_demand(horizon, pattern, seed=1000 + i)
                else:
                    demand = make_demand(horizon, pattern)
                agents = make_agents(cfg, model, tag=tag)
                # fresh prompt-hash log per run (module-level list, cleared before)
                from . import client as client_mod
                client_mod.prompt_hashes.clear()
                log = run_game(agents, demand, sim_cfg)
                log.agents = agents  # attach for failure metrics
                run_logs.append(log)
                prompt_tok = sum(getattr(a, "prompt_tokens", 0) for a in agents.values())
                comp_tok = sum(getattr(a, "completion_tokens", 0) for a in agents.values())
                total_prompt_tokens += prompt_tok
                total_completion_tokens += comp_tok
                # collision check: any repeated exact-prompt fingerprint within the run
                hashes = client_mod.prompt_hashes
                dups = len(hashes) - len(set(hashes))
                if dups:
                    collision_total += dups
                    collision_runs += 1
                results.append({
                    "run": i,
                    "config": name,
                    "model": model,
                    "tag": tag,
                    "total_cost": log.total_cost(),
                    "orders": {r: log.orders(r) for r in ROLES},
                    "backlogs": {r: log.backlogs(r) for r in ROLES},
                    "failures": {r: agents[r].failures for r in ROLES},
                    "calls": {r: agents[r].calls for r in ROLES},
                    "tokens": {"prompt": prompt_tok, "completion": comp_tok},
                    "prompt_hash_dups": dups,
                    # Wave 2: per-decision traces (order/confidence/reasoning/gated) for gap mining
                    "traces": {r: getattr(agents[r], "traces", []) for r in ROLES},
                    # Wave 4: tool-calling traces (what code the model wrote)
                    "tool_traces": {r: getattr(agents[r], "tool_traces", []) for r in ROLES},
                })
                run_error = None
                break
            except Exception as e:  # noqa: BLE001 - a failed run must not kill the matrix
                run_error = e
                if run_attempt < 4:  # transient endpoint flakiness: retry
                    print(f"[run {i}] attempt {run_attempt} failed: {str(e)[:80]} — retrying", flush=True)
                    time.sleep(10 * (run_attempt + 1))
                else:
                    results.append({"run": i, "config": name, "model": model, "tag": tag,
                                    "total_cost": None, "error": str(e)})
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / f"{tag}.jsonl", "a") as f:
            f.write(json.dumps(results[-1]) + "\n")

    try:
        if run_logs:
            metrics = compute_metrics(run_logs)
        else:
            metrics = {"n_runs": 0, "mean_cost": None, "cv_cost": None, "error": "all runs failed"}
    except Exception as e:  # noqa: BLE001 - metrics failure must not kill the matrix
        metrics = {"n_runs": len(run_logs), "error": f"metrics failed: {e}"}
    metrics.update({
        "config": name, "model": model, "tag": tag, "runs": runs,
        "completed": len(run_logs), "failed": len(results) - len(run_logs),
        "wall_sec": time.time() - t0,
        "tokens": {"prompt": total_prompt_tokens, "completion": total_completion_tokens,
                   "total": total_prompt_tokens + total_completion_tokens},
        # cache-collision check: repeated exact-prompt fingerprints (across runs, an
        # identical week-1 prompt would appear 30x; within a run it's a red flag)
        "prompt_hash_dups_total": collision_total,
        "prompt_hash_dups_runs": collision_runs,
    })
    summary_path = outdir / f"{tag}.summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), default="baseline")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("FREEINFERENCE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--pattern", default="step")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", action="store_true", help="list available endpoint models and exit")
    args = ap.parse_args()

    if args.models:
        for m in list_models():
            print(m)
        return

    outdir = Path(args.outdir)
    m = run_config(args.config, args.runs, args.model, args.horizon, args.pattern, outdir)
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=2, default=float))


if __name__ == "__main__":
    main()
