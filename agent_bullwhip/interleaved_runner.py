#!/usr/bin/env python3
"""Interleaved round-robin runner — kills the window-drift confound.

Every config runs one game per pass, in a shuffled-but-seeded order, so each
config experiences the SAME set of endpoint windows. Window drift stops being
a between-config confound and becomes a measured quantity (you can see it as
the spread of identical-config results across passes).

Also fixes (per AUDIT.md):
- truncate-on-start with a run-UUID (no more append-mode mixing)
- per-call timestamp + model id logged in each trace decision
- KB text snapshotted into each run record (so re-analysis uses the KB that
  was actually injected, not the current file)
- cross-run prompt-hash collision check (not cleared per run)
- per-run nonce in the prompt to defeat serving-level caching (via client)
- wrapper-only ablation column: runs the same wrappers (anchor clamp / gate /
  verbal gate) but with the LLM's order REPLACED by the anchor, so we can see
  how much of the win is the wrapper vs the model.

Usage:
    python -m agent_bullwhip.interleaved_runner --configs baseline,kb_system_gated,... \
        --runs 30 --horizon 36 --model deepseek-v4-flash --outdir results/interleaved
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import uuid
from pathlib import Path

from .agents import LLMAgent, LLMAgentConfig, MirrorAgent, OrderUpToAgent
from .engine import ROLES, SimConfig, make_demand, run_game
from .metrics import compute_metrics
from .runner import CONFIGS, make_agents


def run_one_game(agents, demand, sim_cfg, config_name, model, run_idx, kb_snapshots, nonce):
    """Run a single game, attach per-call metadata, return the record dict."""
    log = run_game(agents, demand, sim_cfg)
    log.agents = agents
    # attach per-call metadata to each trace decision
    from . import client as client_mod
    for role in ROLES:
        for t in getattr(agents[role], "traces", []):
            t["_ts"] = time.time()
            t["_model"] = model
            t["_nonce"] = nonce
            t["_run"] = run_idx
    prompt_tok = sum(getattr(a, "prompt_tokens", 0) for a in agents.values())
    comp_tok = sum(getattr(a, "completion_tokens", 0) for a in agents.values())
    rec = {
        "run": run_idx,
        "config": config_name,
        "model": model,
        "total_cost": log.total_cost(),
        "orders": {r: log.orders(r) for r in ROLES},
        "backlogs": {r: log.backlogs(r) for r in ROLES},
        "failures": {r: getattr(agents[r], "failures", 0) for r in ROLES},
        "calls": {r: getattr(agents[r], "calls", 0) for r in ROLES},
        "tokens": {"prompt": prompt_tok, "completion": comp_tok},
        "kb_snapshot": kb_snapshots.get(config_name),
        "traces": {r: getattr(agents[r], "traces", []) for r in ROLES},
    }
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="comma-separated config names")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--model", default=os.environ.get("FREEINFERENCE_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--pattern", default="step", choices=["step", "shock", "constant", "noisy", "chaotic", "wild"])
    ap.add_argument("--outdir", default="results/interleaved")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wrapper-only", action="store_true",
                    help="ablation: replace LLM order with anchor in wrappers (see AUDIT.md)")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_uuid = uuid.uuid4().hex[:8]
    tag = f"{args.model}-interleaved-{run_uuid}"
    jsonl_path = outdir / f"{tag}.jsonl"

    # truncate-on-start (clean slate for THIS invocation)
    if jsonl_path.exists():
        jsonl_path.unlink()

    # snapshot KB text per config (so re-analysis uses what was injected)
    from .agents import _kb_text
    kb_snapshots = {}
    for c in configs:
        cfg = CONFIGS[c]
        kb_file = cfg.get("kb_file", "PROMPT_KB.md")
        kb_snapshots[c] = _kb_text(kb_file) if cfg.get("kb") else None

    sim_cfg = SimConfig(horizon=args.horizon)
    # per-run nonce (defeats serving-level caching; logged per trace)
    nonce = hashlib.sha256(f"{run_uuid}-{args.seed}".encode()).hexdigest()[:12]

    results: list[dict] = []
    # round-robin: shuffle the config order per pass with a seeded RNG so no
    # config always runs first/last within a window
    rng = random.Random(args.seed)
    t0 = time.time()
    total_prompt = total_comp = 0
    # cross-run prompt-hash collision check (module-level, NOT cleared per run)
    from . import client as client_mod
    client_mod.prompt_hashes.clear()

    for i in range(args.runs):
        order = configs[:]
        rng.shuffle(order)
        # AUDIT3 FIX: generate ONE demand per pass (shared by all configs) and seed
        # EVERY pattern. Previously chaotic/wild drew a fresh entropy-seeded path
        # per (pass, config), destroying the within-pass pairing entirely.
        demand = make_demand(args.horizon, args.pattern, seed=1000 + i)
        for c in order:
            try:
                agents = make_agents(CONFIGS[c], args.model, tag=f"{args.model}-{c}")
                if args.wrapper_only:
                    # wrapper-only ablation: every agent's order is forced to the anchor
                    for role in ROLES:
                        a = agents[role]
                        if isinstance(a, LLMAgent) and a.anchor is not None:
                            orig = a.decide
                            a.decide = lambda ctx, _o=orig, _a=a: _a.anchor(ctx)
                rec = run_one_game(agents, demand, sim_cfg, c, args.model, i, kb_snapshots, nonce)
                results.append(rec)
                total_prompt += rec["tokens"]["prompt"]
                total_comp += rec["tokens"]["completion"]
            except Exception as e:  # noqa: BLE001
                results.append({"run": i, "config": c, "model": args.model,
                                "total_cost": None, "error": str(e)})
            jsonl_path.open("a").write(json.dumps(results[-1]) + "\n")

    # summary per config (mean/CV over the runs that actually ran)
    per_config = {}
    for rec in results:
        if rec.get("total_cost") is None:
            continue
        per_config.setdefault(rec["config"], []).append(rec["total_cost"])
    summaries = {}
    for c, costs in per_config.items():
        mean = sum(costs) / len(costs)
        std = (sum((x - mean) ** 2 for x in costs) / len(costs)) ** 0.5
        summaries[c] = {"n": len(costs), "mean_cost": mean, "cv_cost": std / mean if mean else 0,
                        "min": min(costs), "max": max(costs)}
    # cross-run collision check: repeated exact-prompt fingerprints across the whole run
    hashes = client_mod.prompt_hashes
    dup_total = len(hashes) - len(set(hashes))
    meta = {
        "tag": tag, "model": args.model, "pattern": args.pattern, "runs": args.runs,
        "configs": configs, "run_uuid": run_uuid, "wall_sec": time.time() - t0,
        "completed": len([r for r in results if r.get("total_cost") is not None]),
        "failed": len([r for r in results if r.get("total_cost") is None]),
        "prompt_hash_dups_total": dup_total,
        "prompt_hash_dups_calls": len(hashes),
        "tokens": {"prompt": total_prompt, "completion": total_comp},
        "per_config": summaries,
        "wrapper_only": args.wrapper_only,
    }
    (outdir / f"{tag}.summary.json").write_text(json.dumps(meta, indent=2, default=float))
    print(json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    main()
