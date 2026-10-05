"""Wave-7 decision-native eval: batched readout over the wave-1-6 protocol arms.

WAVE7_SPEC.md D5 + WAVE7-AUDIT-1 fixes (binding):

- [P0 step n=30 is n=1] the step path is deterministic and argmax readout is
  deterministic, so ALL step-argmax runs are identical. This file therefore
  emits the step arm as a SINGLE labeled cost (run=0) and adds a SAMPLED
  readout arm (draw from pi, torch.Generator seeded from 1000+i per run) as the
  n=30 distributional step comparison. The inferential gates (C1/C3 sign tests)
  move to noisy n=30, where the 30 demand paths genuinely differ.
- JSONL schema EXACTLY the wave-1-6 runner shape (eval-analyze.py consumer:
  run/config/model/tag/total_cost/orders/backlogs/failures/calls/tokens/
  prompt_hash_dups/traces/tool_traces). failures is asserted 0 per run —
  parse-fail==0 is an assertion, not a hope. eval-analyze's rogue columns read
  zero by construction (readout orders are grid values 0..40).
- One batched forward pass per week across all runs x roles.
- `--collect-states`: roll out the CURRENT policy (argmax) on curriculum
  episodes and dump every visited decision state with full engine internals +
  curriculum params — the round-2 relabel input (DAgger-style, audit fix 8);
  decision_labeler consumes it via --states-jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .engine import ROLES, GameStepper, SimConfig, make_demand
from .decision_labeler import _seeded_rng, sample_curriculum_demand_ex
from .decision_readout import ORDER_GRID, PINNED_REVISION, ReadoutModel, render_prompt


def sample_orders(probs_rows: list[list[float]], run_ids: list[int],
                  seed_base: int):
    """One sample per (run, slot) from the policy distribution; each run gets
    its own torch.Generator seeded from a stable hash of (seed_base + run) so
    sampled runs are reproducible and mutually independent."""
    import torch
    out: dict[tuple[int, int], int] = {}
    for r in sorted(set(run_ids)):
        gen = torch.Generator()
        gen.manual_seed(int(hashlib.sha256(f"{seed_base + r}:sample".encode())
                            .hexdigest()[:16], 16) % (2**32))
        idxs = [i for i, rid in enumerate(run_ids) if rid == r]
        rows = torch.tensor([probs_rows[i] for i in idxs],
                            dtype=torch.float32)
        picks = torch.multinomial(rows, 1, generator=gen).squeeze(-1).tolist()
        for i, pick in zip(idxs, picks):
            out[(r, i)] = int(pick)
    return out


def make_demands(pattern: str, mode: str, runs: int, horizon: int,
                 seed_base: int) -> list[list[int]]:
    """Protocol demand arms. step+argmax is the single deterministic game
    (audit P0); step+sampled and noisy reuse seeds 1000+i per run."""
    if pattern == "noisy":
        return [make_demand(horizon, "noisy", seed=seed_base + i)
                for i in range(runs)]
    if mode == "argmax":
        return [make_demand(horizon, "step")]
    return [make_demand(horizon, "step", seed=seed_base + i)
            for i in range(runs)]


def run_batched_eval(model: ReadoutModel, demands: list[list[int]],
                     mode: str, seed_base: int, horizon: int,
                     tag: str, model_name: str, *, require_mass: float = 0.99,
                     masses: list[float] | None = None) -> list[dict]:
    """R episodes in lockstep; one batched readout per week across runs x roles.
    Returns the wave-1-6 schema records. Raises if any failures (parse-fail
    ==0 assertion) — the readout cannot fail by construction. `require_mass`
    is forwarded to readout_batch (WAVE7_SPEC A1: 0.99 fatal for the untrained
    readout, 0.0 for trained adapters); per-readout masses go to `masses`."""
    R = len(demands)
    steppers = [GameStepper(d, SimConfig(horizon=horizon)) for d in demands]
    orders: list[dict[str, list[int]]] = [
        {r: [] for r in ROLES} for _ in range(R)]
    prompt_tokens = [0] * R
    while not all(s.done() for s in steppers):
        ctx_rows: list[tuple[int, str, dict]] = []
        for ri, s in enumerate(steppers):
            obs = s.observations()
            for r in ROLES:
                ctx_rows.append((ri, r, obs[r]))
        prompts = [render_prompt(ctx, model.tokenizer) for _, _, ctx in ctx_rows]
        recs = model.readout_batch(prompts, require_mass=require_mass)
        if masses is not None:
            masses.extend(rec["mass"] for rec in recs)
        if mode == "sampled":
            picks = sample_orders([r["probs"] for r in recs],
                                  [ri for ri, _, _ in ctx_rows], seed_base)
            chosen = [ORDER_GRID[picks[(ri, i)]] for i, (ri, _, _)
                      in enumerate(ctx_rows)]
        else:
            chosen = [rec["argmax_order"] for rec in recs]
        week_orders: dict[int, dict[str, int]] = {}
        for i, (ri, r, _) in enumerate(ctx_rows):
            week_orders.setdefault(ri, {})[r] = chosen[i]
        for ri, s in enumerate(steppers):
            if not s.done():
                s.advance(week_orders[ri])
        for ri, r, _ in ctx_rows:
            orders[ri][r].append(week_orders[ri][r])
        for ri, rec in enumerate(recs):
            prompt_tokens[ctx_rows[ri][0]] += rec["tokens"]
    records = []
    for ri in range(R):
        log = steppers[ri].log()
        cfg = SimConfig(horizon=horizon)
        records.append({
            "run": ri,
            "config": "baseline",
            "model": model_name,
            "tag": tag,
            "total_cost": log.total_cost(),
            "orders": orders[ri],
            "backlogs": {r: log.backlogs(r) for r in ROLES},
            "failures": {r: 0 for r in ROLES},
            "calls": {r: len(orders[ri][r]) for r in ROLES},
            "tokens": {"prompt": prompt_tokens[ri], "completion": 0},
            "prompt_hash_dups": 0,
            "traces": {r: [] for r in ROLES},
            "tool_traces": {r: [] for r in ROLES},
        })
    for rec in records:
        assert all(v == 0 for v in rec["failures"].values()), \
            f"run {rec['run']}: parse-fail must be 0"
    return records


def collect_policy_states(model: ReadoutModel, n_episodes: int, seed: int,
                          curricula: list[str], horizon: int,
                          out_path: Path) -> int:
    """Roll out the CURRENT policy (argmax) on curriculum episodes and dump
    every visited decision state with full engine internals + curriculum params
    (round-2 relabel input for decision_labeler --states-jsonl). Uses the
    labeler's call-identical curriculum samplers; per-episode rng is a stable
    SHA seed, so the dump is reproducible.

    The policy is a trained adapter, so the label-mass invariant is recorded,
    not asserted (WAVE7_SPEC A1): each row carries its `mass`, and
    `<out>.notes.json` records min/median over the whole collect."""
    n = 0
    masses: list[float] = []
    with open(out_path, "w") as f:
        for ep in range(n_episodes):
            cur = curricula[ep % len(curricula)]
            rng = _seeded_rng(f"{seed}:ep:{ep}")
            params, demand = sample_curriculum_demand_ex(rng, cur, horizon)
            step = GameStepper(demand, SimConfig(horizon=horizon))
            il_hist: dict[str, list[int]] = {r: [] for r in ROLES}
            while not step.done():
                obs = step.observations()
                for r in ROLES:
                    il_hist[r].append(obs[r]["incoming_last"])
                prompts = [render_prompt(obs[r], model.tokenizer)
                           for r in ROLES]
                recs = model.readout_batch(prompts, require_mass=0.0)
                chosen = {r: int(rec["argmax_order"]) for r, rec
                          in zip(ROLES, recs)}
                mass_of = {r: float(rec["mass"]) for r, rec in zip(ROLES, recs)}
                masses.extend(mass_of.values())
                for r in ROLES:
                    row = {
                        "episode": ep, "week": step.t, "role": r,
                        "curriculum": cur, "params": params, "ctx": obs[r],
                        "il_hist": {r2: list(il_hist[r2]) for r2 in ROLES},
                        "state": {
                            "t": step.t,
                            "tiers": {r2: [step.state[r2].oh,
                                           step.state[r2].b,
                                           step.state[r2].o]
                                      for r2 in ROLES},
                            "inbound": {r2: list(step.inbound[r2])
                                        for r2 in ROLES},
                            "prev_incoming": dict(step.prev_incoming),
                            "order_history": {r2: list(step.order_history[r2])
                                              for r2 in ROLES},
                        },
                        "order": chosen[r],
                        "mass": mass_of[r],
                    }
                    f.write(json.dumps(row, sort_keys=True) + "\n")
                    n += 1
                step.advance(chosen)
    masses.sort()
    notes = {"n_states": n, "require_mass": 0.0,
             "min_mass": masses[0] if masses else None,
             "median_mass": masses[len(masses) // 2] if masses else None,
             "note": "trained-adapter collect: label mass recorded per row, "
                     "not asserted (WAVE7_SPEC A1)"}
    notes_path = out_path.with_name(out_path.name + ".notes.json")
    notes_path.write_text(json.dumps(notes, indent=2, sort_keys=True))
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_bullwhip.decision_eval")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=PINNED_REVISION)
    ap.add_argument("--adapter", default=None, help="trained LoRA adapter dir")
    ap.add_argument("--pattern", choices=["step", "noisy"], default="step")
    ap.add_argument("--mode", choices=["argmax", "sampled"], default="argmax")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--seed-base", type=int, default=1000,
                    help="eval demand/sampling seeds 1000+i (reserved — the "
                         "labeler rejects that band)")
    ap.add_argument("--model-name", default=None,
                    help="JSONL 'model' field (defaults to a derived tag)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default="results/eval")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--collect-states", default=None)
    ap.add_argument("--n-collect", type=int, default=8)
    ap.add_argument("--collect-seed", type=int, default=7000)
    ap.add_argument("--collect-curricula", default="poisson,truncnorm")
    args = ap.parse_args(argv)

    arm = "untrained" if not args.adapter else "trained"
    tag = args.tag or f"qwen3-0.6b-readout-{arm}-{args.mode}-{args.pattern}"
    model_name = args.model_name or f"qwen3-0.6b-{arm}"

    model = ReadoutModel(args.model, device=args.device, dtype=args.dtype,
                         adapter_dir=args.adapter, revision=args.revision)

    if args.collect_states:
        n = collect_policy_states(model, args.n_collect, args.collect_seed,
                                  args.collect_curricula.split(","),
                                  args.horizon, Path(args.collect_states))
        print(f"collected {n} policy states -> {args.collect_states} "
              f"(label mass recorded, not asserted; see .notes.json)")
        return 0

    n_runs = 1 if (args.pattern == "step" and args.mode == "argmax") \
        else args.runs
    demands = make_demands(args.pattern, args.mode, n_runs, args.horizon,
                           args.seed_base)
    t0 = time.time()
    # WAVE7_SPEC A1: mass >= 0.99 is the format proof of the UNTRAINED readout
    # (fatal); an SFT'd adapter erodes it, so a trained eval records min_mass.
    require_mass = 0.0 if args.adapter else 0.99
    masses: list[float] = []
    records = run_batched_eval(model, demands, args.mode, args.seed_base,
                               args.horizon, tag, model_name,
                               require_mass=require_mass, masses=masses)
    costs = [r["total_cost"] for r in records]
    stats = {
        "n_runs": len(records), "runs": n_runs, "config": "baseline",
        "model": model_name, "tag": tag,
        "mean_cost": sum(costs) / len(costs),
        "min_cost": min(costs), "max_cost": max(costs),
        "failure_rate": 0.0, "parse_fail": 0,
        "completed": len(records), "failed": 0,
        "wall_sec": time.time() - t0,
        "require_mass": require_mass,
        "min_mass": min(masses) if masses else None,
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{tag}.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    with open(outdir / f"{tag}.summary.json", "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True, default=float)
    print(f"{tag}: runs={len(records)} mean={stats['mean_cost']:,.0f} "
          f"min={min(costs):,.0f} max={max(costs):,.0f} parse_fail=0 "
          f"min_mass={stats['min_mass']:.5f} "
          f"wall={stats['wall_sec']:.1f}s -> {outdir / (tag + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())