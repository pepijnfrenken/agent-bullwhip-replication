"""Decision labeler: simulator-as-teacher (WAVE7_SPEC.md D3).

The labeler produces, for each engine state visited under a mixed behavior
policy on wave-6 curriculum demand, the softmax target p*(a) ∝ exp(-J(s,a)/tau)
over the order grid 0..40 step 2 (A..U), where J(s,a) is the expected cost of
the episode remainder when this decision is `a` and all later decisions follow a
fixed continuation policy.

WAVE7-AUDIT-1 fixes baked in (re-read at P2 freeze, 2026-10-05):

1. [P1 hindsight] J is E[J] over M >= 16 common-random-number futures drawn from
   the *episode's own curriculum parameters* (lambda / mu,sigma), not one
   realized path. CRN: the same M futures score all 21 candidates.
2. [P1 stateful floor] The continuation floor's exponential-smoothing forecast
   is warmed by replaying the role's observed incoming_last sequence from week 0
   (test: warm-from-replay == warm-from-run). The candidate role skips its week-t
   floor call; the warm replay already accounts for week t's update.
3. [P1 continuation policy] FIXED (theta, lam), tuned out-of-sample on held-out
   curriculum episodes (60 tune + 60 validation, seeds 4000..4119, 2026-10-05):
   (3.5, 0.2) — mean 8,766 / validation 9,066 vs untuned (3.0,0.5) 15,544/15,997.
   No per-path walk-forward search: that would leak future demand into labels.
4. [P1/P2 grid] The 0..40 grid is pre-registered (D2) and NOT widened here; the
   edge-argmin rate is reported as a label-side gate (fraction of states whose
   E[J] argmin sits at the grid ends), and flat/terminal states are excluded:
   t >= H - 2 for the ordering role (an order placed there lands after the
   horizon — its J is identically flat).
5. [P2 tau] tau_s = tau_scale * max(std_a(E[J]), 1e-3), tau_scale default 0.5
   ("medium"); ablate {0.25, 0.5, 1.0} on the label side; the rule is recorded in
   the manifest. Median target entropy is reported per run.
6. [P2 same-week co-decisions] other roles' week-t orders (and the candidate
   role's later weeks) follow the continuation policy — "deviate once, then
   everyone plays the floor" (one policy-improvement step over the continuation).
7. [D5 hygiene] eval demand uses seeds 1000+i (reserved); the labeler REJECTS any
   seed in [1000, 1030) and draws demand only from the wave-6 curricula
   (call-for-call identical to grpo_env.sample_curriculum_demand — tested).

Determinism: every rng is derived from a stable SHA-256 string
(seed:ep:w:role), never from process/iteration order, so multiprocessing cannot
change the output. The engine cost arithmetic is plain Python floats (no numpy
in the J path). Decision content (J, argmin, flat, ctx) is bit-identical across
machines; tau/p carry <=1 ulp differences across Python MINORS because builtin
sum() of floats is Neumaier-compensated since CPython 3.12 (gh-100425) and the
three float-critical sums below use it (hub py3.13 vs cluster py3.11.3:
25063/95200 rows differ in tau, 54255 in p — 2026-10-06 forensics in
hpc-umbrella/HASH-FORENSICS.md). `--hash-rows` writes the SHA-256 of the first
N jsonl rows; compare machines with tools/label_fingerprint.py +
hpc-umbrella/labels_probe.py (decisions + J + p-12dp), not raw bytes.

Torch-free: this module imports numpy only (npz mirror); prompt rendering
happens in the trainer, which owns the tokenizer.

Usage:
    python -m agent_bullwhip.decision_labeler --seed 6000 --episodes 400 \
        --out results/wave7_labels/labels --processes 8 --hash-rows 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .engine import ROLES, GameStepper, SimConfig, TierState
from .agents import OrderUpToAgent
from .decision_readout import ORDER_GRID, LABELS

# Fixed continuation floor — see AUDIT-1 fix 3 (tuned out-of-sample, logged).
CONTINUATION_THETA = 3.5
CONTINUATION_LAM = 0.2
CONTINUATION_TUNING_NOTE = (
    "out-of-sample grid search (theta in 1.5..6.0 x lam in 0.1..0.9) over 60 "
    "wave-6 curriculum episodes (seeds 4000..4059, poisson/truncnorm), "
    "validated on 60 held-out episodes (seeds 4060..4119): "
    "tune mean 8,766 / validation mean 9,066 vs untuned (3.0,0.5) 15,544/15,997"
)

# Eval demand seeds are reserved (runner.py: noisy seeds 1000+i). The labeler
# must never draw them (AUDIT-1 3b / brief: "seeds 1000+i reserved").
EVAL_SEED_MIN, EVAL_SEED_MAX = 1000, 1030


def _seeded_rng(spec: str) -> random.Random:
    """Deterministic RNG from a stable string (process/order independent)."""
    h = hashlib.sha256(spec.encode("utf-8")).digest()[:8]
    return random.Random(int.from_bytes(h, "big"))


# --------------------------------------------------------------------------- #
# Curriculum demand (wave-6 replicas with exposed params; call-for-call equal
# to grpo_env.sample_curriculum_demand — hud tested). grpo_env is wave-6 code
# (read-only per the brief), so the sampler is duplicated here with the params
# it samples internally returned for future draw generation.
# --------------------------------------------------------------------------- #
def _poisson_knut(rng: random.Random, lam: float) -> int:
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return min(k, 50)
        k += 1


def sample_curriculum_demand_ex(rng: random.Random, curriculum: str,
                                horizon: int = 36) -> tuple[dict, list[int]]:
    """(params, demand) — same draws + same order as grpo_env's sampler."""
    if curriculum == "poisson":
        lam = rng.uniform(5, 20)
        return {"lam": lam}, [_poisson_knut(rng, lam) for _ in range(horizon)]
    if curriculum == "truncnorm":
        mu = rng.uniform(8, 20)
        sigma = rng.uniform(2, 6)
        out: list[int] = []
        while len(out) < horizon:
            v = rng.gauss(mu, sigma)
            if 0 <= v <= 50:
                out.append(int(round(v)))
        return {"mu": mu, "sigma": sigma}, out
    raise ValueError(f"unknown curriculum: {curriculum}")


def draw_future_demand(rng: random.Random, curriculum: str, params: dict,
                       length: int) -> list[int]:
    """iid demand continuation with the episode's OWN params (AUDIT-1 fix 1)."""
    if curriculum == "poisson":
        lam = params["lam"]
        return [_poisson_knut(rng, lam) for _ in range(length)]
    mu, sigma = params["mu"], params["sigma"]
    out: list[int] = []
    while len(out) < length:
        v = rng.gauss(mu, sigma)
        if 0 <= v <= 50:
            out.append(int(round(v)))
    return out


# --------------------------------------------------------------------------- #
# Behavior policy (state generator) — floors + random, per spec D3.
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorConfig:
    p_floor: float = 0.75        # order-up-to with per-episode (theta,lam)
    p_random: float = 0.25       # uniform over the order grid
    theta_range: tuple[float, float] = (1.5, 5.0)
    lam_range: tuple[float, float] = (0.2, 0.8)

    def __post_init__(self):
        if not math.isclose(self.p_floor + self.p_random, 1.0):
            raise ValueError("behavior probabilities must sum to 1")


class BehaviorPolicy:
    """Mixed per-week per-role policy. Stateful floor agents per episode (their
    forecast drift after random weeks is fine — they only GENERATE states; the
    continuation policy used for J is rebuilt by warm replay, fix 2)."""

    def __init__(self, cfg: BehaviorConfig, rng: random.Random):
        self.cfg = cfg
        self._floors = {r: OrderUpToAgent(
            rng.uniform(*cfg.theta_range), rng.uniform(*cfg.lam_range))
            for r in ROLES}

    def act(self, role: str, ctx: dict, rng: random.Random) -> int:
        if rng.random() < self.cfg.p_floor:
            return self._floors[role](ctx)
        return rng.choice(ORDER_GRID)


# --------------------------------------------------------------------------- #
# Episode replay (deterministic from the seed string) — used by both the state
# enumerator and the label workers, so warm histories and snapshots agree.
# --------------------------------------------------------------------------- #
def _replay_to_week(seed: int, ep: int, week: int, curriculum: str,
                    horizon: int, behavior: BehaviorConfig
                    ) -> tuple[GameStepper, dict[str, list[int]], dict]:
    """Replay the behavior-policy episode deterministically up to (not including)
    `week`, returning (stepper at week, per-role incoming_last history INCLUDING
    week `week` itself, curriculum params). The history includes the decision
    week's incoming_last: the floor updates its forecast with it BEFORE deciding
    (OrderUpToAgent.__call__), so warm-from-replay must too (AUDIT-1 fix 2)."""
    rng = _seeded_rng(f"{seed}:ep:{ep}")
    params, demand = sample_curriculum_demand_ex(rng, curriculum, horizon)
    pol = BehaviorPolicy(behavior, rng)
    step = GameStepper(demand, SimConfig(horizon=horizon))
    il_hist: dict[str, list[int]] = {r: [] for r in ROLES}
    while step.t < week:
        obs = step.observations()
        for r in ROLES:
            il_hist[r].append(obs[r]["incoming_last"])
        step.advance({r: pol.act(r, obs[r], rng) for r in ROLES})
    obs = step.observations()                 # decision week: il for weeks 0..t
    for r in ROLES:
        il_hist[r].append(obs[r]["incoming_last"])
    return step, il_hist, params




def warm_forecast(theta: float, lam: float,
                  incoming_last_seq: list[int]) -> float:
    """OrderUpToAgent forecast after replaying incoming_last weeks 0..t
    (AUDIT-1 fix 2: warm-from-replay must equal warm-from-run)."""
    f = 0.0
    for il in incoming_last_seq:
        f = lam * il + (1 - lam) * f
    return f


def make_continuation(theta: float, lam: float,
                      il_hist: dict[str, list[int]], week: int
                      ) -> dict[str, OrderUpToAgent]:
    """Continuation floor TEMPLATES warmed to the PRE-update state of decision
    week `week`: their forecast is what the floor holds before its week-`week`
    call (replay of incoming_last weeks 0..week-1). Roles that are CALLED at
    week `week` then update with that week's incoming_last themselves; the
    candidate role (whose week-`week` call is replaced by the candidate order)
    is warmed one step further in `branch_cost` via `candidate_warm`
    (AUDIT-1 fix 2: warm-from-replay == warm-from-run on both paths)."""
    cont: dict[str, OrderUpToAgent] = {}
    for r in ROLES:
        a = OrderUpToAgent(theta=theta, lam=lam)
        a._forecast = warm_forecast(theta, lam, il_hist[r][:week])
        cont[r] = a
    return cont


def branch_cost(snapshot: GameStepper, t: int, role: str, candidate: int,
                cont_template: dict[str, OrderUpToAgent], future: list[int],
                candidate_warm: float | None = None) -> float:
    """J(s, a) for ONE future: role orders `candidate` at week t, everyone else
    (and role on later weeks) plays the continuation floor. Returns the cost of
    weeks t..H-1. The branch copy and `run_game` agree on an un-branched path
    (tested); future demand replaces snapshot.demand[t:]. Each branch deep-copies
    the continuation templates, so no candidate/future can contaminate the next
    one's forecasts."""
    import copy as _copy
    snap = _copy.deepcopy(snapshot)
    snap.demand = list(snapshot.demand[:t]) + list(future)
    base_len = len(snap.weeks)                       # pre-branch week records
    cont = {r: _copy.deepcopy(a) for r, a in cont_template.items()}
    if candidate_warm is not None:
        cont[role]._forecast = candidate_warm   # post-update warm, then NO week-t call
    obs = snap.observations()
    orders: dict[str, int] = {}
    for r in ROLES:
        orders[r] = candidate if r == role else cont[r](obs[r])
    snap.advance(orders)
    while not snap.done():
        obs = snap.observations()
        snap.advance({r: cont[r](obs[r]) for r in ROLES})
    log = snap.log()
    return float(sum(w.cost for w in log.weeks[base_len:]))


def score_candidates(snapshot: GameStepper, t: int, role: str,
                     candidates: list[int],
                     cont_template: dict[str, OrderUpToAgent],
                     futures: list[list[int]], theta: float, lam: float,
                     il_hist: dict[str, list[int]]) -> list[float]:
    """E[J] over the CRN futures for every candidate (fix 1: same futures for
    all candidates — the argmin comparison is on common random numbers)."""
    candidate_warm = warm_forecast(theta, lam, il_hist[role][:t + 1])
    out = []
    for cand in candidates:
        tot = 0.0
        for future in futures:
            tot += branch_cost(snapshot, t, role, cand, cont_template, future,
                               candidate_warm=candidate_warm)
        out.append(tot / len(futures))
    return out


def softmax_minus_j(j_vals: list[float], tau_scale: float
                    ) -> tuple[list[float], float, bool]:
    """Targets p* ∝ exp(-J/tau_s), tau_s = tau_scale * max(std(J), 1e-3).
    Returns (p, tau_s, flat). Flat (std below threshold) -> uniform targets."""
    n = len(j_vals)
    # NOTE (2026-10-06): builtin sum() of floats is Neumaier-compensated since
    # CPython 3.12 (gh-100425); on <=3.11 it is plain left-to-right. This sum,
    # the std sum below, and z at the end are the only cross-machine float
    # dependence in a row (<=1 ulp in tau/p; J above is exact binary fractions
    # and immune). Cross-version byte-identity would need math.fsum here.
    mean = sum(j_vals) / n
    std = math.sqrt(sum((j - mean) ** 2 for j in j_vals) / n)
    flat = std < 1e-3
    tau = tau_scale * max(std, 1e-3)
    if flat:
        return [1.0 / n] * n, tau, True
    jmin = min(j_vals)
    # max-shift: exp(-(j-jmin)/tau) is the same distribution, immune to
    # underflow when tau is tiny relative to the J level (late-week states can
    # carry huge flat-ish costs — the naive exp(-j/tau) sums to 0).
    e = [math.exp(-(j - jmin) / tau) for j in j_vals]
    z = sum(e)  # builtin sum(): Neumaier since 3.12 — see note at the top
    return [x / z for x in e], tau, False


def _label_state_task(task: dict) -> dict:
    """Worker: score one state and return its row payload. Task is numeric-only
    (seed, ep, week, role, curriculum, behavior, theta, lam, M, tau_scale,
    horizon) — snapshot + warm history rebuilt by deterministic replay."""
    seed = task["seed"]; ep = task["ep"]; week = task["week"]
    role = task["role"]; cur = task["curriculum"]; M = task["m_futures"]
    theta = task["theta"]; lam = task["lam"]; tau_scale = task["tau_scale"]
    horizon = task["horizon"]
    behavior = BehaviorConfig(task["p_floor"], task["p_random"])
    step, il_hist, params = _replay_to_week(seed, ep, week, cur, horizon, behavior)
    ctx = step.observations()[role]
    cont = make_continuation(theta, lam, il_hist, week)
    # M CRN futures from the episode's own curriculum params
    rng_f = _seeded_rng(f"{seed}:ep:{ep}:w:{week}:{role}")
    futures = [draw_future_demand(rng_f, cur, params, horizon - week)
               for _ in range(M)]
    j = score_candidates(step, week, role, ORDER_GRID, cont, futures,
                         theta, lam, il_hist)
    p, tau, flat = softmax_minus_j(j, tau_scale)
    argmin = min(range(len(j)), key=j.__getitem__)
    return {
        "episode": ep, "week": week, "role": role, "curriculum": cur,
        "params": params, "ctx": dict(ctx), "J": j, "p": p, "tau": tau,
        "flat": flat, "argmin_idx": argmin,
        "argmin_order": ORDER_GRID[argmin],
    }


# --------------------------------------------------------------------------- #
# Top-level run + outputs
# --------------------------------------------------------------------------- #
def enumerate_states(seed: int, n_episodes: int, horizon: int,
                     curricula: list[str]) -> list[dict]:
    """All labelable states: (ep, week, role, curriculum) with week < H - 2
    (AUDIT-1 fix 4: an order placed at t >= H-2 lands after the horizon)."""
    tasks = []
    for ep in range(n_episodes):
        cur = curricula[ep % len(curricula)]
        for week in range(max(0, horizon - 2)):
            for role in ROLES:
                tasks.append({"seed": seed, "ep": ep, "week": week,
                              "role": role, "curriculum": cur})
    return tasks


def _row_hash(rows: list[dict], n: int) -> str:
    h = hashlib.sha256()
    for row in rows[:n]:
        h.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Round-2 relabeling: label PRE-COLLECTED states (DAgger-style, audit fix 8).
# decision_eval --collect-states dumps them with full engine internals +
# curriculum params; --states-jsonl scores them with the same CRN machinery.
# --------------------------------------------------------------------------- #
def stepper_from_state(st: dict, horizon: int) -> GameStepper:
    """Rebuild a branchable engine stepper at week t from a collected state."""
    from collections import deque
    step = GameStepper([0] * horizon, SimConfig(horizon=horizon))
    step.t = int(st["t"])
    step.state = {r: TierState(*st["tiers"][r]) for r in ROLES}
    step.inbound = {r: deque(st["inbound"][r]) for r in ROLES}
    step.prev_incoming = {r: int(v) for r, v in st["prev_incoming"].items()}
    step.order_history = {r: list(st["order_history"][r]) for r in ROLES}
    step.weeks = []
    return step


def _label_collected_task(task: dict) -> dict:
    """Score one pre-collected state (mirrors _label_state_task)."""
    seed = task["seed"]; st = task["state_row"]; M = task["m_futures"]
    theta = task["theta"]; lam = task["lam"]; tau_scale = task["tau_scale"]
    week = st["week"]; role = st["role"]; cur = st["curriculum"]
    params = st["params"]
    step = stepper_from_state(st["state"], task["horizon"])
    cont = make_continuation(theta, lam, st["il_hist"], week)
    rng_f = _seeded_rng(f"{seed}:states:{st['episode']}:{week}:{role}")
    futures = [draw_future_demand(rng_f, cur, params,
                                  task["horizon"] - week) for _ in range(M)]
    j = score_candidates(step, week, role, ORDER_GRID, cont, futures,
                         theta, lam, st["il_hist"])
    p, tau, flat = softmax_minus_j(j, tau_scale)
    argmin = min(range(len(j)), key=j.__getitem__)
    return {"episode": st["episode"], "week": week, "role": role,
            "curriculum": cur, "params": params, "ctx": st["ctx"], "J": j,
            "p": p, "tau": tau, "flat": flat, "argmin_idx": argmin,
            "argmin_order": ORDER_GRID[argmin]}


def run_labeler_from_states(states_path: str | Path, *, seed: int = 6000,
                            horizon: int = 36, m_futures: int = 16,
                            theta: float = CONTINUATION_THETA,
                            lam: float = CONTINUATION_LAM,
                            tau_scale: float = 0.5, processes: int = 1,
                            out_prefix: str | None = None,
                            hash_rows: int = 2000) -> dict:
    """Label pre-collected (policy-visited) states. Same seed-band guard, same
    late-week exclusion (t >= H - 2), same CRN E[J] and tau rule as the main
    labeler. Deterministic from (seed, file rows)."""
    if EVAL_SEED_MIN <= seed < EVAL_SEED_MAX:
        raise ValueError(f"labeler seed {seed} is in the reserved eval band")
    t0 = time.time()
    with open(states_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tasks = []
    for i, st in enumerate(rows):
        if st["week"] >= horizon - 2:
            continue
        tasks.append({"seed": seed, "state_row": st, "m_futures": m_futures,
                      "theta": theta, "lam": lam, "tau_scale": tau_scale,
                      "horizon": horizon})
    if processes > 1:
        from multiprocessing import Pool
        with Pool(processes) as pool:
            out_rows = list(pool.imap(_label_collected_task, tasks, chunksize=8))
    else:
        out_rows = [_label_collected_task(t) for t in tasks]
    out_rows.sort(key=lambda r: (r["episode"], r["week"], ROLES.index(r["role"])))
    n = len(out_rows)
    n_flat = sum(1 for r in out_rows if r["flat"])
    edge = [r for r in out_rows if r["argmin_idx"] in (0, len(ORDER_GRID) - 1)]
    ent = sum(-sum(pi * math.log(pi) for pi in r["p"] if pi > 0)
              for r in out_rows) / n if n else float("nan")
    manifest = {
        "seed": seed, "source": str(states_path), "n_collected": len(rows),
        "n_excluded_late": len(rows) - len(tasks),
        "m_futures": m_futures, "processes": processes, "grid": ORDER_GRID,
        "labels": LABELS, "continuation_theta": theta, "continuation_lam": lam,
        "continuation_tuning": CONTINUATION_TUNING_NOTE,
        "tau_rule": ("tau_s = tau_scale * max(std_a(E[J]), 1e-3); "
                     f"tau_scale={tau_scale}"),
        "n_states": n, "n_flat": n_flat,
        "flat_rate": n_flat / n if n else float("nan"),
        "edge_argmin_rate": len(edge) / n if n else float("nan"),
        "mean_target_entropy_nats": ent, "wall_sec": time.time() - t0,
    }
    if out_prefix:
        out_prefix = str(out_prefix)
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        with open(out_prefix + ".jsonl", "w") as f:
            for r in out_rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        try:
            import numpy as np
            np.savez(out_prefix + ".npz",
                     p=np.array([r["p"] for r in out_rows], dtype=np.float32),
                     J=np.array([r["J"] for r in out_rows], dtype=np.float32),
                     argmin=np.array([r["argmin_idx"] for r in out_rows],
                                     dtype=np.uint8))
        except ImportError:
            manifest["npz"] = "skipped (numpy unavailable)"
        if hash_rows:
            manifest["hash"] = _row_hash(out_rows, hash_rows)
            manifest["hash_rows"] = min(hash_rows, n)
        with open(out_prefix + ".manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    manifest["rows"] = out_rows
    return manifest


def run_labeler(seed: int, n_episodes: int, *, horizon: int = 36,
                curricula: list[str] | None = None, m_futures: int = 16,
                theta: float = CONTINUATION_THETA, lam: float = CONTINUATION_LAM,
                tau_scale: float = 0.5, p_floor: float = 0.75,
                p_random: float = 0.25, processes: int = 1,
                out_prefix: str | None = None, hash_rows: int = 2000,
                ) -> dict:
    if EVAL_SEED_MIN <= seed < EVAL_SEED_MAX:
        raise ValueError(
            f"labeler seed {seed} is in the reserved eval band "
            f"[{EVAL_SEED_MIN}, {EVAL_SEED_MAX}) — eval demand paths must never "
            f"enter training labels")
    curricula = curricula or ["poisson", "truncnorm"]
    for cur in curricula:
        if cur not in ("poisson", "truncnorm"):
            raise ValueError(f"unknown curriculum {cur!r}")
    t0 = time.time()
    base_tasks = enumerate_states(seed, n_episodes, horizon, curricula)
    tasks = [dict(t, m_futures=m_futures, theta=theta, lam=lam,
                  tau_scale=tau_scale, p_floor=p_floor, p_random=p_random,
                  horizon=horizon) for t in base_tasks]
    if processes > 1:
        from multiprocessing import Pool
        with Pool(processes) as pool:
            rows = list(pool.imap(_label_state_task, tasks, chunksize=8))
    else:
        rows = [_label_state_task(t) for t in tasks]
    # deterministic order: episode, week, role (ROLES order)
    rows.sort(key=lambda r: (r["episode"], r["week"], ROLES.index(r["role"])))

    n = len(rows)
    n_flat = sum(1 for r in rows if r["flat"])
    edge = [r for r in rows if r["argmin_idx"] in (0, len(ORDER_GRID) - 1)]
    # mean target entropy (nats) over the grid — the tau band report
    ent = 0.0
    for r in rows:
        ent += -sum(pi * math.log(pi) for pi in r["p"] if pi > 0)
    ent /= n
    manifest = {
        "seed": seed, "n_episodes": n_episodes, "horizon": horizon,
        "curricula": curricula, "m_futures": m_futures, "processes": processes,
        "grid": ORDER_GRID, "labels": LABELS,
        "continuation_theta": theta, "continuation_lam": lam,
        "continuation_tuning": CONTINUATION_TUNING_NOTE,
        "tau_rule": "tau_s = tau_scale * max(std_a(E[J]), 1e-3); "
                    f"tau_scale={tau_scale} (ablate 0.25/0.5/1.0)",
        "behavior": {"p_floor": p_floor, "p_random": p_random},
        "n_states": n,
        "excluded_late_weeks": f"t >= {horizon - 2} (order lands after horizon)",
        "n_flat": n_flat, "flat_rate": n_flat / n if n else float("nan"),
        "edge_argmin_rate": len(edge) / n if n else float("nan"),
        "edge_argmin_orders": sorted({r["argmin_order"] for r in edge}),
        "mean_target_entropy_nats": ent,
        "wall_sec": time.time() - t0,
    }
    if out_prefix:
        out_prefix = str(out_prefix)
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        with open(out_prefix + ".jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        try:
            import numpy as np
            np.savez(out_prefix + ".npz",
                     J=np.array([r["J"] for r in rows], dtype=np.float32),
                     p=np.array([r["p"] for r in rows], dtype=np.float32),
                     tau=np.array([r["tau"] for r in rows], dtype=np.float64),
                     argmin=np.array([r["argmin_idx"] for r in rows], dtype=np.uint8),
                     role_idx=np.array([ROLES.index(r["role"]) for r in rows], dtype=np.uint8),
                     week=np.array([r["week"] for r in rows], dtype=np.uint8),
                     episode=np.array([r["episode"] for r in rows], dtype=np.int32),
                     flat=np.array([r["flat"] for r in rows], dtype=bool))
        except ImportError:
            manifest["npz"] = "skipped (numpy unavailable)"
        if hash_rows:
            manifest["hash"] = _row_hash(rows, hash_rows)
            manifest["hash_rows"] = min(hash_rows, n)
        with open(out_prefix + ".manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
    manifest["rows"] = rows
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_bullwhip.decision_labeler")
    ap.add_argument("--seed", type=int, default=6000,
                    help="labeler seed; MUST stay out of the reserved eval band "
                         f"[{EVAL_SEED_MIN}, {EVAL_SEED_MAX})")
    ap.add_argument("--states-jsonl", default=None,
                    help="relabel PRE-COLLECTED states (decision_eval "
                         "--collect-states output; DAgger round-2) instead of "
                         "rolling new behavior-policy episodes")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--curricula", default="poisson,truncnorm")
    ap.add_argument("--m-futures", type=int, default=16,
                    help="CRN futures per state for E[J] (>=16 per AUDIT-1 P1)")
    ap.add_argument("--theta", type=float, default=CONTINUATION_THETA)
    ap.add_argument("--lam", type=float, default=CONTINUATION_LAM)
    ap.add_argument("--tau-scale", type=float, default=0.5)
    ap.add_argument("--p-floor", type=float, default=0.75)
    ap.add_argument("--p-random", type=float, default=0.25)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--out", required=False,
                    help="out prefix (writes .jsonl + .npz + .manifest.json)")
    ap.add_argument("--hash-rows", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.states_jsonl:
        out = run_labeler_from_states(
            args.states_jsonl, seed=args.seed, horizon=args.horizon,
            m_futures=args.m_futures, theta=args.theta, lam=args.lam,
            tau_scale=args.tau_scale, processes=args.processes,
            out_prefix=args.out, hash_rows=args.hash_rows)
    else:
        out = run_labeler(
        args.seed, args.episodes, horizon=args.horizon,
        curricula=args.curricula.split(","), m_futures=args.m_futures,
        theta=args.theta, lam=args.lam, tau_scale=args.tau_scale,
        p_floor=args.p_floor, p_random=args.p_random, processes=args.processes,
        out_prefix=args.out, hash_rows=args.hash_rows)
    m = dict(out)
    rows = m.pop("rows")
    print(f"states={m['n_states']} flat={m['flat_rate']:.4f} "
          f"edge={m['edge_argmin_rate']:.4f} entropy={m['mean_target_entropy_nats']:.3f} "
          f"wall={m['wall_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())