"""Deterministic beer-game engine faithful to arXiv 2605.17036 §5.1 operational model.

Four echelons: retailer -> wholesaler -> distributor -> factory (tier k=1..4),
customer demand is tier 0. Shipments take `shipment_delay` weeks; orders are
transmitted same-week (matches the paper's timing: q_{k-1,t} enters tier k's
demand at t, while the period-t order is placed on info through t-1).

State per tier (paper eqs. 9-11):
    OH_{t+1} = OH_t + r_t - s_t
    B_{t+1}  = B_t  + q_{k-1,t} - s_t
    O_{t+1}  = O_t  + q_t - r_t
    IP_t     = OH_t + O_t - B_t   (paper eq. 2; Prop 1: IP_{t+1} = IP_t + q_t - q_{k-1,t})
Receipts: r_{k,t} = s_{k+1,t-lag} (factory: its own orders, unlimited upstream).
Shipments: s_{k,t} = min(OH_t + r_t, q_{k-1,t} + B_t).

`GameStepper` holds this week loop; `run_game` drives it synchronously (one role at a
time) and the batched rollout driver in `grpo_env` drives many episodes in lockstep
(STEER-01: one implementation, no mirror). The week semantics must not change silently:
`tests/test_engine_golden.py` pins per-week orders + costs for the waves 1-5 settings.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

ROLES = ["retailer", "wholesaler", "distributor", "factory"]


@dataclass
class SimConfig:
    horizon: int = 36
    shipment_delay: int = 2          # weeks (paper default)
    holding_cost: float = 1.0        # per unit per week
    backlog_cost: float = 2.0        # per unit per week
    init_on_hand: int = 0
    init_outstanding: int = 0
    init_backlog: int = 0


@dataclass
class TierState:
    oh: int = 0
    b: int = 0
    o: int = 0


@dataclass
class WeekRecord:
    t: int
    orders: dict[str, int] = field(default_factory=dict)        # q_{k,t}
    incoming: dict[str, int] = field(default_factory=dict)      # q_{k-1,t} (d_t for retailer)
    receipts: dict[str, int] = field(default_factory=dict)
    shipments: dict[str, int] = field(default_factory=dict)
    on_hand: dict[str, int] = field(default_factory=dict)
    backlog: dict[str, int] = field(default_factory=dict)
    outstanding: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class RunLog:
    config: SimConfig
    demand: list[int]
    weeks: list[WeekRecord] = field(default_factory=list)

    def total_cost(self) -> float:
        return sum(w.cost for w in self.weeks)

    def orders(self, role: str) -> list[int]:
        return [w.orders[role] for w in self.weeks]

    def backlogs(self, role: str) -> list[int]:
        return [w.backlog[role] for w in self.weeks]

    def on_hand(self, role: str) -> list[int]:
        return [w.on_hand[role] for w in self.weeks]


def step_demand(t: int, pattern: str = "step") -> int:
    """Canonical demand patterns. 'step' = classic beer game jump (4 then 8).

    'noisy' = step + AR(1)-style noise (seeded per-run via make_demand seed):
    tests whether reliability conclusions survive when the deterministic floor
    itself faces variance. Noise: d_t = step_demand + 0.5*(d_{t-1}-step_{t-1}) + eps,
    eps ~ U{-2,-1,0,1,2}, clipped to [0, 20].
    """
    if pattern == "constant":
        return 4
    if pattern == "step":
        return 8 if t >= 4 else 4
    if pattern == "shock":
        return 12 if t == 12 else (8 if t >= 4 else 4)
    if pattern == "noisy":
        raise ValueError("noisy needs make_demand (stateful)")
    raise ValueError(f"unknown demand pattern: {pattern}")


def make_demand(horizon: int, pattern: str = "step", seed: int | None = None) -> list[int]:
    """Demand vector. Deterministic patterns are identical across runs (that's the
    point: CV isolates agent instability). 'noisy' uses a per-call RNG; pass a
    seed for reproducibility (runner seeds per-run when pattern='noisy' so each
    run sees a fresh realization while staying reproducible)."""
    if pattern == "noisy":
        import random
        rng = random.Random(seed)
        d: list[int] = []
        prev_noise = 0.0
        for t in range(horizon):
            base = step_demand(t, "step")
            prev_noise = 0.5 * prev_noise + rng.choice([-2, -1, 0, 1, 2])
            d.append(max(0, min(20, base + int(round(prev_noise)))))
        return d
    if pattern == "chaotic":
        # Regime-switching: demand jumps between low/mid/high regimes, with AR(1)
        # noise and occasional big shocks. Non-stationary but structured — a tuned
        # fixed-parameter formula should start to degrade here (its fit goes stale
        # when the regime moves).
        import random
        rng = random.Random(seed)
        d: list[int] = []
        regime = rng.choice([4, 8, 16])
        prev_noise = 0.0
        for t in range(horizon):
            if rng.random() < 0.08:  # regime switch ~every 12 weeks
                regime = rng.choice([4, 8, 16])
            prev_noise = 0.5 * prev_noise + rng.choice([-3, -2, -1, 0, 1, 2, 3])
            val = regime + prev_noise
            if rng.random() < 0.05:  # shock ~2x per horizon
                val += rng.choice([-12, -10, 10, 12, 16])
            d.append(max(1, min(40, int(round(val)))))
        return d
    if pattern == "wild":
        # Geometric random walk + rare jumps/crashes. Multiplicative and
        # non-stationary — a fixed (theta, lambda) smoothing formula cannot track
        # this; this is the environment where an adaptive agent should finally
        # earn its keep.
        import random
        rng = random.Random(seed)
        d: list[int] = []
        val = 8.0
        for t in range(horizon):
            shock = rng.random()
            if shock < 0.06:
                val *= rng.choice([2.5, 3.0, 4.0])  # rare jump up
            elif shock < 0.10:
                val *= rng.choice([0.25, 0.35])      # rare crash
            else:
                val *= 1.0 + rng.gauss(0, 0.25)     # multiplicative noise
            val = max(1.0, min(60.0, val))
            d.append(int(round(val)))
        return d
    return [step_demand(t, pattern) for t in range(horizon)]


class GameStepper:
    """One week of the engine's operational model, as a shared stepper.

    STEER-01: `run_game` (synchronous, one role at a time) and the batched rollout
    driver in `grpo_env` (many episodes in lockstep, one generate() call per week)
    both drive THIS code, so the week semantics cannot drift between them.

    Contract per week:
        ctxs = step.observations()          # ctx per role; mutates nothing
        step.advance({role: order, ...})    # consumes the orders, appends a WeekRecord

    `observations()` may be called repeatedly; `advance` requires an order for every
    role (missing roles raise: a silently-defaulted order would corrupt a rollout)
    and clamps negatives to 0 (the operational model's floor).
    """

    def __init__(self, demand: list[int], cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        horizon = self.cfg.horizon if self.cfg.horizon else len(demand)
        self.horizon = horizon
        self.demand = list(demand[:horizon])
        self.t = 0
        self.state = {r: TierState(self.cfg.init_on_hand, self.cfg.init_backlog,
                                  self.cfg.init_outstanding) for r in ROLES}
        # incoming shipment queue per tier (FIFO, length = shipment_delay); factory's
        # queue is filled by its own orders (unlimited upstream supplier, delayed).
        self.inbound: dict[str, deque[int]] = {
            r: deque([0] * self.cfg.shipment_delay) for r in ROLES}
        self.prev_incoming: dict[str, int] = {r: 0 for r in ROLES}  # q_{k-1,t-1}
        self.order_history: dict[str, list[int]] = {r: [] for r in ROLES}
        self.weeks: list[WeekRecord] = []

    def done(self) -> bool:
        return self.t >= self.horizon

    def observations(self) -> dict[str, dict]:
        """Observation for every role at the start of the current week.

        AUDIT2 FIX: only the retailer sees the CURRENT week's exogenous customer
        demand. Upstream tiers must NOT see the tier-below's same-week order (that is
        a one-week lookahead the deterministic floor doesn't get, and it systematically
        advantages the LLM — see AUDIT2.md §3). They decide on information through t-1
        only (`incoming_last`). This is also why a week's four decisions are mutually
        independent and may be batched into one model call.
        """
        ctxs: dict[str, dict] = {}
        for r in ROLES:
            ctxs[r] = {
                "role": r,
                "t": self.t,
                "on_hand": self.state[r].oh,
                "backlog": self.state[r].b,
                "outstanding": self.state[r].o,
                "incoming_last": self.prev_incoming[r],
                "incoming_now": self.demand[self.t] if r == "retailer" else None,
                "last_orders": self.order_history[r][-6:],
            }
        return ctxs

    def advance(self, orders: dict[str, int]) -> WeekRecord:
        """Apply one week's orders (steps 2-5 + bookkeeping) and record the week."""
        missing = [r for r in ROLES if r not in orders]
        if missing:
            raise ValueError(f"week {self.t}: no order for {missing}")
        t = self.t
        ords = {r: max(0, int(orders[r])) for r in ROLES}   # engine floor: q >= 0
        for r in ROLES:
            self.order_history[r].append(ords[r])

        # 2) receipts arrive (shipments sent `shipment_delay` weeks ago).
        receipts: dict[str, int] = {}
        for r in ROLES:
            receipts[r] = self.inbound[r].popleft()

        # 3) shipments constrained by available inventory vs effective demand.
        shipments: dict[str, int] = {}
        for r in ROLES:
            incoming_eff = self.demand[t] if r == "retailer" else ords[ROLES[ROLES.index(r) - 1]]
            avail = self.state[r].oh + receipts[r]
            shipments[r] = min(avail, incoming_eff + self.state[r].b)

        # 4) push shipments upstream into the next tier's inbound queue (factory: own
        #    orders). Tier k receives what tier k+1 ships this week, with `shipment_delay`
        #    weeks of lag.
        for i, r in enumerate(ROLES):
            self.inbound[r].append(ords[r] if r == "factory" else shipments[ROLES[i + 1]])

        # 5) update state (paper eqs. 9-11) and account the week's cost.
        cost = 0.0
        incoming: dict[str, int] = {}
        for r in ROLES:
            incoming_eff = self.demand[t] if r == "retailer" else ords[ROLES[ROLES.index(r) - 1]]
            s = self.state[r]
            s.oh = s.oh + receipts[r] - shipments[r]
            s.b = s.b + incoming_eff - shipments[r]
            s.o = s.o + ords[r] - receipts[r]
            cost += self.cfg.holding_cost * s.oh + self.cfg.backlog_cost * s.b
            self.prev_incoming[r] = incoming_eff
            incoming[r] = incoming_eff

        rec = WeekRecord(
            t=t,
            orders=ords,
            incoming=incoming,
            receipts=receipts,
            shipments=shipments,
            on_hand={r: self.state[r].oh for r in ROLES},
            backlog={r: self.state[r].b for r in ROLES},
            outstanding={r: self.state[r].o for r in ROLES},
            cost=cost,
        )
        self.weeks.append(rec)
        self.t += 1
        return rec

    def log(self) -> RunLog:
        return RunLog(config=self.cfg, demand=self.demand, weeks=self.weeks)


def run_game(
    agents: dict[str, object],
    demand: list[int],
    cfg: SimConfig | None = None,
) -> RunLog:
    """agents: {role: callable(ctx) -> int order}. ctx carries the observed state."""
    cfg = cfg or SimConfig()
    step = GameStepper(demand, cfg)
    while not step.done():
        # 1) agents place orders on state known at the start of the week (see
        #    GameStepper.observations for the AUDIT2 no-lookahead rule)
        ctxs = step.observations()
        orders: dict[str, int] = {}
        for r in ROLES:
            q = agents[r](ctxs[r])
            q = int(max(0, q)) if q is not None else 0
            orders[r] = q
        step.advance(orders)
    return step.log()
