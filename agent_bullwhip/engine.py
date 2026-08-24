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
    """Canonical demand patterns. 'step' = classic beer game jump (4 then 8)."""
    if pattern == "constant":
        return 4
    if pattern == "step":
        return 8 if t >= 4 else 4
    if pattern == "shock":
        return 12 if t == 12 else (8 if t >= 4 else 4)
    raise ValueError(f"unknown demand pattern: {pattern}")


def make_demand(horizon: int, pattern: str = "step") -> list[int]:
    return [step_demand(t, pattern) for t in range(horizon)]


def run_game(
    agents: dict[str, object],
    demand: list[int],
    cfg: SimConfig | None = None,
) -> RunLog:
    """agents: {role: callable(ctx) -> int order}. ctx carries the observed state."""
    cfg = cfg or SimConfig()
    horizon = cfg.horizon if cfg.horizon else len(demand)

    state = {r: TierState(cfg.init_on_hand, cfg.init_backlog, cfg.init_outstanding) for r in ROLES}
    # incoming shipment queue per tier (FIFO, length = shipment_delay); factory's queue
    # is filled by its own orders (unlimited upstream supplier, delayed).
    inbound: dict[str, deque[int]] = {r: deque([0] * cfg.shipment_delay) for r in ROLES}
    prev_incoming: dict[str, int] = {r: 0 for r in ROLES}   # q_{k-1,t-1} observed by agent
    order_history: dict[str, list[int]] = {r: [] for r in ROLES}
    log = RunLog(config=cfg, demand=demand[:horizon])

    for t in range(horizon):
        # 1) agents place orders simultaneously, on state known at start of week t.
        orders: dict[str, int] = {}
        for r in ROLES:
            incoming_now = demand[t] if r == "retailer" else orders[ROLES[ROLES.index(r) - 1]]
            # paper timing: order placed on info through t-1 -> use prev_incoming in
            # the prompt; but current-week incoming is also visible to the model (HBR).
            ctx = {
                "role": r,
                "t": t,
                "on_hand": state[r].oh,
                "backlog": state[r].b,
                "outstanding": state[r].o,
                "incoming_last": prev_incoming[r],
                "incoming_now": incoming_now,
                "last_orders": order_history[r][-6:],
            }
            q = agents[r](ctx)
            q = int(max(0, q)) if q is not None else 0
            orders[r] = q
            order_history[r].append(q)

        # 2) receipts arrive (shipments sent `shipment_delay` weeks ago).
        receipts: dict[str, int] = {}
        for r in ROLES:
            receipts[r] = inbound[r].popleft()

        # 3) shipments constrained by available inventory vs effective demand.
        shipments: dict[str, int] = {}
        for r in ROLES:
            incoming_eff = demand[t] if r == "retailer" else orders[ROLES[ROLES.index(r) - 1]]
            avail = state[r].oh + receipts[r]
            eff_demand = incoming_eff + state[r].b
            shipments[r] = min(avail, eff_demand)

        # 4) push shipments upstream into the next tier's inbound queue (factory: own orders).
        #    tier k receives what tier k+1 ships this week; factory receives its own orders
        #    (unlimited external supplier), both with `shipment_delay` weeks of lag.
        for i, r in enumerate(ROLES):
            if r == "factory":
                inbound[r].append(orders[r])
            else:
                inbound[r].append(shipments[ROLES[i + 1]])

        # 5) update state (paper eqs. 9-11).
        cost = 0.0
        for r in ROLES:
            incoming_eff = demand[t] if r == "retailer" else orders[ROLES[ROLES.index(r) - 1]]
            s = state[r]
            s.oh = s.oh + receipts[r] - shipments[r]
            s.b = s.b + incoming_eff - shipments[r]
            s.o = s.o + orders[r] - receipts[r]
            cost += cfg.holding_cost * s.oh + cfg.backlog_cost * s.b
            prev_incoming[r] = incoming_eff

        log.weeks.append(WeekRecord(
            t=t,
            orders=dict(orders),
            incoming={r: (demand[t] if r == "retailer" else orders[ROLES[ROLES.index(r) - 1]]) for r in ROLES},
            receipts=receipts,
            shipments=shipments,
            on_hand={r: state[r].oh for r in ROLES},
            backlog={r: state[r].b for r in ROLES},
            outstanding={r: state[r].o for r in ROLES},
            cost=cost,
        ))

    return log
