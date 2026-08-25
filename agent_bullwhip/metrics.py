"""Metrics: paper's agent bullwhip (Psi/Phi), CV, tails, failure rates."""
from __future__ import annotations

import statistics

from .engine import ROLES, RunLog

EPS = 1e-9


def bullwhip_ratio(var_upstream: float, var_downstream: float) -> float:
    """Psi_k(t) = Var_r(q_k,t)/Var_r(q_{k-1,t}); >1 = amplification."""
    if var_downstream <= EPS:
        return float("nan") if var_upstream > EPS else 1.0
    return var_upstream / var_downstream


def compute_metrics(runs: list[RunLog]) -> dict:
    """Aggregate over N runs. Expects identical demand/config across runs.

    Returns cost stats, CV, per-echelon order variance over time, Psi/Phi,
    tail metrics, instruction-failure rate (from agent objects, passed via runs' agent attr).
    """
    costs = [r.total_cost() for r in runs]
    mean = statistics.mean(costs)
    std = statistics.stdev(costs) if len(costs) > 1 else 0.0
    cv = std / mean if mean else 0.0

    # per-echelon per-week order variance across runs
    T = len(runs[0].weeks)
    var_by_week: dict[str, list[float]] = {r: [] for r in ROLES}
    for t in range(T):
        for r in ROLES:
            vals = [run.weeks[t].orders[r] for run in runs]
            var_by_week[r].append(statistics.variance(vals) if len(vals) > 1 else 0.0)

    # agent bullwhip: Psi_k(t) = var_k / var_{k-1} per week; Phi_k(t) = var_{t+1}/var_t
    psi: dict[str, list[float]] = {}
    phi: dict[str, list[float]] = {}
    for i, r in enumerate(ROLES):
        if i == 0:
            psi[r] = [float("nan")] * T  # no downstream tier to compare at retailer
        else:
            downstream = ROLES[i - 1]
            psi[r] = [
                bullwhip_ratio(var_by_week[r][t], var_by_week[downstream][t]) for t in range(T)
            ]
        phi[r] = [
            bullwhip_ratio(var_by_week[r][t + 1], var_by_week[r][t]) for t in range(T - 1)
        ]

    # tails
    all_backlogs = [b for run in runs for w in run.weeks for b in w.backlog.values()]
    all_orders = [o for run in runs for w in run.weeks for o in w.orders.values()]

    # instruction failure rate from agent objects (if attached)
    agents = getattr(runs[0], "agents", None)
    failure_rate = None
    if agents:
        total_calls = sum(getattr(a, "calls", 0) for a in agents.values())
        total_fail = sum(getattr(a, "failures", 0) for a in agents.values())
        failure_rate = total_fail / total_calls if total_calls else 0.0

    def pct(xs: list, p: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
        return float(s[k])

    return {
        "n_runs": len(runs),
        "mean_cost": mean,
        "std_cost": std,
        "cv_cost": cv,
        "min_cost": min(costs),
        "max_cost": max(costs),
        "p95_cost": pct(costs, 0.95),
        "p99_cost": pct(costs, 0.99),
        "var_by_week": var_by_week,
        "psi": psi,
        "phi": phi,
        "psi_median_upstream": {
            r: statistics.median([v for v in psi[r] if v == v]) for r in ROLES if r != "retailer"
        },
        "phi_median": {r: statistics.median([v for v in phi[r] if v == v]) for r in ROLES},
        "p95_backlog": pct(all_backlogs, 0.95),
        "max_backlog": max(all_backlogs) if all_backlogs else 0,
        "max_order": max(all_orders) if all_orders else 0,
        "failure_rate": failure_rate,
    }
