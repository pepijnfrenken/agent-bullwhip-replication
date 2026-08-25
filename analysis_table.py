#!/usr/bin/env python3
"""Build the cost-vs-reliability frontier table for the agent-bullwhip paper.

Reads results/*.summary.json, aligns configs across models, and prints
the paper-ready table: mean cost, CV, Ψ/Φ medians, tail metrics, tokens,
and gate outcomes per the pre-registered decision rules (protocol.md §5).
"""
import json
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"

# Pre-registered gates (protocol.md §5): relative CV reductions vs baseline.
GATES = {
    "voting5": 0.10,   # H2: voting reduces CV by >=10%
    "budget": 0.20,    # H3: guardrail reduces CV by >=20%
    "anchor": 0.20,    # H4: anchor reduces CV by >=20% AND beats voting5
    "combined": 0.20,  # extension (no formal gate, report)
    "prompt_weighted": 0.10,
}


def load(model: str, config: str):
    f = RESULTS / f"{model}-{config}.summary.json"
    if not f.exists():
        return None
    return json.load(open(f))


def cv(d):
    return d.get("cv_cost")


def main():
    models = ["deepseek-v4-flash", "qwen3.6-35b"]
    configs = ["baseline", "budget", "prompt_weighted", "voting5",
               "anchor", "combined", "mirror", "order_up_to"]

    print("=" * 92)
    print("AGENT BULLWHIP — cost-vs-reliability frontier")
    print("=" * 92)

    for model in models:
        base = load(model, "baseline")
        if base is None:
            print(f"\n[{model}] no baseline yet — skipped\n")
            continue
        base_cv = cv(base)
        print(f"\n### {model}  (baseline mean_cost={base['mean_cost']:,.0f} cv={base_cv:.3f})")
        print(f"{'config':<16}{'mean':>13}{'cv':>8}{'min':>11}{'max':>11}{'p95':>11}"
              f"{'fail':>6}{'Ψmed':>7}{'Φmed':>7}{'tok':>9}  gate")
        for c in configs:
            d = load(model, c)
            if d is None:
                print(f"{c:<16}  (no summary)")
                continue
            psi_med = d.get("psi_median_upstream")
            phi_med = d.get("phi_median")
            psi_v = psi_med.get("factory", float("nan")) if isinstance(psi_med, dict) else (psi_med if psi_med is not None else float("nan"))
            phi_v = phi_med.get("factory", float("nan")) if isinstance(phi_med, dict) else (phi_med if phi_med is not None else float("nan"))
            toks = d.get("tokens", {})
            if isinstance(toks, dict):
                toks = toks.get("total", 0)
            gate = ""
            if c in GATES and base_cv:
                red = (base_cv - cv(d)) / base_cv if cv(d) is not None else None
                if red is not None:
                    gate = f"{'✅' if red >= GATES[c] else '❌'} Δcv {red:+.0%}"
                    if c == "anchor" and red >= GATES[c]:
                        v5 = load(model, "voting5")
                        if v5 and cv(v5) is not None:
                            gate += f" vsV5 {cv(d)/cv(v5):.2f}x"
            print(f"{c:<16}{d['mean_cost']:>13,.0f}{cv(d):>8.3f}{d.get('min_cost',0):>11,.0f}"
                  f"{d.get('max_cost',0):>11,.0f}{d.get('p95_cost',0):>11,.0f}"
                  f"{d.get('failure_rate',0):>6.3f}"
                  f"{psi_v:>7.2f}{phi_v:>7.2f}"
                  f"{(toks or 0):>9,}  {gate}")
    print("\n" + "=" * 92)


if __name__ == "__main__":
    sys.exit(main())
