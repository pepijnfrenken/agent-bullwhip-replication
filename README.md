# Agent Bullwhip — Replication & Extension

**Can cheap, inference-time reliability fixes reduce the "agent bullwhip" of LLM supply-chain agents?**

Independent replication + extension of **arXiv 2605.17036** — *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long, Simchi-Levi, Zhu, Su, Calmon & Calmon, Harvard/MIT/Georgia Tech).

## TL;DR (post-audit, audit-surviving)

**Where the intelligence in an "AI agent" actually lives: the deterministic wrapper, not the model.** We replicated the paper's Beer Game with free-API LLM agents and found:

1. **Plain LLM agents are instability machines** — baseline cost CV 4.97 (fixed demand), worse than the paper's 0.13–0.46. The paper's reliability concern replicates, amplified.
2. **A deterministic anchor (order-up-to) + confidence-gate wrapper tames them completely** — `kb_system_gated` = 4,113 / CV 0.073 (fixed, interleaved), ~12% above the floor, and 10–22× better than the ungated twin in the same windows.
3. **But the wrapper, not the LLM, is the entire win** — wrapper-only ablation = exactly 3,681 (the floor); the LLM inside adds +12–34% cost on fixed demand. The LLM is a passenger being pulled to safety.
4. **Under noisy demand, a *tuned* formula ties the LLM** — a 1-minute (θ,λ) sweep (4,701) matches the gated LLM (4,849); the apparent "LLM beats the formula" result was an artifact of benchmarking against an *untuned* (3.0, 0.5) and a same-week information leak that has now been fixed.
5. **The apparent LLM advantage under uncertainty was lookahead, not reasoning** — upstream tiers were seeing the tier-below's current-week order (`incoming_now`); a deterministic floor given the same information scored 37–40% below the LLM. **That leak is now removed** (see AUDIT2 §3, fix applied).

**Honest conclusion: on this task, deterministic structure + tuned formulas beat LLM judgment on cost; LLM agents only appear competitive when handed information the formulas don't get. The "agent" is the wrapper.**

*(Full evidence: `AUDIT.md`, `AUDIT2.md`, `results/interleaved_val2/`, `results/wrapper_only/`, `results/noisy_interleaved/`.)*

## The question

The paper showed LLM agents managing a 4-echelon Beer Game **beat human teams by up to 67% on average cost** — but are **unreliable**: run the same model 30× on the same demand path and costs vary 13–46% (CV). They call this the **agent bullwhip**: decision instability amplifies upstream and over time. Their fix (GRPO RL post-training) is out of reach for most teams.

**We test whether cheap inference-time fixes work instead** — self-consistency voting, order guardrails, ES-anchoring, prompt reframing, knowledge-base playbooks, introspection + confidence-gating — on free-API models (deepseek-v4-flash), and whether the paper's "repeated sampling doesn't help" result generalizes.

## Setup

```bash
cd agent-bullwhip-replication
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # or: pip install -e .
# env: FREEINFERENCE_API_KEY (set), optional FREEINFERENCE_MODEL (default deepseek-v4-flash)
```

## Run

```bash
# smoke test: 5 runs, 10 weeks
.venv/bin/python -m agent_bullwhip.runner --config baseline --runs 5 --horizon 10

# full matrix (30 runs, 36 weeks each; see run_all.sh)
bash run_all.sh

# robustness arm: noisy demand (AR(1) noise on the step), 30 runs
.venv/bin/python -m agent_bullwhip.runner --config kb_system_gated --runs 30 --horizon 36 --pattern noisy

# list endpoint models
.venv/bin/python -m agent_bullwhip.runner --models
```

Results: `results/<config>.jsonl` (per-run) + `<config>.summary.json` (metrics).

## Configs

| Config | What it tests | Paper link |
|---|---|---|
| `baseline` | default decentralized LLM agents | replication anchor |
| `budget` | order cap = 2× recent incoming (paper's "budget" policy) | their Table 1 (GPT-4o mini CV 0.126→0.056) |
| `prompt_weighted` | "minimize weighted avg of backlog+holding" | their §3.4 |
| `voting5` | self-consistency voting (median of 5 samples) | their §4.3 negative result |
| `anchor` | ES-forecast target anchoring (±6) | our extension (§5: kill the decision-shock channel) |
| `combined` | budget + anchor + weighted prompt | our best structural config |
| `kb` / `kb_system` / `kb_pointer` | decision-playbook injection (inline / system msg / pointer-only) | our Wave 2 |
| `introspect` / `kb_*_introspect` | reason + confidence traces, logged per decision | our Wave 2 |
| `kb_*_gated` | low-confidence → fallback to deterministic order-up-to anchor | our Wave 2 (the winners) |
| `kb_general` | domain-agnostic control principles (transfer test) | our Wave 2d |
| `kb_pointer_verbal` | verbal-consistency gate (says "cover backlog" but orders 0 → anchor) | our Wave 2e (trace-driven) |
| `mirror` / `order_up_to` | deterministic baselines | sanity + benchmarks |

**Wave 1** (paper's own levers): baseline, budget, prompt_weighted, voting5.
**Wave 2** (our extensions): anchor, combined, all kb/introspect/gated configs, kb_general, kb_pointer_verbal.
Both waves run on deepseek-v4-flash (and qwen3.6-35b for Wave 1) → cost-vs-reliability frontier.

## Metrics

- Total supply-chain cost (holding $1, backlog $2 per unit/week)
- **CV of total cost across runs** — headline reliability metric
- **Agent bullwhip:** Ψ_k(t) = Var_r(q_k,t)/Var_r(q_{k−1,t}) (upstream amplification), Φ_k(t) = Var_r(q_k,t+1)/Var_r(q_k,t) (intertemporal)
- Tail risk: p95/p99 backlog, max order
- Instruction-following failure rate
- **Prompt-hash collision check** (`prompt_hash_dups` in summary): identical exact prompts across runs (cache-collision risk)

## Key findings (so far)

1. **Introspection dominates KB choice.** Once you add reason + confidence, all KB placements converge within ~2×. The KB placement matters only *without* introspection (pointer > system > inline).
2. **The confidence gate is the cost killer.** Gated configs land at 4.1–4.8K mean cost / CV 0.09–0.29 — within 11–30% of the deterministic floor. The gate fires on *low confidence*; the trace mining shows the model is often *confidently wrong* (says "cover backlog", orders 0, conf 0.7–0.85) — which the verbal-consistency gate (Wave 2e) targets.
3. **The general KB works.** Domain-agnostic principles (124K) beat the hyper-fitted beer-game rules in the no-introspect setting — the transferable control principles are genuinely useful, not overfit.
4. **Repeated sampling doesn't help** (confirms the paper's negative result): voting5 ≈ baseline.
5. **The deterministic floor is the sanity bound.** order_up_to: 3,681 / CV 0. The LLM's job is to get *close* while staying adaptive — and gated introspection gets within 11%.

## Status

- [x] Engine (deterministic, paper §5.1 operational model; Prop 1 IP-recursion verified by test)
- [x] Baselines (mirror, order-up-to with ES smoothing) + LLM agent with wrappers
- [x] Metrics (cost, CV, Ψ/Φ, tails, failures, prompt-hash collisions)
- [x] Runner (CLI, JSONL + summary), tests (41 passing)
- [x] Live smoke test (deepseek-v4-flash, 40 calls, 0 failures — baseline LLM chaotic as the paper predicts)
- [x] Full 30-run matrix (deepseek-v4-flash, all configs)
- [x] Wave 2d general-KB transfer test + Wave 2e verbal-consistency gate
- [ ] Model ladder (qwen3.6-35b for Wave 2, minimax-m3 available on endpoint)
- [ ] Noisy-demand robustness arm (in progress)
- [ ] Independent confound audit (herdr OMP)
- [ ] Plots + writeup (X/LinkedIn per socials KB)

## Honest limits

- Protocol-specified replication, not bit-exact: the authors' exact demand vector/cost constants aren't public; we use canonical values (step demand 4→8; holding 1, backlog 2).
- Free-API models ≠ their frontier models; the cheap-model reliability gap is our contribution.
- No human baseline (anchor to their normalized human costs + deterministic baselines).
- CV is measured on a **fixed** demand path — that isolates agent instability (the paper's design), but the deterministic floor trivially wins on the *easiest* scenario. The noisy-demand arm tests whether the reliability advantage survives when the floor itself faces variance.

## Links

- KB: `~/.omp/knowledge/supply-chain-ai/` (paper deep-dive, study design, sources)
- Paper: https://arxiv.org/abs/2605.17036
- Audit: `AUDIT.md` (independent confound audit — see Status)
