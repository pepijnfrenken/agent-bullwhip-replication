# Agent Bullwhip — Replication & Extension

**Can cheap, inference-time reliability fixes reduce the "agent bullwhip" of LLM supply-chain agents?**

Independent replication + extension of **arXiv 2605.17036** — *Reliability and Effectiveness of Autonomous AI Agents in Supply Chain Management* (Long, Simchi-Levi, Zhu, Su, Calmon & Calmon, Harvard/MIT/Georgia Tech).

## The question

The paper showed LLM agents managing a 4-echelon Beer Game **beat human teams by up to 67% on average cost** — but are **unreliable**: run the same model 30× on the same demand path and costs vary 13–46% (CV). They call this the **agent bullwhip**: decision instability amplifies upstream and over time. Their fix (GRPO RL post-training) is out of reach for most teams.

**We test whether cheap inference-time fixes work instead** — self-consistency voting, order guardrails, ES-anchoring, prompt reframing — on free-API models (deepseek-v4-flash), and whether the paper's "repeated sampling doesn't help" result generalizes.

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
| `combined` | budget + anchor + weighted prompt | our best-config |
| `mirror` / `order_up_to` | deterministic baselines | sanity + benchmarks |

**Wave 1** (paper's own levers): baseline, budget, prompt_weighted, voting5.
**Wave 2** (our extensions): anchor, combined.
Both waves run on both models (deepseek-v4-flash, qwen3.6-35b) → cost-vs-reliability frontier.

## Metrics

- Total supply-chain cost (holding $1, backlog $2 per unit/week)
- **CV of total cost across runs** — headline reliability metric
- **Agent bullwhip:** Ψ_k(t) = Var_r(q_k,t)/Var_r(q_{k−1,t}) (upstream amplification), Φ_k(t) = Var_r(q_k,t+1)/Var_r(q_k,t) (intertemporal)
- Tail risk: p95/p99 backlog, max order
- Instruction-following failure rate

## Status

- [x] Engine (deterministic, paper §5.1 operational model; Prop 1 IP-recursion verified by test)
- [x] Baselines (mirror, order-up-to with ES smoothing) + LLM agent with wrappers
- [x] Metrics (cost, CV, Ψ/Φ, tails, failures)
- [x] Runner (CLI, JSONL + summary), tests (12 passing)
- [x] Live smoke test (deepseek-v4-flash, 40 calls, 0 failures — baseline LLM chaotic as the paper predicts)
- [ ] Full 30-run matrix
- [ ] Model ladder (qwen3.6-35b, minimax-m3 available on endpoint)
- [ ] Plots + writeup (X/LinkedIn per socials KB)

## Honest limits

- Protocol-specified replication, not bit-exact: the authors' exact demand vector/cost constants aren't public; we use canonical values (step demand 4→8; holding 1, backlog 2).
- Free-API models ≠ their frontier models; the cheap-model reliability gap is our contribution.
- No human baseline (anchor to their normalized human costs + deterministic baselines).

## Links

- KB: `~/.omp/knowledge/supply-chain-ai/` (paper deep-dive, study design, sources)
- Paper: https://arxiv.org/abs/2605.17036
