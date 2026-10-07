# Protocol — Agent Bullwhip Replication (pre-registered 2026-08-24)

This is the pre-registered experimental protocol for the agent bullwhip study.
We commit to this design BEFORE running the full matrix. Changes get logged here.

## 1. Research question

Can cheap inference-time reliability fixes (voting, guardrails, anchoring, prompt
reframing) reduce the agent bullwhip (run-to-run decision instability) of LLM
supply-chain agents, and does the paper's negative result on repeated sampling
generalize to free-API models?

## 2. Environment (fixed across runs)

- 4-echelon serial chain: retailer → wholesaler → distributor → factory
- Horizon: 36 weeks
- Demand: step pattern (4 for weeks 0–3, 8 for weeks 4–35), fixed across runs
- Shipment delay: 2 weeks; order transmission: same-week
- Holding cost: $1/unit/week; backlog cost: $2/unit/week
- Initial state: all zero (on-hand, outstanding, backlog)
- LLM: deepseek-v4-flash (FreeInference), temperature 0.7, max_tokens 64, retries 3
- N = 30 runs per (model × config), same demand path every run

## 3. Configs (confirmatory: A–C; exploratory: D–F)

- **A. baseline** — default decentralized LLM agents (replication anchor)
- **B. voting5 / voting10** — median of 5 / 10 samples per weekly order
- **C. guardrail** — order capped at max(0, 2 × incoming_last)
- **D. anchor** — order clamped to ES-forecast target ± 6 (our extension)
- **E. prompt_weighted** — "minimize weighted avg of backlog + holding costs" (§3.4 reframe)
- **F. combined** — guardrail + anchor + weighted prompt

Deterministic baselines (mirror, order-up-to) run as sanity + benchmarks.

## 4. Metrics

- Total cost = Σ_t Σ_k (holding·OH_{k,t} + backlog·B_{k,t})
- CV_cost = std(total cost) / mean(total cost) across 30 runs
- Ψ_k(t) = Var_r(q_{k,t}) / Var_r(q_{k−1,t}); Φ_k(t) = Var_r(q_{k,t+1}) / Var_r(q_{k,t})
- Tail: p95/p99 total cost, p95 backlog, max order
- Failure rate = unparseable LLM responses / total calls

## 5. Pre-registered decision rules (effect-size gates, no p-hacking)

- **H2 (voting):** supported only if votingN reduces CV_cost by ≥10% relative to baseline.
- **H3 (guardrails):** supported if guardrail reduces CV_cost by ≥20% relative to baseline
  (paper's GPT-4o mini showed 0.126→0.056 = −56%; we set a weaker gate for a cheap model).
- **H4 (anchoring):** supported if anchor reduces CV_cost by ≥20% relative to baseline
  AND beats voting5 (t-test or bootstrap CI on the CV difference, α=0.05).
- **H5 (model ladder):** report CV per model; no gate (descriptive, N=1 per model).
- All comparisons on the SAME 30 demand paths; report full distributions (min/max/quartiles).

## 6. Honest caveats (documented in every artifact)

- Protocol-specified replication, not bit-exact (authors' exact constants not public).
- Free-API models only; model updates possible mid-study (log model IDs + timestamps).
- Single demand pattern; no human baseline (anchor to deterministic baselines + paper's normalized humans).

## 7. Changes log

- (none yet — pre-registered)

## 8. Deliverables

Repo (engine/agents/metrics/runner), results/ (JSONL + summaries), plots,
writeup (X → LinkedIn per socials KB), KB update.
