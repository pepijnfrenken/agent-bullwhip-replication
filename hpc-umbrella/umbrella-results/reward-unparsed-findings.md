# Reward vs unparsed rollouts — findings (sweep-363909 open item)

**Scope:** what the GRPO reward does with completions from which no order can be
parsed, per the sweep's standing open item. Read path: `agent_bullwhip/grpo_train.py`,
`grpo_env.py`, `grpo_trl.py`; evidence from `umbrella-results/results/grpo_*/train_log.jsonl`
(a0_base + the four wave-6 variants, 300 steps each). This matters before A1 trains.

## Finding: the reward does nothing special with unparsed samples

There is no parse-awareness anywhere in the reward or update path. The complete chain:

1. **Parse** — `grpo_env.extract_int()` (`grpo_env.py:175`): first signed integer in the
   completion, else **0**. `run_episodes_batched` feeds that straight into the game:
   `q = int(parse(text))` (`grpo_env.py:305`) and `stepper.advance(orders[g])`. An unparsed
   completion is therefore a **real order-0 decision** in the episode — not skipped, not
   clamped out. `has_int()` (`grpo_env.py:187`) exists only as the discriminator behind the
   per-step *parse_failure_rate metric* (`grpo_train.py:908,930`); no code path consults it
   outside the metric.
2. **Reward** — episode weekly system costs `sys_cost_by_week` → `return_to_go_rewards`
   (`grpo_env.py:71`): r_{k,t} = −Σ_{τ≥t} c_τ, *identical for all four agents*. No term
   depends on parseability.
3. **Advantage** — `group_normalize` (`grpo_env.py:325`): per-(agent, week) standardization
   across the step's G episodes. Same normalization for unparsed and parsed rows alike.
4. **Update** — TRL-side reward closure (`grpo_trl.py:71-88`) returns the precomputed
   advantage per row verbatim (`batch.advantages[cursor…]`) — no filtering, no masking, no
   extra penalty (e.g., no "unparseable ⇒ −X" term anywhere). `grpo_step` scores **all**
   completions' token logprobs by that advantage (`grpo_train.py:556`); `policy_push` is a
   detached diagnostic of update magnitude (`grpo_train.py:128`), not a mechanism.

Consequence: for an unparsed rollout the tokens that get gradient are the tokens of the
rambling text, weighted by the cell-level advantage of a decision that entered the game as
order 0. There is no token-level credit for emitting a number, and no penalty for not
emitting one.

## Evidence (train_log.jsonl, 300 steps each)

| run | parse-fail mean | std | first50 | last50 | corr(parse-fail, epcost) | corr(parse-fail, mean_adv) | final order_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| a0_base    | 0.847 | 0.013 | 0.840 | 0.859 | −0.217 | −0.111 | 8.3 |
| a11_seed2  | 0.867 | 0.021 | 0.840 | 0.894 | −0.047 | −0.102 | 0.2 |
| a7_kl_low  | 0.845 | 0.012 | 0.840 | 0.851 | −0.069 | +0.064 | 8.3 |
| a9_lr_low  | 0.845 | 0.012 | 0.839 | 0.855 | +0.022 | −0.026 | 2.8 |
| a10_lr_high| 0.862 | 0.021 | 0.845 | 0.896 | −0.210 | −0.048 | 0.3 |

- **~85% of decisions are trained as order-0 with a cell-level advantage.** Parse failure
  is uniformly high across all five configs (0.845–0.867 mean) and does **not** improve with
  training (last50 ≥ first50 in every run). The higher-lr/higher-kl runs (a11, a10) drift
  *up* most (0.840→0.894 / 0.845→0.896).
- **No adaptive coupling**: |corr(parse-fail, episode cost)| ≤ 0.22 and |corr with mean
  advantage| ≤ 0.11 across all runs — parse failure is not correlated with the signal the
  reward produces, in either direction.
- **Sample texts** (step 299, all sampled completions unparsed): a0_base `"So, you have a
  choice to order…"`, a10_lr_high `"Please show your final answer within the box…"` — the
  model still produces reasoning-shaped prose with no integer at the end of training. The
  parsed minority yields tiny final order_means (0.2–8.3); e.g., parsed a0_base step-49
  `order=1 :: "Constraints: 1. If a current…"` — the number is usually a small incidental
  integer, not a deliberate quantity.

## Reading + implication for A1

At 0.6B this is a structural property of the reward design, not a config fluke: with a
system-level return-to-go reward applied uniformly and a first-int-or-0 parse, ~85% of the
gradient comes from unparsed prose carrying a cell-level advantage. Two things follow for A1:

1. **Expect the same 85% unparsed-rollout regime at any base size** unless the reward adds
   parse structure (a numeric-completion constraint, a parse bonus/penalty term, or forcing
   the completion format). The wave-6 variants are all trained inside this regime, so the
   eval pairing (converted LoRA → f16 GGUF on the same raw-continuation harness) remains the
   fair comparison.
2. **A1's decision-relevant question is not "does the reward punish unparsed" (it does not)
   but "is the *parsed minority* improving enough to beat the control"—** which is precisely
   what the wave-6 step-arm paired analysis measures. Any downstream reward change should be
   validated against the same paired eval protocol, and the parse-failure rate should be
   tracked per wipe as a first-class metric (it is already logged per step).