# WAVE6-FIX_SPEC.md — contract/format micro-probe on the wave-6 free-form GRPO lane (DRAFT — pre-registration, nothing below has run)

**Status:** IMPLEMENTED 2026-10-07 (mission `bullwhip-impl`); code + tests + staged sbatch only — **no run has happened**, no gate has
been read. The spec below is the unchanged pre-registration; every implementation decision and deviation is in the Changes log at
the bottom. Per repo protocol (root `README.md` §research protocol), changes after approval are logged there.

**Layout note.** Wave docs are cited by FILENAME and follow the post-cleanup layout: `WAVE6_SPEC.md`, `WAVE7.md`,
`WAVE7_SPEC.md`, `W7-READY.md` live in `docs/waves/`; `WAVE7-AUDIT-1.md`/`-2.md` in `docs/audits/`; `VERIFY-W7-EVAL.md`
in `docs/verify/`; `STATE.md`, `ARTIFACTS.md` in `docs/`. Code/test/cluster paths (`agent_bullwhip/*`, `tests/*`,
`scripts/*`, `hpc-umbrella/*`) are stable and cited exactly (file:line at repo HEAD `9f04e59`, 2026-10-07; re-read lines
before editing — they are hints).

**Lane:** wave 6 free-form lane (`agent_bullwhip/grpo_train.py` + `grpo_env.py`). Wave-7 (decision-native) files and
verdicts are untouched. This probe does NOT replace wave 6's eval verdict (`eval-a0-report.md`); it decides how that
verdict may be read.

---

## 0. Question and why it matters

**Question.** Is the wave-6 null (GRPO on Qwen3-0.6B, free-form lane: parse-fail ≈ 0.85 for 300 steps, trained arm no
better on aggregate cost) a *training result*, or an *artifact of how the answer was requested and parsed*?

**Why it matters.** `docs/STATE.md` OPEN ISSUE 1 already says the wave-6 numbers "cannot yet be read as a test of the
paper's GRPO claim". Until this probe answers, wave 6 is a fidelity blocker: the project cannot say whether "GRPO did not
transfer to 0.6B" or "we never gave GRPO a decision to optimise". This probe is the cheapest experiment that separates
the two (≈ 25 GPU-h, free cluster L4s), and its outcome fixes which follow-up is worth spending on (§9).

**What this probe is NOT.** It does not re-test the paper's reliability claim, does not use the eval protocol
(step n=30 / noisy n=10), and trains ≤ 100 steps. It measures whether the lane can *emit a parseable typed decision at
all* and whether the training signal reaches the decision tokens.

---

## 1. Verified facts this spec rests on

Every row was re-derived from the repo on 2026-10-07 (HEAD `9f04e59`). "Verify" gives the command that reproduces it.

| # | Orchestrator finding | Verdict | Evidence + verify command |
|---|---|---|---|
| 1 | Parse-fail 84.5–86.7 % across `a0_base` + 4 variants, flat over 300 steps | **VERIFIED (and wider)** | Per-arm mean `parse_failure_rate` over 300 steps: a0_base 0.8472, a7_kl_low 0.8446, a9_lr_low 0.8445, a10_lr_high 0.8624, a11_seed2 0.8665 (= the five arms of `hpc-umbrella/umbrella-results/reward-unparsed-findings.md`); all 12 completed arms span 0.843–0.867. a0_base windows: steps 0–49 mean 0.8401 (sd 0.013), steps 45–49 0.8491, 95–99 0.8377, last-50 0.8586 — never improves. Verify: `python3 -c "import json,glob,statistics as s;[print(p.split('/')[-2], round(s.mean(json.loads(l)['parse_failure_rate'] for l in open(p) if '\"parse_failure_rate\"' in l),4)) for p in sorted(glob.glob('hpc-umbrella/umbrella-results/results/grpo_*/train_log.jsonl')) if sum('\"parse_failure_rate\"' in l for l in open(p))>10]"` (run from the repo root; the 4-step pilot dirs are skipped). Caveat: this is the *training-time regex* rate (`has_int`); eval-harness parse-fail is 0.34–0.40 (`eval-a0-report.md`, "Parse-failure counting differs") — different counter, different completion budget, not a contradiction. |
| 2 | Reward path is parse-blind; unparsed → `extract_int()` = 0 → real order-0 decision; no mask/penalty; same advantage for all 4 agents | **VERIFIED** | `agent_bullwhip/grpo_env.py:175-184` `extract_int` → 0 if no int; `grpo_env.py:305` `q = int(parse(text))`, `:310` `stepper.advance(orders[g])`; `grpo_env.py:71-79` `return_to_go_rewards` returns the *same* list for all K=4 agents (system-level scope); `grpo_env.py:325-343` `group_normalize` (per agent-week across G); `grpo_train.py:684` `flat_adv.append(a[rec.agent_id][rec.t])`; `grpo_trl.py:54-56`; `grpo_train.py:556` (the `policy_push` diagnostic) and `:501-503` (loss uses every row's advantage); `has_int` (`grpo_env.py:187`) is consulted only for the metric at `grpo_train.py:908,930`. Chain documented in `reward-unparsed-findings.md`. |
| 3 | Lane serves the instruct checkpoint as RAW continuation without the chat template; wave-7 uses the template on the same checkpoint and reports 0 parse failures | **VERIFIED, with one nuance** | Checkpoint: `Qwen/Qwen3-0.6B` snapshot `c1899de289a04d12100db370d81485cdf75e47ca` is the post-trained hybrid checkpoint (README `base_model: Qwen/Qwen3-0.6B-Base`, "Training Stage: Pretraining & Post-training") and ships a 4,168-char `chat_template` (`tokenizer_config.json`, ends `<|im_start|>assistant\n` + `<think>\n\n</think>\n\n` when `enable_thinking is false`). Wave-6 path: `grpo_train.py:257` `tok(prompts, …add_special_tokens=True)` on the raw `build_prompt` string; `:332` and `:449` re-tokenize the same raw string for scoring; `client.py:157-164` serves eval via llama.cpp `/completion` with the raw prompt (`eval-a0-report.md`: "no chat template"). `grep -rn "apply_chat_template\|chat_template" agent_bullwhip/grpo_*.py agent_bullwhip/client.py agent_bullwhip/runner.py` → no hits; hits only in `decision_readout.py:144-161` (wave 7: `apply_chat_template(…, enable_thinking=False)`) and probes. **Nuance:** wave-7's "parse_fail = 0" (`results/eval/w7-*.summary.json`) is true *by construction* — the answer is read from next-token logits over option letters, there is no text to parse. The relevant wave-7 evidence for this probe is different: letter mass ≥ 0.9976 on the untrained templated readout (`WAVE7.md` §1) = the model follows a *format instruction in the first answer token* when templated. It says nothing yet about holding a multi-token JSON line under sampling at T=1.0 — that is exactly what A1 tests. |
| 4 | Paper's own answer contract (§A.2): JSON on its own line, `"order_quantity": <n>`, "parsed by a program, so the format must be valid and exact", with an example; the repo implements no such contract | **VERIFIED** | Paper text (`context/paper-2605.17036v3.txt`, Appendix A.2, lines 719-746), quoted in §3.1. `grep -rn "order_quantity" --include=*.py --include=*.md --include=*.sh .` (excluding `results/`, `docs/archive/`) → the only hit is `docs/STATE.md:54` (the planned probe). The repo contract is "first signed integer anywhere in the text" (`grpo_env.py:43` `_INT_RE`, `client.py:219-222` `parse_order`). `WAVE6_SPEC.md` §3.1 recorded the prompt choice explicitly: "Paper prompts (App. A.2) are reference, not adopted". |

**Two further facts discovered while verifying (not in the orchestrator's list; both change how A0 must be read):**

* **F-A. A0's "parsed" 15 % is mostly incidental digits.** Logged samples are continuations like `" Choose from 3 options: 0"`
  (order **3**), `" if you order 0"` (0), `" So the Japanese answer is just one integer"` (no int). `extract_int` takes the
  first integer anywhere, so a parsed A0 row is usually a stray number from rambling text, not a decision. Evidence:
  `train_log.jsonl` `samples` of `grpo_a0_base` (steps 0, 150, 299). Consequence: A0's "parse rate" is an *upper bound* on
  deliberate answers; the plateau should be read as "≈ 0 % deliberate answers".
* **F-B. The rollout budget is 8 new tokens.** `_Policy.rollout(…, max_new: int = 8)` (`grpo_train.py:235`) and
  `collect_step` never overrides it (`:665`). The paper's contract line `{"order_quantity": 12}` is **8 Qwen3 tokens**
  (`{"`, `order`, `_quantity`, `":`, ` `, `1`, `2`, `}` — measured with the pinned tokenizer, 2026-10-07; a 3-digit order
  is 9, a 4-digit 10) **plus** the end-of-turn token. An 8-token cap cannot hold the contract for orders ≥ 100 and leaves
  no room for EOS or trailing text. Raising the cap is required by any contract arm, so it is confounded with the
  template/contract change unless controlled — that is why §4 adds control arm **A0b**.

---

## 2. Pre-registered claims and what falsifies them

Validity gate first (V-gates), then the substantive claims. Thresholds are fixed here, before any run.

**V-gates (a probe that fails any V-gate is void, not "negative"):**

* **V0 — reproduction.** A0 re-run (new code, legacy flags) reproduces the plateau: mean parse-fail over steps 0–49 ∈
  **[0.81, 0.87]** (historical 0.8401 ± 0.013 sd; the band is ±~2.3 sd). Outside → harness regression (any of: loop RNG,
  prompt bytes, `max_new`, tokenization) — stop, diagnose, do not read A1/A2.
* **V1 — prompt bytes.** A0's rendered prompt bytes are identical to the pre-change code path (golden SHA of the first
  prompt of step 0, seed 12345, recorded from the *old* code before the change); A1/A2 prompts equal
  `tokenizer.apply_chat_template(…, enable_thinking=False)` output (§4.2).
* **V2 — scoring consistency.** `clip_frac` and `max_abs_logratio` in A1/A2 logs stay at A0's bf16 noise floor (A0 a0_base:
  `clip_frac` ≈ 0.003, step-0 logs); a 10× excess means rollout/score tokenization diverged (the F2/F8 failure class).
* **V3 — parser.** Golden-string unit tests of `parse_contract` (§4.3) pass on the hub.

**Claims:**

* **H1 (artifact hypothesis).** The wave-6 plateau is substantially an interface artifact. *Supports:* A1 contract-parse
  rate over steps 0–4 ≥ **0.50** (≥ 3× A0's 0.16 `has_int` rate, itself an upper bound — F-A). *Falsified:* A1 steps 0–4
  < **0.20** (the contract does not help even before any training).
* **H2 (contract is sufficient).** With the contract, the parse-blind reward is no longer the binding problem. *Supports:*
  A1 P50 ≥ **0.90** (both seeds). *Falsified:* A1 P100 < 0.50 while A2 P100 ≥ 0.90 (the reward *was* needed).
* **H3 (parse-aware reward helps when the contract is not enough).** *Supports:* if A1 P50 < 0.90, A2 P50 ≥ A1 P50 + **0.20**
  (both seeds). *Falsified:* A2 P100 ≤ A1 P100 + 0.05 (the signal fails to bootstrap — see §8, "no variance" risk).
* **H4 (0.6B capacity).** *Supports (the "cannot hold a typed contract" reading):* A1 **and** A2 P100 < **0.20** in both seeds
  (parse-fail ≥ 0.80 — the same bar as the wave-6 plateau). *Falsified:* any arm P100 ≥ 0.20.
* **H5 (substance, secondary, not gated).** If parse rises, orders become decisions: parsed-order entropy is non-degenerate
  (> 1 bit by step 50) and `mean_episode_cost` windows do not rise vs A0 on the same demand paths. This is reported, never
  used to pass or fail H1–H4.

P25 / P50 / P100 = mean `contract_parse_rate` over the 5-step windows ending at steps 25 / 50 / 100 (indices 20–24,
45–49, 95–99); per-step values are always also reported. Two seeds must agree on a gate's direction; otherwise the gate
reads "mixed" and §6's `R-D` applies.

---

## 3. The two contracts

### 3.1 What the paper requests (quoted; `context/paper-2605.17036v3.txt`, Appendix A.2)

> **Your Task:** Decide how many cases of beer to order from your upstream this week based on your current situation.
> Start your response with a JSON object **on its own line** in the following exact format:
> `"order_quantity": <number_of_cases>`
> Important:
> - Replace ‘<number_of_cases>‘ with your actual numeric decision.
> - Do not add any text, notes, or punctuation after the JSON.
> - This will be parsed by a program, so the format must be valid and exact.
> Example (your response should end like this):
> `"order_quantity": 5`

Points to carry into the spec, none of them resolved by the paper:

* The staged text has **no braces** around the object (`{}` is absent in the extraction; "JSON object" requires them). Treat the
  brace-less rendering as an extraction artifact: **verify against the original PDF page before freezing the constant**
  (open arXiv 2605.17036v3, Appendix A.2). Until verified, the spec uses the braced form, which is the only valid JSON.
* The paper is internally inconsistent: "**Start** your response with a JSON object" vs "your response **should end** like
  this". The parse rule in §4.3 therefore keys on the FIRST line and measures trailing text separately.
* The paper does not say whether Qwen-3 4B ran in thinking mode, nor give G, lr, β, steps, max tokens, LoRA-vs-full (also
  stated in `WAVE6_SPEC.md` §2). Our `enable_thinking=False` is a disclosed choice (a thinking trace cannot fit a
  24-token budget, and "start your response with the JSON" contradicts it).
* The paper's prompt states costs 0.50/1.00 and lead times 1/2; our simulator (`engine.SimConfig`) is 1/2 and lead 2 and the
  repo prompt says so. A1 deliberately keeps the repo's state text and swaps **only the answer request** — otherwise the
  probe would change the game description too. This pre-existing difference is outside this probe.

### 3.2 What wave 6 requested (A0, as-is)

`agent_bullwhip/agents.py:377`: the prompt ends `"Reply with ONLY an integer: the number of units to order this week (0 or
more)."`, rendered raw (no chat template, no EOS), continuation sampled at T = 1.0, `top_k=0, top_p=1.0`, **8 new tokens**,
`parse = extract_int` (first signed integer anywhere).

---

## 4. Arms (single-variable discipline)

Everything not listed in a row is identical to `a0_base` (§5). Arms differ from each other only in the rows named.

| arm | what changes vs A0 | purpose |
|---|---|---|
| **A0** | nothing (re-run, 50 steps) | V0 reproduction; expected to reproduce the ≈ 0.84 plateau |
| **A0b** | `--max-new 24` only (raw continuation, legacy prompt, legacy parse) | separates "more tokens" from "template/contract" in A1 (a 24-token budget alone raises `has_int`; A1's gain must be net of this). 25 steps, seed 12345 |
| **A1** | `--prompt-format chat_contract --max-new 24` (chat template, thinking off, paper contract, contract parser) | paper-parity answer contract; reward untouched |
| **A2** | A1 **+** `--parse-aware-reward 1 --fmt-weight 1.0` | A1 + parse-aware reward (the only additional change) |

### 4.1 A0 — as-is

No code path changes. After the implementation (§7) the default flags (`--prompt-format raw`, `--max-new 8`,
`--parse-aware-reward 0`) must reproduce the old behaviour byte-for-byte (V1). Historical A0 trajectories
(`hpc-umbrella/umbrella-results/results/grpo_a0_base/` seed 12345, `grpo_a11_seed2/` seed 2; both 300 steps) are the
full-length references; the A0 re-run is a *regression check*, 50 steps only.

### 4.2 A1 — paper-parity contract

**Prompt = chat template over ONE user message**, exactly how waves 1–5 sent a prompt (`agents.py:108`
`messages=[{"role":"user","content":prompt}]`), rendered with thinking disabled:

```
<|im_start|>user\n{USER}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

(verified against `tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)`
on the pinned snapshot; hub-measured prompt length 226 tokens vs 214 raw for one wholesaler state). Implement the render as
a **pure-string function** (`render_chat_nothink`) with a unit test asserting equality against `apply_chat_template` when
transformers is importable (skipped on the torch-less hub) — wave 7 did the same with a prompt-hash fixture
(`decision_readout.prompt_hash`); the hub has no torch, so the string form is what the hub can test. The golden SHA of the
first A1 prompt (seed 12345, step 0) is recorded at implementation time.

`{USER}` = `build_prompt(role, ctx, "default")` with its final sentence replaced:

```python
LEGACY_REPLY = "Reply with ONLY an integer: the number of units to order this week (0 or more)."   # agents.py:377
assert p.endswith(LEGACY_REPLY)
USER = p[: -len(LEGACY_REPLY)] + CONTRACT_TAIL
```

`CONTRACT_TAIL` (exact bytes; the paper's A.2 task block with three disclosed edits — braces added, "cases of beer" →
"units" to match the repo's state text, typographic quotes → ASCII):

```
—————————
Your Task:
Decide how many units to order from your supplier this week based on your current situation.

Start your response with a JSON object **on its own line** in the following exact format:
{"order_quantity": <number_of_units>}

Important:
- Replace '<number_of_units>' with your actual numeric decision.
- Do not add any text, notes, or punctuation after the JSON.
- This will be parsed by a program, so the format must be valid and exact.

Example (your response should end like this):
{"order_quantity": 5}
```

(The first line is the paper's separator `—————————`, U+2014 ×9; everything else of the state text — role line, state block,
cost sentence, "Shipments … 2 weeks" — is unchanged.)

### 4.3 The parse rule (A1/A2; also logged for A0/A0b)

```python
CONTRACT_RE = re.compile(r'^\{\s*"order_quantity"\s*:\s*(\d{1,4})\s*\}$')
def parse_contract(text: str):
    t = (text or "").strip()
    first = t.split("\n", 1)[0].strip()
    m = CONTRACT_RE.match(first)
    if not m:
        return ParseResult(value=None, exact=False)
    return ParseResult(value=int(m.group(1)), exact=(t == first))
```

* **contract-parsed** (the primary metric, `contract_parse_rate`): the first non-empty line *is* the JSON object with a
  non-negative integer of ≤ 4 digits. `value` is the order. **exact**: nothing follows it (the paper's "no text after the
  JSON"); reported as `contract_exact_rate`, not required for the order to count. Negative, decimal, or ≥ 10⁴ values, code
  fences, or JSON not on line 1 are **unparsed**. The ≤ 4-digit cap is our choice (wave-6's 10⁹ digit-runs were a
  digit-run artifact); it is a deviation from "any number" and is why a rogue order cannot appear in A1/A2.
* Also logged per step for ALL arms (so arms are comparable): `parse_lenient_rate` (= 1 − legacy `parse_failure_rate`, i.e.
  `has_int`), `loose_json_rate` (`"order_quantity"\s*:\s*\d+` anywhere), `contract_parse_rate`, `contract_exact_rate`.
* For A0/A0b the *game* still uses `extract_int`; for A1/A2 the game uses `parse_contract(...).value` and **0 when unparsed**
  (same order-0 convention as wave 6; changing it would add a second variable — see §4.4).

### 4.4 A2 — parse-aware reward (the one extra change; a stated deviation from the paper)

The paper's recipe (§5–§6) is: system/agent-level cost reward, return-to-go, group-normalised advantage, KL-regularised,
no format term (no mention of unparsed outputs at all). A2 adds a format term. Precisely, per training step, for every
decision row *i* = (episode *e*, agent *k*, week *t*) with parsed indicator `f_i = 1[contract-parsed]`:

1. **Cost advantage** `Aᶜ_i` — computed exactly as today (`group_normalize`, `grpo_env.py:325`; eps 1e-4), unchanged.
2. **Mask:** for unparsed rows `Aᶜ_i := 0`. Rationale: an unparsed row's order-0 is not a decision of the policy, so crediting
   it with the episode's cost advantage assigns reward/blame to prose tokens (the finding-2 defect).
3. **Format advantage** `Aᶠ_i` — per (agent, week) cell across the G=8 episodes of the step, the same normalisation as
   `group_normalize`: `Aᶠ_i = (f_i − mean_cell f) / (std_cell f + 1e-4)` (population std). Cells that are all-parsed or
   all-unparsed give 0 (no format signal there).
4. **Combined row advantage** (what `grpo_step` receives): `A_i = f_i · Aᶜ_i + w · Aᶠ_i`, `w = --fmt-weight = 1.0`
   (pre-registered; both terms are z-scores, so w = 1 weights them equally; no tuning inside this probe).
5. **Does an order still enter the simulation?** Yes — unparsed ⇒ order **0**, as in A0/A1. The world the other three agents
   see is therefore unchanged by A2; only the gradient changes. (Alternative "unparsed ⇒ mirror last order" is the wave 1–5
   harness fallback (`LLMAgentConfig.fallback="mirror"`); it changes the dynamics, so it is a named follow-up knob K5 (§6), not
   part of A2.)
6. **Token-level credit:** unchanged `grpo_step` semantics (`grpo_train.py:456-503`): per-row *mean* token log-prob × A_i
   (the paper's own "mean token-level log-probability weighted by the advantage", §5 Model update), KL to the frozen base per
   token, PPO-clip 0.2. Both advantage terms therefore spread over *every* completion token of the row (≤ 24, incl. the EOS
   token, which is scored because only `pad_token_id` is stripped — `grpo_train.py:274`). **Credit restricted to the value
   tokens (digits) is NOT in A2** — it is a larger departure from the paper and is knob K4.
7. **Where it lives:** the combination is computed in `collect_step` (`grpo_train.py:629-694`) before `grpo_step`, so
   `grpo_step` and the KL/clip/optimizer code are untouched. `StepRollout.advantages` stays one float per row; add
   `StepRollout.parsed: list[bool]`.

**Stated deviations from the paper in A1/A2 (all disclosed in the results doc):** (i) thinking disabled; (ii) ≤ 4-digit
integer values only; (iii) braces added/“cases”→“units” in the contract text; (iv) A2 only: parse mask + format advantage;
(v) unchanged pre-existing deviations of the wave-6 lane (system-level reward scope where the paper reports agent-level
rollout reward as strongest — paper §6; LoRA r=16 on q/v; KL β=0.02; shared demand across the G episodes of a step).

---

## 5. Fixed knobs (identical across arms; = `a0_base`)

Source: `hpc-umbrella/campaign.tsv` row `a0_base`, `agent_bullwhip/grpo_train.py:702-724,175-232,389-393`,
`hpc-umbrella/config.sh` (`ROLLOUT_BATCH=16`, `SAVE_EVERY=10`).

| knob | value | where |
|---|---|---|
| model / snapshot | `Qwen/Qwen3-0.6B`, `c1899de289a04d12100db370d81485cdf75e47ca` (`MODEL_REVISION`) | `config.sh` |
| LoRA | r=16, α=32, dropout 0.05, targets `q_proj`,`v_proj` (2,293,760 trainable) | `grpo_train.py:211-212` |
| G / horizon / curricula | 8 / 36 / `poisson,truncnorm` alternating by step | `campaign.tsv` |
| lr / β_KL / clip / max-grad-norm | 1e-5 / 0.02 / 0.2 / 0.5 (AdamW) | `grpo_train.py:389-393,709-710` |
| rollout | T = 1.0, `top_k=0`, `top_p=1.0`, `min_new_tokens=1`, `--rollout-batch 16` | `grpo_train.py:262-267` |
| bf16 on CUDA, `TORCHDYNAMO_DISABLE=1` | as wave 6 | `grpo_train.py:199-210` |
| shared demand per step (F17), reward = system-level return-to-go | as wave 6 | `grpo_train.py:657-662` |
| seeds | **12345** (the historical default) and **2** (= `a11_seed2`, `--seed 2`) | `grpo_train.py:727-729` |
| steps | A0 50 · A0b 25 · A1 100 · A2 100 | §2 |
| save-every | 10 (Slurm-kill cost ≤ 10 steps) | `config.sh` |

**Why seeds 12345 and 2:** with the same `--seed` the loop RNG (`rng.randint` per step, `grpo_train.py:896`) is independent of
the policy, so **every arm sees the identical sequence of demand paths**, which is also identical to the historical
`a0_base` (12345) and `a11_seed2` (2) logs. That gives paired per-step cost comparisons for free and lets those 300-step logs
serve as the A0 reference. (Verify at run time: log a `demand_sha` per step — new field — and recompute it independently
from `random.Random(seed).randint(0, 2**31)` → `sample_curriculum_demand`.)

**Per-arm flag diff** (new flags in §7; legacy defaults reproduce wave 6):

| flag | A0 | A0b | A1 | A2 |
|---|---|---|---|---|
| `--prompt-format` | `raw` | `raw` | `chat_contract` | `chat_contract` |
| `--max-new` | 8 | 24 | 24 | 24 |
| `--parse-aware-reward` | 0 | 0 | 0 | 1 |
| `--fmt-weight` | – | – | – | 1.0 |
| `--steps` | 50 | 25 | 100 | 100 |
| `--dump-completions-every` | 10 | 10 | 10 | 10 |

---

## 6. Metrics, gates, decision rules

### 6.1 Per-step metrics (all in `train_log.jsonl`; new fields marked +)

Primary: `contract_parse_rate` (+; A0/A0b log it too, ≈ 0), reported at P25 / P50 / P100 against the a0_base plateau
(`parse_failure_rate` 0.8401 first-50 → **lenient parse 0.160**, itself an upper bound on deliberate answers, F-A).

Secondary:
* `parse_lenient_rate` (+), `loose_json_rate` (+), `contract_exact_rate` (+) — rates over the step's 1,152 rows.
* Cost trajectory: `mean_episode_cost` (existing) — windowed means over steps 41–50 and 91–100, compared **paired by step**
  with the historical a0_base (seed 12345) / a11_seed2 (seed 2) on the identical demand paths; also the share of steps where
  the arm is cheaper. Per-step cost is noisy and curriculum-alternating (poisson/truncnorm); never read a single step.
* Emitted-quantity entropy: `order_entropy_bits` (+) = Shannon entropy of the empirical distribution of the step's *parsed*
  order values; with `order_nunique`, `order_mean`, `order_std`, `order_max` (existing). A1/A2 orders are ≤ 9,999 by the
  parser; orders > 100 are counted as `n_orders_gt100` (+) (the paper reports post-training max order < 100).
* Fraction of gradient-bearing decision tokens, two numbers (+): `cost_rows_frac` = rows whose cost advantage enters the loss
  (non-zero after masking) / 1,152 — equals ~1.0 in A0/A1 (including junk rows!) and = parse rate in A2;
  `value_token_frac` = digit-only completion tokens of parsed rows / all completion tokens of the step — how much of the
  gradient mass sits on the quantity itself. A0: ≈ 0.01; A1/A2: ≈ 0.1–0.25 once parsing works. This is the direct measure of
  "does the training signal reach the decision".
* A2 only: `fmt_adv_share` (+) = Σ|w·Aᶠ| / Σ|A| over rows (how much of the update is format vs cost).
* Existing health metrics kept: `entropy_first`, `grad_norm`, `loss_kl`, `clip_frac`, `max_abs_logratio`, `n_nonfinite`
  (F1 tripwire aborts a run with exit 3; never resume it).

### 6.2 Gates and decision rules (read after all six runs finish; both seeds must agree on direction else "mixed")

Validity first: V0–V3 (§2). Then, with P50 = A1/A2 window means:

| rule | condition (both seeds) | reading | action |
|---|---|---|---|
| **R-A** contract was the blocker | A1 P50 ≥ 0.90 | the 0.85 plateau was an interface artifact; the reward's parse-blindness is *not* binding once the contract holds | write the **full-length fidelity run** (§9, F1): A1 flags, `a0_base` config, 300 steps × 2 seeds, + the §9 eval harness change. Wave-6 verdict reclassified "interface artifact, not a GRPO result". A2 not needed for parse; kept only if its cost trajectory beats A1 (§6.1) |
| **R-B** reward was also needed | A1 P50 < 0.90 and A2 P50 ≥ 0.90 (and A2 − A1 ≥ 0.20) | contract alone insufficient; the parse-aware reward is what makes the signal reach the decision | full-length run uses A2's reward; deviation (iv) is declared in every downstream table |
| **R-C** 0.6B cannot hold a typed contract | A1 P100 < 0.20 **and** A2 P100 < 0.20 (i.e. parse-fail ≥ 0.80 in all arms) | capacity, not interface: the typed contract does not install at 0.6B in 100 steps | **reassign the fidelity claim to the size ladder**: a 50-step A1 micro-probe at `Qwen/Qwen3-1.7B` then `Qwen/Qwen3-4B` (the paper's reference, §5/§6 "Qwen-3 4B"), same flags; wave-6's 0.6B null stands as "non-informative about GRPO" (not "GRPO fails") |
| **R-D** middle | anything else (e.g. A1 P100 ∈ [0.20, 0.90) or seeds disagree) | partial | next knob, one at a time, 50 steps each, seed 12345, in this order: **K1** template-only (A1 prompt with the *legacy* "Reply with ONLY an integer" line + `chat_template` + lenient parser — splits "template" from "contract"); **K2** rollout temperature 0.7 (the eval harness value; training used 1.0); **K3** assistant-turn prefill of `{"order_quantity": ` (the model emits only digits + `}`; credit on digits automatically; the strongest departure from the paper, so last); **K4** value-token-only cost credit; **K5** unparsed ⇒ mirror-last-order in the simulation. Stop at the first knob that gets P50 ≥ 0.90 |

**Early stop (saves ≈ half the compute):** after step 50, an A1/A2 run whose P50 < **0.02** is stopped (a rate that low at
step 50 cannot reach 0.20 by step 100 under a flat trend; the P100 gate is then read as the P50 value and flagged
"stopped early"). A `n_nonfinite > 0` abort is a run failure, not a data point: rerun that run from scratch with a new
`--out` (never `--resume`).

**Step-0 reading (H1):** the first-5-step `contract_parse_rate` of A1 is the *untrained* compliance — if it is already ≥ 0.90,
training is irrelevant for format and R-A holds immediately; the arm still runs to step 100 (cost trajectory, collapse
checks).

---

## 7. Implementation change list (for the later implementer; nothing done here)

All edits keep legacy defaults ⇒ wave-6 reproducibility (V1). Code paths are stable; re-read before editing.

1. **`agent_bullwhip/contract.py` (new, torch-free):** `LEGACY_REPLY`, `CONTRACT_TAIL`, `render_chat_nothink(user: str) -> str`,
   `build_contract_prompt(role, ctx) -> str`, `CONTRACT_RE`, `ParseResult`, `parse_contract(text)`, `count_digit_tokens(...)`.
   No equivalent exists (`grep order_quantity` → none).
2. **`agent_bullwhip/grpo_env.py`:** `run_episodes_batched(..., prompt_builder: Callable[[str, dict], str] | None = None,
   parse=…)` — today `build_prompt(role, ctx, prompt_variant)` is hard-wired at `:284`; add the hook, default = old call.
   The existing `parse=` hook (`:233`) already lets A1/A2 pass `lambda t: parse_contract(t).value or 0`.
3. **`agent_bullwhip/grpo_train.py`:**
   * `parse_args` (`:702-724`): `--prompt-format {raw,chat_contract}` (default raw), `--max-new INT` (default 8),
     `--parse-aware-reward {0,1}` (default 0), `--fmt-weight FLOAT` (default 1.0), `--dump-completions-every N` (default 0).
   * `_Policy.rollout` gets `max_new` from `collect_step` (`:665` currently omits it); `min_new_tokens=1` unchanged.
   * `collect_step` (`:629-694`): thread `prompt_builder`/`parse`; compute `parsed[]`; if `parse_aware_reward`, replace
     `flat_adv[i]` by `f_i·Aᶜ_i + w·Aᶠ_i` (§4.4) using one extra pass of `group_normalize` on the `f` tensor shaped as
     `{agent:[per-week]}`; put `parsed` on `StepRollout`.
   * `run_loop` (`:908-937`): new log fields (§6.1) + `demand_sha`; keep the legacy `parse_failure_rate` (=1−lenient) untouched
     so old and new logs compare. Every `--dump-completions-every` steps write `completions_step<k>.jsonl` (all 1,152 rows:
     `{i, agent_id, t, text, ids, order, parsed}`) next to `train_log.jsonl`.
   * `state.json` / `run_info.json`: record the four new flags and `contract_version="w6fix-1"`.
4. **Tests (torch-free):** `tests/test_w6fix_contract.py` — parser goldens (`{"order_quantity": 12}`→12 exact;
   `{"order_quantity": 12}\nThanks`→12 not exact; `  {"order_quantity":7}  `→7; `{"order_quantity": -1}`, `…: 1.5`, `…: 12345`,
   ```` ```json\n{…}\n``` ````, `Sure! {"order_quantity": 5}`, `""`→unparsed); render golden vs `apply_chat_template`
   (skip without transformers); A2 advantage math on a hand-built 3-episode/2-agent/2-week grid (mask + format z-score +
   all-parsed cell = 0 + w); stub-policy run of `run_loop` at three compliance rates {0.0, 0.5, 1.0} asserting the logged
   `contract_parse_rate`, `cost_rows_frac`, `fmt_adv_share`; legacy-defaults test asserting `collect_step` output is
   bit-identical to the pre-change function on the existing `_StubPolicy` (tests/test_grpo_train.py:61).
5. **`scripts/w6fix_analysis.py` (new, stdlib only, modelled on `scripts/wave7_analysis.py`):** reads
   `results/grpo_w6fix_*/train_log.jsonl` + the historical `hpc-umbrella/umbrella-results/results/grpo_a0_base|a11_seed2/`,
   writes `results/w6fix_report.md` + `.json`: V-gates, P25/P50/P100 table per arm/seed, §6.2 rule evaluation, paired cost
   windows, entropy, token-fraction tables.
6. **Cluster:** `hpc-umbrella/campaign-w6fix.tsv` (columns `name group horizon curricula lr beta_kl steps prompt_format max_new
   parse_aware fmt_weight`) and `hpc-umbrella/gpu-w6fix.sbatch` cloned from `gpu-campaign.sbatch` (same `SLURM_SUBMIT_DIR`
   block, TERM forwarding, `EXPECTED_COMMIT` gate, `--out results/grpo_${NAME}` naming, `_seed<N>` suffix → `--seed N`).
   `tests/test_sbatch_contract.py` / `config.sh --check` enforce header pins — the new sbatch must repeat the pinned headers.
   `hpc-umbrella/verify_pull.py` EXPECT table gets the six new dirs (+ their step counts).

---

## 8. Artifacts, verification, compute

### 8.1 Artifacts (repo conventions: text evidence tracked, weights untracked — `docs/ARTIFACTS.md`)

| path | content |
|---|---|
| `results/grpo_w6fix_a0/`, `grpo_w6fix_a0b/`, `grpo_w6fix_a1/`, `grpo_w6fix_a1_seed2/`, `grpo_w6fix_a2/`, `grpo_w6fix_a2_seed2/` | `train_log.jsonl`, `state.json`, `run_info.json`, `completions_step{0,10,…}.jsonl`; `checkpoint-*/` (untracked, per `.gitignore`) |
| `hpc-umbrella/campaign-w6fix.tsv`, `hpc-umbrella/gpu-w6fix.sbatch` | run definition |
| `results/w6fix_report.md`, `results/w6fix_report.json` | generated gate readout (`scripts/w6fix_analysis.py`) |
| `results/w6fix_logs/w6fix-<jobid>_<task>.log` | Slurm logs (provenance) |
| `docs/waves/WAVE6-FIX.md` (written after the runs) | results doc in repo voice; this spec stays the pre-registration |
| `docs/verify/VERIFY-W6FIX.md` | independent recomputation (below) |

Cluster names: a0 `w6fix_a0` (seed default 12345), `w6fix_a0b`, `w6fix_a1`, `w6fix_a1_seed2`, `w6fix_a2`, `w6fix_a2_seed2`
(the `_seed<N>` suffix convention of `gpu-campaign.sbatch` sets `--seed`). Results travel by git
(`bash hpc-umbrella/push_results.sh -m "w6fix runs (job <id>)" results/grpo_w6fix_* results/w6fix_logs`), hub `git pull --ff-only`,
then `python3 hpc-umbrella/verify_pull.py ~/projects/agent-bullwhip-replication --size` (OPERATOR.md Step 6).

### 8.2 Verification (numbers to recompute independently; separate code, no import of `w6fix_analysis.py`)

1. **Parse rates from raw completions.** From each `completions_step<k>.jsonl`, re-parse with a fresh, independently written
   regex and compare to the logged `contract_parse_rate`, `contract_exact_rate`, `parse_lenient_rate`, `loose_json_rate` at
   that step — **exact equality** (counts of 1,152 rows).
2. **Plateau reference.** Recompute the A0 windows quoted in §1/§2 (0.8401 steps 0–49; 0.8491 steps 45–49 …) from
   `grpo_a0_base/train_log.jsonl` and the re-run.
3. **A2 advantage.** For the step-0 and step-50 dumps rebuild per-cell `Aᶠ`, the mask and `A_i` from the dump alone; compare
   the logged `cost_rows_frac`, `fmt_adv_share`, `mean_adv`, `min_adv`, `max_adv` (tolerance 1e-6).
4. **Demand pairing.** Recompute `demand_sha` for steps 0–99 for seeds 12345 and 2 from `random.Random(seed)`; must equal
   every arm's logged value and the historical runs' implied paths.
5. **Prompt bytes.** Render the step-0 first prompt of each arm with `apply_chat_template` (cluster venv) and compare SHA to
   the golden constants; compare A0's to the pre-change SHA (V1).
6. **Wave-6 statements** the results doc will repeat (parse 0.847; "eval parse-fail 0.34–0.40"; 188/58 rogue orders) are
   re-cited from `eval-a0-report.md`, not recomputed here.
7. **Cost windows**: recompute the paired `mean_episode_cost` window means/medians and the cheaper-step share from the logs.

### 8.3 Venue and compute

**Hub (CPU, no torch):** implement + `uv run python -m pytest tests/test_w6fix_contract.py tests/test_grpo_train.py
tests/test_grpo_env.py tests/test_sbatch_contract.py -q` (baseline 326 passed / 15 skipped, README 2026-10-07); then the
**stub dry-run** — `run_loop` with a stub policy at compliance {0, .5, 1} for 10 steps writing to a temp `--out` (exercises
logging, dumps, A2 maths, resume) — and `scripts/w6fix_analysis.py` on that output.

**Cluster (TU/e Umbrella L4s):** training stayed on the cluster in wave 6 (`WAVE6_SPEC.md` D6; Modal OUT per D3 and used
only for the CPU llama.cpp eval, `eval-a0-report.md`). Mirror it: `hpc-umbrella/OPERATOR.md` Steps 0–2 (stage; `bash
config.sh --check`; set `EXPECTED_COMMIT`), Step 3 CPU harness, Step 4 pilot sanity (4 steps of A1), then
`cd /scratch-shared/$USER/bullwhip/kit && sbatch --array=0-5 gpu-w6fix.sbatch` (`tue.gpu.q -C l4`, 1 GPU, 8 CPU, 16 G,
walltime 14 h with `--signal=B:TERM@600`, `--resume` on resubmit). Six independent array tasks.

**GPU-hour estimate** (measured basis: a0_base averaged 30,317 s / 300 steps = **101 s/step** on an L4 — collect ≈ 42 s,
train ≈ 60 s; `bench_360642.json`, rollout-batch 16):

| run | steps | s/step | GPU-h |
|---|---:|---:|---:|
| A0 | 50 | 101 | 1.4 |
| A0b | 25 | 110–190 | 0.8–1.3 |
| A1 seed 12345 / seed 2 | 100 | 110–190 | 3.1–5.3 each |
| A2 seed 12345 / seed 2 | 100 | 110–190 | 3.1–5.3 each |
| **total** | | | **≈ 14–24 GPU-h; wall ≈ 3–6 h (6-way parallel); upper bound 24 h** |

The 110–190 s/step range is `[INFERENCE]`: the 24-token cap triples the decode phase at worst (batched `generate` runs until
the slowest row stops) and the training forward grows ~10 %; measure on the 4-step pilot and update the table. Early-stop
(§6.2) can remove up to ~half of an arm. Free (institutional) compute: **$0**.

---

## 9. Follow-ups this probe gates (not part of this spec's runs)

* **F1 (if R-A or R-B): full-length fidelity run.** `a0_base` config (G=8, 300 steps, lr 1e-5, β 0.02) with the winning
  flags, 2 seeds (~10–16 GPU-h each on the measured 101–190 s/step). Its **eval must use the same contract**: the llama.cpp
  `/completion` raw path (`client.py:157-164`, `BULLWHIP_COMPLETIONS_API=1`) cannot serve a templated prompt — add a
  `BULLWHIP_CHAT_TEMPLATE=qwen3_nothink` mode that wraps the prompt with `render_chat_nothink`, uses `n_predict=24`, and parses
  with `parse_contract` (unparsed ⇒ the harness fallback, labelled — AUDIT2/3 anti-pattern is an *unlabeled* fallback), then
  re-use `hpc-umbrella/eval-analyze.py` + `scripts/wave7_floors.py` floors on the standard protocol (step n=30, noisy n=10 →
  20). Only then may wave-6 numbers be called a test of the paper's GRPO claim.
* **F2 (if R-C): size ladder** — same A1/A2 flags at `Qwen/Qwen3-1.7B`, then `Qwen/Qwen3-4B` (paper's reference model;
  HF id/revision to be pinned at staging — unverified here). 4B LoRA training on a 22 GiB L4 likely needs micro-batching
  `[INFERENCE]`.

---

## 10. Risks, stop-loss, honest-negative clause

* **Risk: A1's step-0 compliance is high but meaningless** (a 0.6B model can emit the template line while ignoring the state).
  Mitigation: H5/§6.1 entropy and `n_orders_gt100`; and the §9 F1 eval is where substance is judged — this probe claims *only*
  about parse/credit, never about decision quality.
* **Risk: A2 cannot bootstrap.** If A1's compliance is ≈ 0, cells are all-unparsed → `Aᶠ = 0` everywhere and A2 ≡ "A1 with the
  cost advantage masked" = no gradient at all. Detectable at step 0 (`fmt_adv_share` ≈ 0). That outcome is itself the
  informative result for H3 (reward shaping cannot create a behaviour that never occurs at G=8) and routes to R-C / K3.
* **Risk: max-token confound.** Addressed by A0b. If A0b's `parse_lenient_rate` is ≥ 0.5 the A1 gain must be read net of it.
* **Risk: contract text ambiguity** (braces, "start" vs "end"). Addressed by §3.1 (verify against the PDF) and by logging
  `loose_json_rate`/`contract_exact_rate` so the choice cannot silently decide the verdict.
* **Risk: 100 steps is short.** wave-6 showed no movement in 300 steps for the legacy format, so a *contract* effect, if any,
  should be visible in tens of steps (H1 is a step-0..4 claim). The early-stop and R-D knobs cover "slow but moving".
* **Stop-loss:** total ≤ 24 GPU-h (free). Any arm with `n_nonfinite > 0` is rerun once from scratch; a second abort stops that arm
  and is reported. Any V-gate failure stops the whole probe until diagnosed (no reading of A1/A2).
* **Honest-negative clause.** Every outcome is publishable as written: R-C ("0.6B does not hold the contract; wave-6's null is
  non-informative about GRPO and the fidelity claim moves to the size ladder"), R-D ("parse rose but not to the gate; knob K*
  decided it") and R-A ("the plateau was an interface artifact — wave-6's *null* was not a GRPO result") are all results; none is
  re-run with different thresholds. If the gate numbers move after seeing data, that is logged below with the reason, never
  silently.

---

## 11. Discrepancies found in the repo while writing this spec (reported, not fixed)

1. `WAVE7.md` (§1/§6) cites wave-6 "34–40 % parse failures"; `STATE.md` cites 84.5–86.7 %. Both are right and measure
   different things (eval harness fallback counter vs training-time `has_int`) — `eval-a0-report.md` says so. Any doc quoting
   one must name the counter.
2. `reward-unparsed-findings.md` analyses 5 arms ("a0_base + the four wave-6 variants"); `campaign.tsv` has 12 completed 300-step
   arms (a0–a11). The parse-fail claim holds for all 12 (0.843–0.867).
3. The brief/`WAVE7_SPEC.md` describe the community RLCD LoRA as "~5M trainable"; the wave-6 and wave-7 LoRAs on Qwen3-0.6B are
   both **2,293,760** trainable parameters (computed from `config.json`: 28 layers, hidden 1024, 16 q-heads, 8 kv-heads,
   head_dim 128). The 5M figure is not ours.
4. `docs/waves/WAVE7.md` §7 option 1 says permuted-option augmentation was "pre-registered … but not implemented"; the code has it on by
   default (`decision_train.py:372-373`, `permute_options: 1` in `results/wave7_sft_seed0/state.json`). Relevant to WAVE8_SPEC.md.

## Changes log

- 2026-10-07: draft written (spec-only; waiting for repo cleanup to finish and Pino's approval).
- 2026-10-07 (implementation, mission `bullwhip-impl`): code + tests + kit implemented; **nothing was run on a GPU or the cluster**. Decisions
  and deviations (smallest deviation that preserves the claim; no arm, gate or number changed):
  1. **Braces / PDF check.** Contract constant kept in the braced form (the only valid JSON, as §3.1 prescribes). The paper PDF
     (`curl https://arxiv.org/pdf/2605.17036v3` then `pdftotext -raw`, 2026-10-07, App. A.2) also renders the example WITHOUT braces
     (`"order_quantity": <number_of_cases>` / `"order_quantity": 5`), i.e. the staged text extraction was faithful to the PDF text layer; whether the
     brace-less form is a typesetting artifact of the paper cannot be decided from the text layer [unverified — would need the LaTeX source or a
     page image]. Consequence: deviation (iii) stays as written ("braces added"); `loose_json_rate` (brace-agnostic) is the telemetry that keeps the
     choice from silently deciding the verdict.
  2. **Renderer.** `agent_bullwhip/qwen_chat.py::render_nothink` (pure string) was verified equal to
     `tokenizer.apply_chat_template(..., enable_thinking=False)` on one real `build_prompt` output with the pinned snapshot, via an external
     venv (`/home/pino/absolver/.venv`, transformers 5.14.1, offline; not a repo dependency). Same run measured the contract line
     `{"order_quantity": 12}` = 8 tokens (`{"`, `order`, `_quantity`, `":`, ` `, `1`, `2`, `}`), matching F-B. The in-repo equality test is
     `importorskip`-gated (skipped on the hub).
  3. **Golden SHAs** recorded: A0 first prompt (step 0, seed 12345, poisson) from the PRE-change code
     `a61e56c58958011688445c0d8b9ed368ec5010122110b2d9b810b0bc8ec4b370`; A1/A2 first prompt from the new code
     `97a3b97f378272d98ee4225cb90c89b0e45064c16dda883e6bbd78d707efdb2f` (`tests/fixtures/w6fix_collect_legacy_golden.json`; also pins
     bit-identical legacy `collect_step` output for two stub policies x 2 seeds x 2 curricula).
  4. **Parser.** `CONTRACT_RE` uses ASCII `[0-9]{1,4}` (not `\d`, which accepts Unicode digits); leading zeros (`007`) parse to 7 (pinned by test).
  5. **Flag guards.** `--parse-aware-reward 1` requires `--prompt-format chat_contract` (argparse error otherwise): the reward is defined on the
     contract parse. `--max-new` must be >= 1.
  6. **Telemetry definitions.** `demand_sha` = first 16 hex of sha256 of `",".join(demand)` (shared demand; with `shared_demand=False` the G demands
     joined by `|`). `order_entropy_bits` = Shannon entropy in bits over the orders of rows the ARM'S OWN parser accepted (raw: has_int; contract:
     contract-parsed; 0.0 if none). `cost_rows_frac` = rows with non-zero cost advantage after masking / n. `fmt_adv_share` null outside A2.
     `first_prompt_sha256` (full hex of the step's first prompt) is logged every step. `value_token_frac` needs a tokenizer on the policy (null for stubs).
  7. **Dump superset.** `completions_step<k>.jsonl` rows carry `{i, e, agent_id, t, text, ids, order, parsed, adv_cost, adv_fmt, adv}` (`e`, `adv_*`
     added to §7.3's list so §8.2.3 can be recomputed from the dump alone).
  8. **Seeds / V1 for seed-2 arms.** Golden prompt SHA exists for seed 12345 only; seed-2 arms' step-0 prompt is not gated (analysis prints it).
  9. **Pilot.** `gpu-pilot.sbatch` is a three-stage benchmark script (optimizer probe, rollout bench, GRPO pilot) and resists a minimal switch; instead
     `gpu-w6fix.sbatch` takes `W6FIX_STEPS=4` (`W6FIX_STEPS=4 sbatch --array=2 gpu-w6fix.sbatch`): the first 4 steps of the real A1 run in its own
     `--out`, resumable by the full submit. Slightly different from the "first 4 steps + scancel" fallback; no scancel needed.
  10. **verify_pull.** EXPECT has the six `grpo_w6fix_*` dirs; an A1/A2 arm stopped early (§6.2) reports INCOMPLETE there by design (no override flag
     exists); judge by `results/w6fix_report.md`.
  11. **Config.** `config.sh` gains `CAMPAIGN_W6FIX_TSV`; the array is `--array=0-5`, `--time=14:00:00` (same pins as `gpu-campaign.sbatch`).
