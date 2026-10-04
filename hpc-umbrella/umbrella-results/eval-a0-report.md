# Wave-6 eval — Qwen3-0.6B GRPO (a0_base rung), Modal CPU

**Date:** 2026-10-04 · **Code:** (`git rev-parse HEAD`)
**Serving:** all-in-Modal — per arm, ONE container runs llama.cpp (Qwen3-0.6B `Q4_K_M`,
16 threads, private `127.0.0.1:8081`, nothing exposed publicly) AND the eval runner;
results write to the Modal volume `bw6-models` (`results/<arm>-step/`, `-noisy/`) and
are pulled into `results/local-eval/`. Trained arm loads `a0_base-lora-f16.gguf` (from
`hpc-umbrella/umbrella-results/results/grpo_a0_base/checkpoint-299`) on the same base.
**Serving format:** raw prompt continuation via llama.cpp `/completion`
(`BULLWHIP_COMPLETIONS_API=1`) — the same protocol the policy was TRAINED and SCORED
on (`agent_bullwhip/grpo_train.py::_Policy`); no chat template, temperature 0.7
(harness default) vs 1.0 at training time.
**Trained arm:** the modal container adds `--lora /data/a0_base-lora-f16.gguf`.

## Protocol

Both arms: `--config baseline --pattern step|noisy`, MIT step demand
(4,4,4,4,8…8), horizon 36, 4 tiers → 144 decisions/game. Step arms n=30,
noisy arms n=10 (seeds 1000+i, i in 0..9+). Paired on identical demand paths
(same run index = same path). Floors = order-up-to machinery shipped in the repo
(`walkforward_paths.py`, `walkforward_floor.py`, `tuned_floor.py`, `oos_floor.py`,
`tuned_floor_chaos.py`) run on the same paths — outputs under
`results/local-eval/floors/`.

## Arms

(Sections appended by `hpc-umbrella/eval-analyze.py`.)

- **step:** `results/local-eval/{untrained,trained}-step/` (Modal-computed summaries;
  the untrained volume dir carries the retry1 suffix, content is the primary data).
- **noisy:** `results/local-eval/{untrained,trained}-noisy/`.
- **pre-Modal cross-check:** the earlier hub-served partials were moved aside to
  `results/local-eval/pre-modal-hub/` before the Modal results landed (same runner
  protocol, different server threads/host).

## Caveats

- Raw-continuation serving (no chat template) — matches training/scoring, but is not
  what the harness served in waves 1–5 (chat template, temperature default).
- Temperature 0.7 (harness) vs 1.0 (training-time rollouts).
- Q4_K_M quantization of the base; LoRA applied in f16 by llama.cpp.
- Modal CPU serving (16 threads/container, private); the pre-Modal hub partials used
  a different thread count on the same harness — cross-check, not a duplicate run.
- n = 30 step / 10 noisy; not GPU numbers.
- Parse-failure counting differs between training (regex over raw completions) and
  the harness (`parse_order_prefer_label` + `_fallback` per decision) — rates are
  compared, not assumed equal.
- Rogue parsed orders (single order ≥ 1e5) tracked per arm as a watch item
  (sweep-363909 standing item); they dominate mean/max when present. See the
  rogue-order breakdown per arm below.

## step arm (step)

**Untrained**: `results/local-eval/untrained-step/qwen3-0.6b-untrained-baseline.jsonl` · **Trained**: `results/local-eval/trained-step/qwen3-0.6b-trained-baseline.jsonl` · completed 30/30 runs · failed 0/0

| arm | n | mean | std | CV | min | max | parse-fail |
|---|---|---:|---:|---:|---:|---:|---:|
| untrained | 30 | 235,662,919 | 502,636,808 | 2.133 | 194,455 | 2,036,335,297 | 0.340 |
| trained | 30 | 270,313,836 | 778,759,070 | 2.881 | 9,768 | 3,432,123,404 | 0.374 |

Training-time parse-failure reference: 0.847 (mean over 300 GRPO steps, `train_log.jsonl`).

### Order stats + collapse check

| arm | mean order | std order | max order | per-(agent,week) std (across runs) | cell-std frac < 1 |
|---|---:|---:|---:|---:|---:|
| untrained | 247,421.4 | 3,654,024.5 | 128595968 | 1096161.41 (max 25127881.64) | 0.00 |
| trained | 136,447.6 | 2,270,172.0 | 90312788 | 708161.82 (max 16211099.67) | 0.01 |

Rogue orders (single order ≥ 1e5, sweep-363909 watch item): untrained **188**, trained **58** across all runs and roles.

Rogue-order breakdown — games with ≥1 rogue order vs clean games:

| arm | games w/ rogue | rogue freq | n rogue orders | total_cost w/ rogue (n, mean, CV, max) | total_cost wo/ rogue (n, mean, CV, max) |
|---|---:|---:|---:|---|---|
| untrained | 22 | 0.733 | 188 | n=30 mean=235,662,919 CV=2.133 max=2,036,335,297 | n=8 mean=806,055 CV=0.857 max=2,148,878 |
| trained | 10 | 0.333 | 58 | n=30 mean=270,313,836 CV=2.881 max=3,432,123,404 | n=20 mean=435,362 CV=1.963 max=3,370,907 |

Unique order values per role (untrained → trained):

| role | untrained | trained |
|---|---:|---:|
| retailer | 231 | 91 |
| wholesaler | 277 | 137 |
| distributor | 312 | 161 |
| factory | 355 | 150 |

Correlation vs order-up-to floors (same demand paths):

| floor | untrained r | trained r | n points |
|---|---:|---:|---:|
| untuned (3,681) | -0.043 | -0.057 | 144 |
| wf_tuned (3,206) | -0.021 | -0.071 | 144 |

### Paired deltas (trained − untrained, same run index = same demand path)

| n pairs | mean Δ | median Δ | std Δ | min Δ | max Δ | trained worse | sign-test p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 34,650,917 | -2,982,404 | 986,357,125 | -2,036,322,848 | 3,425,751,359 | 7 | 0.0052 |

Relative deltas: mean **+14.7%**, CV **+35.1%**, max **+68.5%**.


## noisy arm (noisy)

**Untrained**: `results/local-eval/untrained-noisy/qwen3-0.6b-untrained-baseline.jsonl` · **Trained**: `results/local-eval/trained-noisy/qwen3-0.6b-trained-baseline.jsonl` · completed 10/10 runs · failed 0/0

| arm | n | mean | std | CV | min | max | parse-fail |
|---|---|---:|---:|---:|---:|---:|---:|
| untrained | 10 | 1,133,101,720 | 2,230,808,250 | 1.969 | 235,200 | 7,525,805,517 | 0.338 |
| trained | 10 | 266,666,666,640,000,007,827,740,374,804,791,136,951,405,720,478,353,528,455,168 | 799,999,999,920,000,068,084,711,521,475,619,693,925,653,706,731,783,597,326,336 | 3.000 | 12,975 | 2,666,666,666,400,000,167,480,384,542,170,403,935,656,930,295,376,981,308,473,344 | 0.398 |

Training-time parse-failure reference: 0.847 (mean over 300 GRPO steps, `train_log.jsonl`).

### Order stats + collapse check

| arm | mean order | std order | max order | per-(agent,week) std (across runs) | cell-std frac < 1 |
|---|---:|---:|---:|---:|---:|
| untrained | 1,587,841.8 | 28,418,466.7 | 1000000000 | 4680943.77 (max 299965719.96) | 0.00 |
| trained | 46,296,296,291,666,667,299,713,654,527,829,739,221,086,552,839,719,747,584.0 | 1,756,210,808,934,625,088,120,064,747,763,771,908,388,883,457,767,165,132,800.0 | 66666666660000000000000000000000000000000000000000000000000 | 138888888875000012788176705053520048491247096335742009344.00 (max 19999999998000001144599158073624913809748385852085552283648.00) | 0.03 |

Rogue orders (single order ≥ 1e5, sweep-363909 watch item): untrained **59**, trained **7** across all runs and roles.

Rogue-order breakdown — games with ≥1 rogue order vs clean games:

| arm | games w/ rogue | rogue freq | n rogue orders | total_cost w/ rogue (n, mean, CV, max) | total_cost wo/ rogue (n, mean, CV, max) |
|---|---:|---:|---:|---|---|
| untrained | 8 | 0.800 | 59 | n=10 mean=1,133,101,720 CV=1.969 max=7,525,805,517 | n=2 mean=506,456 CV=0.536 max=777,712 |
| trained | 2 | 0.200 | 7 | n=10 mean=266,666,666,640,000,007,827,740,374,804,791,136,951,405,720,478,353,528,455,168 CV=3.000 max=2,666,666,666,400,000,167,480,384,542,170,403,935,656,930,295,376,981,308,473,344 | n=8 mean=176,857 CV=1.901 max=1,062,824 |

Unique order values per role (untrained → trained):

| role | untrained | trained |
|---|---:|---:|
| retailer | 100 | 38 |
| wholesaler | 117 | 43 |
| distributor | 131 | 57 |
| factory | 140 | 61 |

Correlation vs order-up-to floors (same demand paths):

| floor | untrained r | trained r | n points |
|---|---:|---:|---:|
| untuned (5,479) | -0.017 | 0.011 | 1440 |
| tuned (4,838) | -0.031 | 0.020 | 1440 |

### Paired deltas (trained − untrained, same run index = same demand path)

| n pairs | mean Δ | median Δ | std Δ | min Δ | max Δ | trained worse | sign-test p |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 266,666,666,640,000,007,827,740,374,804,791,136,951,405,720,478,353,528,455,168 | -13,024,333 | 799,999,999,920,000,068,084,711,521,475,619,693,925,653,706,731,783,597,326,336 | -7,525,691,243 | 2,666,666,666,400,000,167,480,384,542,170,403,935,656,930,295,376,981,308,473,344 | 3 | 0.3438 |

Relative deltas: mean **+23534221332803437156685937503005137271640344814944256.0%**, CV **+52.4%**, max **+35433637773076673909415124397405595159363309591855104.0%**.

### Rogue-order detail (sweep-363909 watch item)

Rogue orders = single parsed order ≥ 1e5, from rambling raw completions (the known
signature of these arms — not a harness fault). The counts above include extreme
digit-run outputs:

- **step/untrained:** 22/30 games contain ≥1 rogue order (188 rogue orders total);
  worst game total_cost 2.04e9, max order 1.29e8.
- **step/trained:** 10/30 games (58 rogue orders); worst game 3.43e9, max order 9.03e7.
- **noisy/untrained:** 8/10 games (59 rogue orders); worst game 7.53e9 — runs 3 and 8
  emitted the digit strings `1000000000` and `123456789` verbatim.
- **noisy/trained:** 2/10 games (7 rogue orders), but **run 8** emitted an order of
  `6.67e58` (~59-digit run of 6s), driving that game's total_cost to 2.67e60 and the
  arm mean to 2.67e59, CV to 3.0. Same rambling-completion phenomenon as the ~1e9
  rogues, at the tail of the distribution.

With rogue games excluded: step clean means 806,055 (untrained, n=8) vs 435,362
(trained, n=20); noisy clean means 506,456 (untrained, n=2) vs 176,857 (trained, n=8).

### Verdict vs the pre-registered criteria (WAVE6_SPEC §4)

The 0.6B rung does NOT show the paper's mean/CV/max improvement on all games:

| metric (step, all games) | untrained | trained | rel. Δ |
|---|---:|---:|---:|
| mean total_cost | 235.7M | 270.3M | +14.7% |
| CV | 2.13 | 2.88 | +35.1% |
| max | 2.04G | 3.43G | +68.5% |

But the rogue games carry the aggregate verdict. Trained is cheaper in **23/30** paired
step runs (sign-test p=0.0052, median Δ −2.98M) and its rogue-game rate is lower
(0.33 vs 0.73 step; 0.20 vs 0.80 noisy); on the 7 both-clean step pairs trained is
cheaper in 6/7 (median Δ −184,605; noisy both-clean n=2, 1/2). No collapse in either
arm: per-(agent,week) order std ≫ 1 (cell-std frac <1 ≈ 0.00–0.03) and order
correlation vs the order-up-to floors ≈ 0 (−0.07 … +0.02) — neither arm is a
distilled-formula regime; both behave like noisy raw LLMs.

**Reading:** at 0.6B on CPU the trained policy makes the *typical* game cheaper
(lower median, fewer rogue games) but its tail rogues grew (worst orders 9.0e7→1.3e8
step, and a 6.7e58 digit-run on noisy), so mean/CV/max move the wrong way. By the §4
criteria the **aggregate replication claim is NULL at 0.6B** (CV and mean go up), with
a real median/count effect that a rogue-robust metric would capture. Parse-failure
0.34–0.40 vs the 0.85 training-time reference — the eval harness falls back far less
than the training-time regex, as noted in the protocol.

**Pre-Modal cross-check:** the earlier hub-served partials
(`results/local-eval/pre-modal-hub/`, same runner protocol, different server) show the
same qualitative picture — parse-fail 0.356/0.371, rogues in both arms, and clean-game
means 809,353 (untrained, n=7) vs 242,514 (trained, n=9), within the same range as the
Modal step clean means above. The results are not an artifact of the Modal serving
path.
