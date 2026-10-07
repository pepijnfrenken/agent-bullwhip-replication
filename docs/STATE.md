# STATE — where the project stands (2026-10-07)

Authoritative per-wave detail lives in the wave docs (list below, under `waves/`). This file
is the read-first summary: what each wave tested, the headline result as written up, where the
doc + data live, current verdicts, and the open issues that are agreed but not started.

---

## Wave table (1–7)

| wave | what it tested | headline result (as documented) | doc → data |
|---|---|---|---|
| 1 | Naive API-LLM arms (configs `voting5`, `budget`, `prompt_weighted`, `anchor`, `combined`) under the pre-audit sequential protocol | **Tombstoned.** Ran only under the sequential protocol that Audit #1 deleted; not part of the surviving table | README §wave-1 configs; no dedicated wave doc |
| 2 | Interleaved round-robin LLM arms (fix for endpoint window drift); noisy arm | Confs: interleaving fixed the 16× window drift; the "LLM beats formula under noise" claim died in AUDIT2 (same-week leak + untuned baseline); the twin-gap confound (byte-identical configs differ 10–20× with a 0.2% gate) remains open | `audits/AUDIT.md` `audits/AUDIT2.md` → `../results/noisy_leakfree/` |
| 3 | Seeded chaotic/wild arms + walk-forward floor; Audit #3 (unseeded runs) | **The surviving headline:** a self-tuning 1970s order-up-to formula beats the LLM agent in every environment (mild/chaotic/wild) at 3–4 orders of magnitude less compute. Leak-free, seeded, 40/40 runs. See the README surviving tables | `audits/AUDIT3.md` → `../results/noisy_leakfree/` `../results/chaotic_seeded/` `../results/wild_seeded/` |
| 4 | Tool-calling arm (`ToolAgent` + `run_python`), one full game on fixed demand | **PRELIMINARY / TBC.** Game cost 6,948 vs 3,681 floor (+89%); match-to-floor 42.4%. The finding is the failure taxonomy (mirror-after-error 13.9%, broken code 22.9%, error ⇒ match drop 51.4%→12.1%). Needs a better model + harness + n≥3 games | README §tool-calling arm → `../results/toolagent_traces/run0.jsonl`; `archive/TOOLCALLING_COMPARISON.md` |
| 5 | Decision-native model (Jev, TypeSafe): typed questions, typed answers + calibrated probabilities, no generation | Bounded, state-conditioned model judgment beats the walk-forward floor **only on noisy demand** — n=10 read −18.2% (10/10, t=−3.89), n=20 read −7.1% (15/20 paths, t=−2.15); loses on step (+22%) and chaotic/wild. Wins where tuning is weak; gate closes (85.2% deferral) where the model is valuable | `waves/WAVE5.md`, `audits/AUDIT4.md` `audits/AUDIT5.md` → `../results/jev_protocol/` `../results/jev_controls_*/` `../results/jev_noisy20/` |
| 6 | GRPO RL post-training on our harness — Qwen3-0.6B free-form lane (`a0_base` + variants) on Umbrella | **NOT READABLE as a GRPO test yet** — see OPEN ISSUES 1. Eval report: step arm trained vs untrained mean +14.7% (paired), both arms rogue-order-dominated (188 / 58 rogue orders); training-time parse-fail 0.847 | WAVE6_SPEC (design); `../hpc-umbrella/umbrella-results/eval-a0-report.md` → `../results/local-eval/` `../results/wave7_logs/` |
| 7 | Decision-native training: closed-choice letter-logit readout (orders 0–40 step 2), soft-target SFT + RLCD arm on Qwen3-0.6B, 3 seeds | **Interface positive, training null-to-negative.** Closed-choice format kills the rogue tail (0 rogue orders, parse_fail=0 across 21 arms / 427 run records, vs 7–188 per wave-6 arm) and gives a stable untrained control (step-argmax 10,464, C2 strong tier). But C1 fails on all six trained arms (0–12/30 wins, sign p 0.90–1.00), the permutation gate fails at 0.000 invariance on all six (+ control), RLCD collapses to single-letter policies (letter-lock), seeds disagree (C4 mixed, binding variant fails) | `waves/WAVE7.md`, `waves/WAVE7_SPEC.md` `waves/W7-READY.md`, `audits/WAVE7-AUDIT-1.md` `audits/WAVE7-AUDIT-2.md`, `verify/VERIFY-W7-EVAL.md` → `../results/eval/` `../results/wave7_labels/` `../results/wave7_{sft,rlcd}_seed{0,1,2}/` `../results/wave7_floors.json` |

Reading order for a newcomer: `../README.md` → root `../README.md` (surviving claim, audit
history) → `audits/AUDIT*.md` (the mistakes, in full) → the wave doc for whatever wave you
are working on.

## Current verdicts

- **The surviving claim (waves 1–5):** a 1970s order-up-to formula that tunes itself from the
  first 18 weeks of history beats the LLM agent in every environment at 3–4 orders of magnitude
  less compute. Every contested word is grounded in `../results/` — see root README.
- **Wave 5's positive:** state-conditioned bounded judgment beats the tuned floor on noisy
  demand (paired, −7.1% at n=20), but not on step/chaotic/wild, and it wins only where tuning is
  weak.
- **Wave 6:** stands as an infrastructure + eval result; its numbers must not be read as a test
  of the paper's GRPO claim (OPEN ISSUES 1).
- **Wave 7:** the closed-choice readout interface is a real, reusable win (parse-free, no rogue
  tail, deterministic, paired); outcome-label training at 0.6B (SFT and RLCD) made policy worse
  and bought no permutation invariance. Options for the next attempt are costed in waves/WAVE7.md §7.

## OPEN ISSUES

> Recorded 2026-10-07 from the orchestrator's audit; not yet written up as a wave doc. Both
> items are **agreed next steps**. Specs written 2026-10-07 (`waves/WAVE6-FIX_SPEC.md`,
> `waves/WAVE8_SPEC.md`; index and ordering notes: `waves/SPECS-2026-10-07-README.md`);
> implementation underway as mission `bullwhip-impl`. Do not start either experiment as part of a
> documentation/cleanup mission.

1. **Wave-6 free-form lane validity.** Parse-fail is 84.5–86.7% across `a0_base` and its 4
   variants, flat over 300 steps. The reward path is parse-blind: `extract_int` returns 0 for
   unparsed output, and that 0 enters the game as a real order-0 decision — so roughly 85% of
   the gradient is not training the intended decision. The lane also serves an instruct
   checkpoint (Qwen3-0.6B) as raw continuation without its chat template, while wave-7's readout
   on the same checkpoint uses the template and has 0 parse failures. Consequence: wave-6
   numbers cannot yet be read as a test of the paper's GRPO claim.
2. **Agreed next steps (specs written 2026-10-07, implementation underway):**
   - (a) a contract/format micro-probe on the free-form lane — as-is vs chat-template + the
     paper's own `order_quantity` JSON answer contract vs + a parse-aware reward;
   - (b) a decision-model arm in the style of Cloudflare's Clef — fixed choice set + one-token
     readout + cost-scored RL, i.e. the RLCD lane with a corrected action grid and a collapse gate.

## Pointers

- Change log (run it what, when, why): root `../CHANGES.md` (newest first).
- Cluster run book: `../hpc-umbrella/OPERATOR.md` + `../hpc-umbrella/README.md`.
- Independent numeric verifications: `verify/VERIFY-2026-10-01.md`, `verify/VERIFY-W7-EVAL.md`.
- Historical progress logs (wave-6 era): `missions/PROGRESS.md`, `missions/PROGRESS_DESKTOP.md`.
- Untracked-but-kept artifact manifest: `ARTIFACTS.md`.
- Where everything lives + how to run it: `../README.md`.