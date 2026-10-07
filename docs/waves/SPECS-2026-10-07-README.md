# bullwhip-plans — implementation-ready specs for `agent-bullwhip-replication`

Spec-only directory (written 2026-10-07). Nothing here has been run; nothing here modified the repo
`~/projects/agent-bullwhip-replication`. Inputs staged under `context/` (paper arXiv 2605.17036v3 text,
Cloudflare Clef launch post).

| file | what | status |
|---|---|---|
| [`WAVE6-FIX_SPEC.md`](WAVE6-FIX_SPEC.md) | contract/format micro-probe on the wave-6 free-form GRPO lane: A0 as-is vs A1 paper `order_quantity` JSON contract (chat template) vs A2 + parse-aware reward; parse-rate gates, GPU estimate (≈ 14–24 GPU-h), follow-up rules | **implemented 2026-10-07 (code + tests + staged sbatch; not run)** |
| [`WAVE8_SPEC.md`](WAVE8_SPEC.md) | Clef-style decision-model arm: corrected 26-letter choice set `w8_g26`, R1 outcome-scored RL / R2 soft-target SFT warm start / R3 grid variants, computed collapse gate, permutation gate, schema extension, size ladder 0.6B→1.7B→4B, GGUF CPU step (≈ 30–45 GPU-h) | **awaiting repo-cleanup completion before implementation** |

## Dependencies

* **Post-cleanup doc layout.** Wave docs are cited by filename only: `WAVE6_SPEC.md`, `WAVE7.md`, `WAVE7_SPEC.md`, `W7-READY.md` →
  `docs/waves/`; `WAVE7-AUDIT-1.md`, `WAVE7-AUDIT-2.md` → `docs/audits/`; `VERIFY-W7-EVAL.md` → `docs/verify/`; `STATE.md`, `ARTIFACTS.md` →
  `docs/`. At writing time (repo HEAD `9f04e59`) that layout was already in place. Code paths (`agent_bullwhip/*`, `tests/*`, `scripts/*`,
  `hpc-umbrella/*`) are cited exactly, with line numbers as hints only.
* **Independent of each other except one:** WAVE8's G3 comparator may add a contract-fixed wave-6 arm if WAVE6-FIX runs first; otherwise it uses the
  historical `a0_base` arm. Either order is valid.
* **Both need** the cluster kit (`hpc-umbrella/OPERATOR.md`, `config.sh` pins, `push_results.sh`/`verify_pull.py`) and Pino's approval before
  the specs are committed into the repo (repo rule: pre-registration is committed only when approved).
* `context/paper-2605.17036v3.txt` is a text extraction without the `{}` of the A.2 JSON example — verify against the PDF before freezing the WAVE6-FIX
  contract constant.

## For whoever executes these later (5 lines)

1. Re-read the repo HEAD, `docs/STATE.md` and the root `README.md` research protocol; re-check every file:line in the spec before editing (lines are hints).
2. Execute the spec's **§Implementation change list** exactly; legacy defaults must keep wave-6/7 byte-for-byte reproducible (test first, then change).
3. Run pre-flight/validity gates **before** any GPU job (WAVE6-FIX V0–V3; WAVE8 F0–F3) — a failed gate voids the run, it is not a negative result.
4. Thresholds and decision tables are pre-registered: do not move them after seeing data; log any change in the spec's changes log with the reason.
5. Publish whatever happens (honest-negative clauses), write `docs/waves/WAVE6-FIX.md` / `WAVE8.md` + a `docs/verify/VERIFY-*.md` independent recomputation, and update `docs/STATE.md`, `CHANGES.md`.
