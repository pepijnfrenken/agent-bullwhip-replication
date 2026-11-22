# W8-8B molab leg 1 — the selected arm, and it passes G1

Run date: 2026-11-22. Lane: molab RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, one
device, 20 CPUs, 160 GB RAM. Stack: torch 2.11.0+cu130, transformers 4.55.4, peft 0.21.1.

This is the arm the frozen selection rule chose (lr 5e-5, seed 0). It is not the arm that
happened to survive October. The three lost legs are being retrained for the same reason.

## Config

`agent_bullwhip.decision_rounds run --arm r1 --seed 0 --lr 5e-5` on `Qwen/Qwen3-8B`
revision `b968826d9c46dd6066d109eabc6255188de91218`, three rounds of 170 steps, batch 32,
max length 1024, gradient checkpointing, save every 10, eight rollout processes, gates
`results/wave8/gates_8b.json`. Weights served from local disk with `HF_HUB_OFFLINE=1`;
the hub lookup failed on the first attempt and was removed from the path entirely.

Frozen inputs verified before the run, sha256:

- `results/wave8/S_nom.jsonl` `f651fc58e5a34c36421848a09e91090275a3e4786743549afb4ad45f0bc6aaa8`
- `results/wave8/gates_8b.json` `b6194407e03adb35d10d4c54edd0a33c14a9b01d4772a551e4202147db8b1f90`

## Timing

62 minutes wall, 14:29Z to 15:31Z. 169 training steps in 706.9 s in round 1, which is
4.2 s/step; rollouts 59.9 s and 40.8 s per pair of state batches. The October A100-80G leg
took 1h28m for the same three rounds.

## Collapse gates

| round | collapsed | I | console medH | console modal_share |
|---|---|---|---|---|
| 1 | false | 0.5017 | 0.001 | 0.895 |
| 2 | false | 0.5103 | — | — |
| 3 | false | 1.0998 | 0.011 | 0.699 |

No round tripped the stop-loss. The JSON payloads carry `medH` and `modal_share` as null and
only `I` is written; the console values above are the printed lines, not JSON fields. That
asymmetry is recorded rather than smoothed over, because a reader who opens the file will
see nulls.

## Band read

30 runs, curated curriculum, argmax: mean **10,892.37**, min 7,168, max 19,524,
`min_mass` 0.99997, `parse_fail` 0, entropy mean 0.0266, confidence mean 0.989,
`wall_sec` 155.1.

Against the matched untrained-8B band **104,676.07** (measured 2026-11-21) the ratio is
**0.104**. Against the October surviving arm's band mean 22,083.37 this arm is roughly half
again cheaper.

## G1, paired, n=30

`scripts/wave8_g1_read.py results/eval/w8-untrained-8b-argmax-noisy.jsonl <arm rows>`

- control median **89,633.00**, arm median **10,097.00**
- median ratio **0.1126**, reduction **88.74%**
- median of per-path reductions 0.1176 (88.24%)
- paired wins **29/30**, one loss, one-sided **p = 2.89e-08**
- all four gates pass: 15% by both statistics, 20/30 wins, p < 0.02, and the paper's 22/30

Pairing is legitimate rather than assumed: the arm's readout `prompt_sha` is
`55bdc663eee13e101be89e1ddf112e48fcafc62d4c8bf4950b39a2b49dafc422`, byte-identical to the
control's, so both reads walked the same 30 demand paths.

## What this changes in the paper

The manuscript currently says 8B is evaluable on one of four arms, and that the arm which
passed is the one the selection rule did not choose. That sentence is now out of date in the
best way: the selected arm passes G1 on the same frozen control. It does not yet close the
rung, because the arm rule wants a second seed at this learning rate. `lr 5e-5 seed 1` is
running, then `lr 1e-5 seed 0`.

## Provenance that must travel with these numbers

These are re-executions on a different stack and a different card from October: torch 2.11
rather than 2.1.2, Blackwell rather than A100, weights read from a local directory rather
than the hub cache. The gates, labels and eval control are byte-identical to the frozen
files, which is what makes the comparison meaningful. Any paper sentence citing these numbers
states this, not just the result.

## Retention, and a transfer failure that has to be on the record

Committed: the 30-row cost extracts `w8-val-r1-8b-lr5e-5-seed{0,1}.costs.jsonl`, the summary
JSONs, and the three collapse-gate JSONs. G1 re-runs to the same verdict from those committed
files alone, so the numbers the paper cites are reproducible from the repo.

Lost, and not hidden: the full-trace eval JSONLs. They were pulled as base64 inside a cell
print and the copies that landed on disk were corrupt — `head` opens correctly and the file
ends mid-object, with only 12 of 30 rows parseable. I reported them as secured before checking
that they parsed, which was wrong. The defect is named: a 750 KB base64 payload does not
survive the cell-output channel intact, while 1 KB extracts do. Anything larger than a few KB
needs an out-of-band route, not a printed blob.

Also lost with the sandbox (`410 sandbox terminated` at about 17:00Z): the LoRA adapters for
both completed arms, and the third arm, `lr 1e-5 seed 0`, which was in its final round and
never wrote a validation read. Its adapter and states are gone with the instance.

What survives is enough for the claim the paper makes: both seeds at the selected learning
rate pass G1 against the frozen control, on 30 paired demand paths, from files in this repo.
Re-running the lost arm and re-deriving the traces costs one GPU session and about an hour,
and the weights are not needed to defend a number the eval rows already carry.

## Legs 2 and 3

Leg 2, the same learning rate with seed 1, ran 15:33Z to 16:35Z on the same lane with the same
config and passed G1 as well: control median 89,633.00, arm median 15,355.00, ratio 0.1713, an
82.87% reduction, 29/30 wins, p = 2.89e-08, band mean 15,410.70, min_mass 0.99973, zero parse
failures. Same `prompt_sha` as the control. Two of two seeds at lr 5e-5 pass, which satisfies
the arm rule that asks for two of three.

The seed spread is real and is stated rather than averaged: two seeds at one learning rate land
at 10,097 and 15,355 on the same control, 1.52x apart. Both clear the bar; neither is quotable
as "the" effect size for the arm.

Leg 3, lr 1e-5 seed 0, started 16:35Z and was in its third round when the sandbox was
reclaimed. No validation read exists for it. That arm is not evaluable in this attempt.
