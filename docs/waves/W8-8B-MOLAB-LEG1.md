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

## G1, paired, n=30 — CORRECTED

The first version of this note reported these legs as passing G1 with 29/30 wins and
p = 2.9e-08. That is withdrawn. The legs were run with `--pattern curriculum`, so their 30 runs
share demand paths with the curriculum control and none with the noisy control. Checked against the
surviving traces: 12 of 12 parseable seed-0 rows match `untrained-curriculum-8b.jsonl` on
`traces.retailer[*].ctx.incoming_now`, and 0 of 12 match `w8-untrained-8b-argmax-noisy.jsonl`.
The prompt SHA is identical in all four files because the rendered prompt is hashed independently
of the demand draw, so it never certified a matched pair.

What actually holds, band against band, on the curriculum control (median 93,602.5, mean 104,676.07):

- seed 0: 30/30 wins, median ratio 0.108, one-sided p = 9.31e-10
- seed 1: 28/30 wins, median ratio 0.164, one-sided p = 4.34e-07

That is a favourable band comparison, not the frozen estimand. G1 is defined on the paired
noisy-argmax generation, and no such read exists for either November leg. The selected rate's G1
outcome is unestablished. The one 8B arm with a pre-registered paired pass remains the October
1e-5 seed-1 survivor (14,595 vs 89,633, 29/30, p = 2.89e-08).

Found by two independent audit lanes on 2026-11-22 and confirmed here from the files before being
accepted.
