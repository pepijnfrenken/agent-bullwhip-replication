# Claims audit, 2026-11-19

Purpose: for every headline number in `docs/paper/paper.tex`, name the in-repo artifact and field it
can be read from today. The paper's own rule is that no number enters the manuscript unless it can
be read from an artifact and the file named. This note tests that rule against the tree at
`c22b234` (paper content last touched at `f427edb`).

Method: `grep -rn` over tracked `*.md`, `*.json`, `*.tex`, `*.log` for each numeric token, with the
build logs (`paper.log`, `paper-supplement.log`) and the PDFs excluded so a paper sentence cannot
count as its own source. Raw-value greps were also run without thousands separators. No number
below was taken from an earlier draft or from memory.

## What the tree does contain

- `results/eval/`: 49 files, every one prefixed `w7-` (six wave-7 arm-seed rows, three read modes,
  plus permutation JSONs). This is the only evidence class in that directory.
- `results/wave8_logs/`: 41 entries, 40 `*.log` Slurm logs. The `wave8-eval_*.log` files record
  0.6B-rung runs only (`rung=0.6b`), each line printing `runs=`, `mean=`, `min=`, `max=`,
  `min_mass=`, `wall=`.
- `results/wave8/`: `collapse_4b` reads are archived in `docs/verify/W8-MANIFEST.json`.
- `docs/verify/W8-MANIFEST.json` keys: `generated`, `w06_wave8_r1_collapse_reads`,
  `w06_wave8_r1_band_record`, `lr_selection_1.7b_json`, `lr_selection_1.7b_mtime`,
  `stop_1.7b_5e5_seed0`, `collapse_4b`, `perm_exact`, `job_timeline`. No cost field exists for any
  rung, trained or untrained.

## Per-claim result

| claim in paper.tex | in-repo source found | class |
|---|---|---|
| 4B medians 8,315 / 7,612 vs control 21,578 (abstract, §6, S3) | `docs/waves/WAVE8_WORD8B_RESULT.md` lines 6, 57, 92 | verdict doc, not raw |
| 4B 30/30 paired wins, sign p = 2^-30 | no wave-8 per-path file exists in the tree, so the wins and the sign test cannot be recomputed here | unverified in tree |
| 4B reductions 61.5% / 64.7% (main text), 61.5--64.8% (abstract) | none. Supplement S1, cited by §6 for these, holds gate definitions and thresholds only | dangling pointer |
| 1.7B medians 21,468 / 43,660 / 71,270, control 11,508 | none (only `paper.tex`, `DRAFT-v0.md`, and the build logs) | unverified in tree |
| 1.7B selection band means 47,125 / 36,734 / 61,073, pooled 48,311 | `docs/verify/W8-MANIFEST.json` → `lr_selection_1.7b_json.pooled_mean_val_cost` (48,310.96) | good |
| 1.7B seed-1 G7 min_mass 7.33e-09 / 8.93e-09 / 7.85e-09 | not in tree; `results/wave8/r1-1.7b-seed1-S_nom.json` cited by the reconciliation note does not exist | unverified in tree |
| 0.6B wave-7 medians 20,594 / 25,721 / 25,508 / 37,384 / 32,516 / 27,652, control 11,569 | `docs/verify/VERIFY-W7-EVAL.md`, `docs/waves/WAVE7.md`, and the six `results/eval/w7-*argmax-noisy.summary.json` | good |
| 0.6B wave-8 stop-loss, five markers, two resumes, G7 3.2e-04 | `docs/waves/W8-R1-DRAINED.md` line 141/155, `docs/verify/W8-MANIFEST.json` → `w06_wave8_r1_collapse_reads` (16 reads) | good |
| 4B collapse reads, eight rows | `docs/verify/W8-MANIFEST.json` → `collapse_4b` | good |
| 4B permutation 24/512 and 20/512 | `docs/verify/W8-MANIFEST.json` → `perm_exact` | good |
| 8B val-band means 35,289 / 12,444 / 71,855 / 22,083, pooled 23,867 / 46,969 | `docs/waves/WAVE8_WORD8B_RESULT.md` scoreboard | good (already declared band-only) |
| per-rung F3 thresholds | `results/wave8/gates_8b.json` present; `gates_1.7b.json` and `gates_4b.json` absent from tree and from every commit, as supplement S1 already discloses | partly declared |

## Findings that need an edit

1. No wave-8 evaluation JSON is in the tree or in any commit. `git log --all --name-only --
   'results/eval/*'` returns only `w7-*` paths. The reproducibility statement's tier (a) places
   trained-1.7B and trained-4B generations, both generations, on the cluster archive; nothing in
   the tree contradicts that, but nothing in the tree supports it either.
2. The passing rung's numbers rest in-repo on one verdict doc. That is the same evidence class the
   paper assigns to 8B, and the manuscript does not say so for 4B.
3. Supplement S3 labels the 4B rows `raw JSONL`. Under finding 1 the label describes a file that is
   not in the commit. It should name the archive path, or cite the verdict doc, or the label moves
   to the document-recorded class.
4. The §6 pointer to "supplement S1" for the 4B mean costs and the per-path proportional
   reductions does not resolve: S1 has no 4B cost figure, and the three percentages appear nowhere
   in the supplement. Either the missing table is restored to S1 or the pointer is deleted and the
   percentages are sourced inline.
5. Tier (a) of the reproducibility statement is unverified as of this note. The TU/e VPN profile
   died on 2026-10-27 (`~/portal/wireproxy.log`: handshake failed after 20 attempts, then keys
   removed). Cluster access needs a fresh SAML browser login, which is Pino's hand. Until that
   login happens, the claim "full per-run JSON on the primary cluster archive" is a claim, not a
   checked fact.

## Where the raw wave-8 data is, and is not

Searched at Pino's question ("should all data be on device"): `find / -xdev` for `w8-r1-4b*`,
`w8-r1-1.7b*`, `w8-untrained*`, `gates_1.7b.json`, `gates_4b.json`, whole root filesystem, plus a
content grep for the path strings across `*.log`, `*.md`, `*.txt`, `*.json` under `$HOME`.

- Not on this machine. The host has one ext4 root and no extra mounts (`df`), so `find / -xdev`
  covered everything. The only hits for the path strings are chat/agent logs (`~/.hermes/sessions/`,
  `~/.omp/logs/`), which are transcripts, not artifacts.
- Never committed. `git log --all --pretty=format: --name-only | grep -ci '4b'` returns 0 across
  all refs. `results/` holds 328 tracked files and every one is wave-7 or earlier.
- Never pulled back. `hpc-umbrella/push_results.sh` is the rsync-plus-commit path used for wave-7
  ("wave-7 eval results (job 398422)"). No wave-8 submission runs it.
- Other devices are not a fallback: `desktop-wsl` (<desktop-tailnet-ip>) times out, two other tailnet hosts are offline on the tailnet, and the hub VM (<hub-tailnet-ip>) refuses this
  machine's key.
- Two small `wave8-diag-*.tar.gz` files exist in `$HOME` (8 Oct) and contain one Slurm log plus a
  diag text each. No evaluation JSON.

So the on-device evidence for the wave-8 rungs is: 40 Slurm logs, `W8-MANIFEST.json` (collapse and
permutation extracts only), `W8-R1-DRAINED.md`, and `WAVE8_WORD8B_RESULT.md`. The per-run JSONL that
the paper says regenerates every number is on the cluster archive or nowhere.

## Archive verification, done with the tunnel up (2026-11-19)

Fresh TU/e profile installed in `~/portal/tue-wireproxy.conf`, wireproxy running, `ssh tue-cluster`
returns `tue-login001.cluster` as `20234428`. Home is writable.

Result: the project archive is empty.

- `find /scratch-shared/20234428 -type f` returns exactly one file: `/scratch-shared/20234428/.keep`.
  The directories `agent-bullwhip-replication`, `kit`, `logs`, `hf`, `pipcache` and `venv` exist as
  empty shells. `du -sh .../agent-bullwhip-replication` is 0, and `git log` there says "not a git
  repository". The 40 Slurm logs that were pulled to this machine in October are now the only logs.
- No tarball or zip survives anywhere under the scratch root.
- Snellius is not a second copy: `ssh snellius` returns `Permission denied (publickey,password)` for
  the configured key, so the two 8B Snellius legs cannot be reached either.
- The hub VM (<hub-tailnet-ip>) refuses this machine's key for both `root` and `pino`, so no
  hub-side mission copy can be checked.
- Job accounting survives in `sacct` (job ids, elapsed, state), which is process evidence for what
  ran, not data evidence for what it produced. The 4B/1.7B evaluation job `456960 bw8-eval` is there,
  elapsed 01:30:40, COMPLETED 2026-10-21. Its stdout lived in the purged `logs/` directory.

Consequence for the manuscript: the 4B row and the 1.7B row now have no per-run artifact anywhere
we can reach. `docs/waves/WAVE8_WORD8B_RESULT.md` is the only surviving carrier of the 4B triple,
and it labels those values "same metric" as the 8B **mean** val cost (lines 4 and 6). No surviving
document calls 8,315 or 7,612 a median. The paper publishes both as final-eval medians. That
labelling cannot be checked and is not supported by the one document that carries the numbers.

The free lane is still usable: `sinfo -p tue.gpu.q` shows L4 nodes mixed and up. A 4B training leg
in `sacct` ran 01:00 to 04:32 wall, and an evaluation job 01:30. So a re-run of the 4B and 1.7B
rungs is a lane question, not a money question.

## Consequence

Findings 1 to 4 are text and labelling, no new evidence required, and they are fixable without
compute. Finding 5 needs one Pino action (SAML login) before any archive path can be written into
the reproducibility statement or into the arXiv package.

The device search above changes the priority. If the cluster archive has already purged the wave-8
raw files, the 4B row stops being checkable at all, and the paper's only positive rung would need a
retrain of that rung before it could be submitted. Everything, including path A on 8B, now queues
behind one SAML login.
