# hpc-umbrella — run kit for TU/e Umbrella (Slurm)

Operational kit for the Wave-6 GRPO training runs on TU/e Umbrella (free institutional Slurm,
L4 nodes). The run book, in order:

1. `probe.sh` — login-node probe; paste the report back before pinning anything.
2. `config.sh` — THE pins file (`--print` / `--pins` / `--check`; fails on any unresolvable pin,
   including sbatch-header drift).
3. `stage.sh` — pools the kit to scratch, clones/updates the repo, builds the venv
   (`requirements-umbrella.txt`, two env tracks), downloads the model.
4. `sbatch cpu-harness.sbatch` — CPU-only validation (free, immediate).
5. `sbatch gpu-pilot.sbatch` — the unblock test + the Snellius-decision numbers.
6. `sbatch --array=0-12 gpu-campaign.sbatch` — the 13 pre-registered configs (`campaign.tsv`).
7. `pull_results.sh` — mirror results off scratch (it expires 14 days after the last write).

Start at **`OPERATOR.md`** (run order + expected outputs + failure playbook); verified cluster
facts live in `00-PROBE-AND-SETUP.md`. All scripts resolve their own directory, so run them from
wherever this directory lives.

Vendored into the repo on 2026-09-30 so one public `git clone` on the cluster brings both the
code and this kit.
