# PROGRESS_DESKTOP.md — Wave-6 desktop bootstrap (hub-driven, done 2026-09-21)

Mission: bootstrap the Windows desktop (WSL2 Ubuntu 24.04, RTX 3060 Ti 8GB) as the bullwhip
compute box, D3 = all-desktop $0. Remote access used: **Path A** (Windows OpenSSH :22 →
`wsl -d Ubuntu-24.04 -- bash -s`), script files piped to stdin. Everything executed via
hub; nothing committed to any repo; no existing desktop processes disturbed
(no GPU compute jobs were running; only Xwayland).

## 1. Inventory (12:47 CEST)

| item | value |
|---|---|
| WSL distro | Ubuntu 24.04.4 LTS, kernel 6.18.33.2-microsoft-standard-WSL2, systemd PID 1 |
| CPU / RAM | 20 cores, 11,958 MB total (10.6 GB free) |
| GPU | NVIDIA GeForce RTX 3060 Ti 8GB, driver 561.17, CUDA 12.6, idle (849 MiB Xwayland), 51C |
| python3 | 3.12.3 (/usr/bin/python3) — lacked python3.12-venv (installed, see §3) |
| git | 2.43.0 |
| bun / omp | absent (installed, see §3) |
| ~/projects | abliteration-local, chessathon, if-ablation (untouched) |
| abliteration-local/.venv | exists (untouched) |
| HF cache | ~/.cache/huggingface = 38 GB |
| WSL IP | 172.27.165.139 |

## 2. What worked (exact patterns)

- **Remote driver** (local hub shell): `ssh -o BatchMode=yes -o ConnectTimeout=10
  pino@100.122.67.126 "wsl -d Ubuntu-24.04 -- bash -s" < /path/script.sh 2>&1 | tr -d '\0\r'`
  Root variant: `wsl -d Ubuntu-24.04 -u root -- bash -s` (no password).
- **Probe proof**: `FREEINFERENCE_API_KEY=<freeinference key> ~/.bun/bin/omp -p
  "Reply with exactly: DESKTOP-OMP-OK"` → printed `DESKTOP-OMP-OK`.

## 3. Installs (in order)

1. **bun** 1.4.2 → `~/.bun/bin/bun`. `curl -fsSL https://bun.sh/install | bash` failed
   (`error: unzip is required to install bun` — desktop lacks unzip); fallback downloaded
   `bun-linux-x64.zip`, extracted with `python3 -m zipfile -e`, copied to `~/.bun/bin/`.
2. **OMP** = `bun install -g @oh-my-pi/pi-coding-agent` → `omp/18.2.7` (matches hub 18.2.7).
   `bun install -g` worked directly (no `add -g` fallback needed).
3. **omp config** = hub's `~/.omp/agent/models.yml` + `config.yml` shipped base64 over the
   ssh pipe, decoded to `~/.omp/agent/` (`chmod 600`). Freeinference key now lives in the
   desktop config; not quoted anywhere in this file.
4. **python3.12-venv** (root): `apt-get update -qq && apt-get install -y -qq python3.12-venv`
   (ensurepip was missing; first venv attempt had failed on it).
5. **`.venv-wave6`** (fresh; `~/projects/abliteration-local/.venv` untouched):
   `cd ~/projects/agent-bullwhip-replication && python3 -m venv .venv-wave6`
   `python -m pip install --upgrade pip` → pip 26.2.1
   `pip install torch --index-url https://download.pytorch.org/whl/cu121`
   `pip install transformers peft trl bitsandbytes accelerate datasets pytest requests numpy pandas`

**Versions:** torch 2.5.1+cu121 · transformers 5.17.0 · peft 0.21.0 · trl 1.13.0 ·
bitsandbytes 0.50.2 · accelerate 1.15.0 · datasets 5.0.1 · pytest 9.1.1 · numpy 2.5.3 ·
pandas 3.0.6 · pip 26.2.1 · git 2.43.0.

**Long-job pattern (learned):** a `setsid nohup … &` child does **not** survive the
`wsl` session exit (killed with the session). Use a systemd transient unit instead:
`systemd-run --unit=<name> --collect --uid=pino --gid=pino --setenv=HOME=/home/pino
--setenv=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash /tmp/script.sh`
(polkit blocks systemd-run as pino; the unit therefore must be launched from the root ssh
path). Install ran this way, log at `~/wave6_install.log`.

## 4. GPU verification

```
torch 2.5.1+cu121 | cuda_available True | NVIDIA GeForce RTX 3060 Ti
```

## 5. Test suite (desktop, wave6 venv)

`cd ~/projects/agent-bullwhip-replication && .venv-wave6/bin/python -m pytest tests/ -q`
→ **101 passed, 1 failed in 9.07s**.

The 1 failure is expected environment skew, NOT a regression:
`../../tests/test_grpo_train.py::test_torch_available_reports_false_on_hub` asserts
`_torch_available() is False` with the comment `# this box has no torch; the loop still ran`.
That assertion is true only on the hub (no torch). The desktop has torch by design
(this mission's whole point), so the test cannot pass here. Hub count "102 passed"
includes it. All 101 other tests — GRPO env/loop, leak guards, floors, curricula — pass
on fresh versions (numpy 2.5.3, pandas 3.0.6, transformers 5.17.0).

## 6. GPU smoke test (go/no-go for pilot — PASSED)

Script: Qwen/Qwen3-0.6B (596M, bf16, device_map cuda), AdamW 1e-5, 2 steps fwd+bwd on 4
short prompts, `torch.cuda.max_memory_allocated()`. Output:

```
model loaded in 40.1s; params=596M
step 0: loss=8.6528
step 1: loss=7.6541
PEAK_VRAM_GB=5.58
total wall 40.6s
```

**Go**: 0.6B QLoRA/bf16 training fits with ~2.4 GB headroom on the 8 GB card.
(Only noise: `torch_dtype` deprecation warning → `dtype`; harmless.)

## 7. Disk before/after

| mount | before | after | delta |
|---|---|---|---|
| `/` (ext4) | 680 GB free | 673 GB free | +7 GB used (venv ~9 GB incl. pip cache 1.9 GB) |
| `/mnt/c` | 23 GB free | 14 GB free | −9 GB |

All installs/caches landed on ext4 (`HF_HOME=~/.cache/huggingface`,
`PIP_CACHE_DIR=~/.cache/pip`) — nothing intentionally written to `/mnt/c`. The C: drop is
structural: the WSL2 distro vhdx lives on C:, so ext4 growth inflates C: used space.
C: is now at 99% (14 GB free). **Operator note (not done — no repo/desktop changes beyond
scope):** `wsl --manage Ubuntu-24.04 --set-sparse true` + vhdx compaction, or moving the
distro to another drive, before the HF cache balloons for 1.2B/4B rungs.

## 8. State at handoff

- `~/projects/agent-bullwhip-replication` @ `afc025ae6da6c4b8f86fdcc58ce10821eb1214a6`
  (`[verified] Wave 6: grpo_train refactor + loop tests`).
- `.venv-wave6` ready; `~/.bun/bin/omp` 18.2.7 with hub-identical models.yml/config.yml
  (freeinference key deployed; probe `DESKTOP-OMP-OK` verified).
- Training CLI exists: `python -m agent_bullwhip.grpo_train --model Qwen/Qwen3-0.6B
  --steps N --group 8 --horizon 36 --curricula poisson,truncnorm --lr 1e-5 --beta-kl 0.02
  --out results/grpo_a0 [--resume]`.

## 9. What remains for the pilot (next mission)

- Run a short real GRPO pilot step series (e.g. `--steps 4 --group 8` → `../../results/grpo_a0_pilot`)
  with `--resume`, confirm per-step reward logs + checkpoint files, then hand off to the
  full A0 rung (defaults: 300 steps, group 8).
- Serve/eval train outputs on CPU (llama.cpp) per ladder plan (A0 rung), $0.
- Revisit `test_torch_available_reports_false_on_hub` so the suite is meaningful on the
  desktop (hub-owned decision; do not edit silently).

## 10. Exact pilot launch command (desktop-side)

From the hub (Path A), to kick the pilot inside WSL, then watch `~/wave6_pilot.log`:

```bash
cat > /tmp/wave6_pilot.sh <<'EOF'
#!/bin/bash
set -u
export PATH="$HOME/.bun/bin:$PATH"
export HF_HOME="$HOME/.cache/huggingface"
export PIP_CACHE_DIR="$HOME/.cache/pip"
cd "$HOME/projects/agent-bullwhip-replication" || exit 1
exec ~/.bun/bin/omp -p \
  "Wave-6 A0 pilot on the desktop. Venv: $HOME/projects/agent-bullwhip-replication/.venv-wave6.
   Run: .venv-wave6/bin/python -m agent_bullwhip.grpo_train --steps 4 --group 8
   --out results/grpo_a0_pilot --resume   (cwd = repo root; env HF_HOME/PIP_CACHE_DIR already set).
   Report: losses per step from results/grpo_a0_pilot/train_log.jsonl, checkpoint files, and
   whether the loop survives step 4. Do not modify repo files except results/GRPO outputs."
EOF
chmod +x /tmp/wave6_pilot.sh
ssh -o BatchMode=yes -o ConnectTimeout=10 pino@100.122.67.126 \
  "wsl -d Ubuntu-24.04 -- bash -s" < /tmp/wave6_pilot.sh 2>&1 | tr -d '\0\r'
```

For multi-hour runs use the §3 systemd-run pattern instead (survives ssh drops):
`systemd-run --unit=wave6-pilot --collect --uid=pino --gid=pino --setenv=HOME=/home/pino
--setenv=PATH=… bash /tmp/wave6_pilot.sh` from the root ssh path.

## 11b. Operator hardening (2026-09-21, inbound update)

- Desktop firewall port 2222 is now restricted to the Tailscale range 100.64.0.0/10; SSH
  rules exclude the Public profile. Operator verified both access paths.
- **DO NOT** run `netsh interface portproxy set/delete/add` or modify any firewall rule —
  doing so could undo the hardening or break remote access. Path B (2222 portproxy) work
  is permanently OFF the table; Path A remains the default.
- FreeInference key handling rule: read from hub `~/.omp/agent/models.yml` when a mission
  needs it; never paste it into any written file. (Desktop `~/.omp/agent/models.yml` =
  deployed hub copy, chmod 600 — that is the intended persistent config.)
- This mission ran **zero** netsh/firewall commands; nothing to undo.

## 11. Anomalies log

- `setsid nohup &` died with the ssh session (first install attempt; log never appeared).
  Root cause: WSL kills session children. Switched to systemd-run — worked.
- Root `cat > /tmp/wave6_install.sh` heredoc hit "Permission denied" once inside a piped
  root script (file existed from pino run); moot — the pre-existing v1 script executed and
  is functionally correct post-ensurepip. Not re-investigated beyond this.
- `/mnt/c` free drifted −2 GB (23→25 GB) before installs began (Windows-side churn, e.g.
  pagefile/vhdx) — see §7 for the install-attributable delta.