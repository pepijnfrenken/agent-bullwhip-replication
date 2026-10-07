#!/usr/bin/env python3
"""verify_pull.py <results-tree> [--size] — completeness check for a pulled results tree.

Expects the 13 wave-6 sweep dirs under <dir>/results/ (pilot + a0..a11), each with state.json,
train_log.jsonl and checkpoint-* dirs (save cadence: every 10 steps, ...9/...9). Prints a per-run
table and an overall verdict; exits 1 on any gap. Stdlib only; safe to run on the laptop or the hub.

Usage (hub side, after `git pull --ff-only` — the same check ran on the old pull_results.sh dest):
  python3 verify_pull.py ~/projects/agent-bullwhip-replication --size
"""
import argparse
import json
import pathlib
import sys

EXPECT = {
    # Wave 7 (decision-native): one SFT dir and one RLCD dir per training seed
    # (RLCD writes its own wave7_rlcd_seed*/, so the SFT dir ends at
    # W7_SFT_STEPS). Value = total optimizer steps (last logged step = value-1);
    # must match the gpu-wave7-train.sbatch knobs — adjust if you override them.
    "wave7_sft_seed0": 120,
    "wave7_sft_seed1": 120,
    "wave7_sft_seed2": 120,
    "wave7_rlcd_seed0": 39,
    "wave7_rlcd_seed1": 39,
    "wave7_rlcd_seed2": 39,
    "grpo_pilot": 4,
    "grpo_a0_base": 300,
    "grpo_a1_group4": 300,
    "grpo_a2_group16": 300,
    "grpo_a3_horizon24": 300,
    "grpo_a4_horizon52": 300,
    "grpo_a5_poisson_only": 300,
    "grpo_a6_truncnorm_only": 300,
    "grpo_a7_kl_low": 300,
    "grpo_a8_kl_high": 300,
    "grpo_a9_lr_low": 300,
    "grpo_a10_lr_high": 300,
    "grpo_a11_seed2": 300,
    # WAVE6-FIX probe (hpc-umbrella/campaign-w6fix.tsv). A1/A2 early-stopped at
    # step 50 (W6F §6.2, P50 < 0.02) report INCOMPLETE here by design.
    "grpo_w6fix_a0": 50,
    "grpo_w6fix_a0b": 25,
    "grpo_w6fix_a1": 100,
    "grpo_w6fix_a1_seed2": 100,
    "grpo_w6fix_a2": 100,
    "grpo_w6fix_a2_seed2": 100,
    # WAVE8 R1 (decision_rounds): flat per-round train dirs, 170 steps each, 2 lrs x 3 seeds
    # x 3 rounds. R2 / R3 / ladder dirs (wave8_r2sft_seed*, wave8_r2rl_*, wave8_r1-1.7b_*,
    # wave8_r1-4b_*) run conditionally and are listed as "extras (fine)" unless added here.
    "wave8_r1_lr1e-5_seed0_round1": 170,
    "wave8_r1_lr1e-5_seed0_round2": 170,
    "wave8_r1_lr1e-5_seed0_round3": 170,
    "wave8_r1_lr1e-5_seed1_round1": 170,
    "wave8_r1_lr1e-5_seed1_round2": 170,
    "wave8_r1_lr1e-5_seed1_round3": 170,
    "wave8_r1_lr1e-5_seed2_round1": 170,
    "wave8_r1_lr1e-5_seed2_round2": 170,
    "wave8_r1_lr1e-5_seed2_round3": 170,
    "wave8_r1_lr5e-5_seed0_round1": 170,
    "wave8_r1_lr5e-5_seed0_round2": 170,
    "wave8_r1_lr5e-5_seed0_round3": 170,
    "wave8_r1_lr5e-5_seed1_round1": 170,
    "wave8_r1_lr5e-5_seed1_round2": 170,
    "wave8_r1_lr5e-5_seed1_round3": 170,
    "wave8_r1_lr5e-5_seed2_round1": 170,
    "wave8_r1_lr5e-5_seed2_round2": 170,
    "wave8_r1_lr5e-5_seed2_round3": 170,
}
CKPT_PAYLOAD = ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json",
                "pytorch_model.bin")


def last_trainlog_step(p):
    last, n = None, 0
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get("step"), (int, float)):
            last = d["step"]
    return n, last


def state_step(p):
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None, (p.stat().st_size if p.exists() else 0)
    for k in ("step", "next_step", "global_step", "last_step"):
        if k in d and isinstance(d[k], (int, float)):
            return d[k], None
    return None, None


def dir_size(p):
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--size", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(args.root).expanduser()
    res = root / "results"
    if not res.is_dir():
        sys.exit(f"FAIL: {res} not found (pass the umbrella-results dir; expect <dir>/results/...)")
    problems = []
    print(f"{'run':<22} {'state':>7} {'logN':>5} {'last':>4} {'ckpts':>5} {'range':<12} {'payload':<8}")
    for name, target in EXPECT.items():
        d = res / name
        if not d.is_dir():
            problems.append(f"{name}: MISSING dir")
            print(f"{name:<22} MISSING")
            continue
        st, sz = state_step(d / "state.json")
        tl = d / "train_log.jsonl"
        n = last = None
        if tl.is_file():
            n, last = last_trainlog_step(tl)
        cks = sorted(int(p.name.split("-")[-1]) for p in d.glob("checkpoint-*")
                     if p.is_dir() and p.name.split("-")[-1].isdigit())
        ckrange = f"{cks[0]}..{cks[-1]}" if cks else "-"
        payload = "?"
        if cks:
            lastck = d / f"checkpoint-{cks[-1]}"
            payload = "yes" if any((lastck / f).exists() for f in CKPT_PAYLOAD) else "EMPTY"
        stx = st if st is not None else (f"~{sz}B" if sz else "?")
        print(f"{name:<22} {str(stx):>7} {str(n):>5} {str(last):>4} {len(cks):>5} {ckrange:<12} {payload:<8}")
        if target > 4:
            if last is None or last < target - 1:
                problems.append(f"{name}: train_log last step {last} < {target - 1}")
            if st is not None and st < target - 2:
                problems.append(f"{name}: state step {st}")
            if not cks:
                problems.append(f"{name}: no checkpoints")
            elif cks[-1] < target - 12:
                problems.append(f"{name}: last checkpoint {cks[-1]}")
            if payload == "EMPTY":
                problems.append(f"{name}: last checkpoint has no adapter payload")
        elif not n:
            problems.append(f"{name}: no train_log records")
    extras = [d.name for d in sorted(res.iterdir()) if d.is_dir() and d.name not in EXPECT]
    if extras:
        print("\nextras (fine):", ", ".join(extras))
    if args.size:
        print("total results size:", round(dir_size(res) / 1e6, 1), "MB")
    print()
    if problems:
        print("VERDICT: INCOMPLETE")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("VERDICT: COMPLETE - all runs present with final artifacts")


if __name__ == "__main__":
    main()
