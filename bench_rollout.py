"""Wave 6 rollout-batching benchmark — the numbers that decide the Snellius escalation.

Measures, on one GPU and through the *real* training path (`grpo_train.collect_step` ->
`grpo_env.run_episodes_batched` -> the model's `generate()`), what batching actually buys:

  baseline   rollout_batch=1   one prompt per generate() call (the pre-D1 behaviour)
  batched    rollout_batch=N   N prompts per call (the 4 same-week roles x G episodes)

Per chunk size it records: wall-clock per training step, time inside `generate()`,
`generate()` calls, µs per call and per decision, generated tokens, tokens/s, and peak
VRAM. Speedup is reported against the smallest chunk measured (the sequential baseline).

Run on a pilot GPU node (inside gpu-pilot.sbatch):
    python3 bench_rollout.py --model Qwen/Qwen3-0.6B --group 8 --horizon 36 \
        --batches 1,4,16,32 --repeats 2 --out results/grpo_bench/bench_$SLURM_JOB_ID.json

Run anywhere, no torch needed (validates the harness itself; numbers are synthetic):
    python3 bench_rollout.py --stub --group 4 --horizon 6 --batches 1,4 --repeats 1 \
        --out /tmp/bench_stub.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import time
from pathlib import Path

from agent_bullwhip.grpo_train import collect_step, harden_compile_path


class _TimedPolicy:
    """Wraps a PolicyLike and instruments `rollout()` (calls, prompts, tokens, seconds)."""

    def __init__(self, inner):
        self.inner = inner
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.prompts_seen = 0
        self.tokens = 0
        self.generate_s = 0.0

    def rollout(self, prompts, temperature: float = 1.0, max_new: int = 8):
        t0 = time.perf_counter()
        texts, ids, logps = self.inner.rollout(prompts, temperature=temperature,
                                               max_new=max_new)
        self.generate_s += time.perf_counter() - t0
        self.calls += 1
        self.prompts_seen += len(prompts)
        self.tokens += sum(len(i) for i in (ids or [])
                           if i is not None)
        return texts, ids, logps

    def grpo_step(self, *a, **kw):
        return self.inner.grpo_step(*a, **kw)


class _StubPolicy:
    """Torch-free stand-in: a fixed cost per call plus a per-prompt cost, so the
    batching effect is visible (and the harness itself is testable off-GPU)."""

    def __init__(self, per_call_s: float = 0.004, per_prompt_s: float = 0.0002):
        self.per_call_s, self.per_prompt_s = per_call_s, per_prompt_s

    def rollout(self, prompts, temperature: float = 1.0, max_new: int = 8):
        time.sleep(self.per_call_s + self.per_prompt_s * len(prompts))
        # synthetic ids keep the benchmark's token counters alive; id 0 is the
        # first vocab id, NOT a pad token — these are dummy completions that are
        # never decoded (the bench counts ids only, which is the F2 path)
        ids = [[0] * max_new for _ in prompts]
        return ["8"] * len(prompts), ids, [None] * len(prompts)

    def grpo_step(self, batch_prompts, batch_completions, advantages, **kw):
        time.sleep(0.002)
        return 0.0


def _model_factory(model_id: str):
    """Build the real torch policy exactly as the trainer does (hardened compile path)."""
    harden_compile_path()
    import torch  # noqa: PLC0415 - lazy: keeps --help/--stub torch-free
    from agent_bullwhip.grpo_train import _Policy

    policy = _Policy(model_id)
    return policy, torch


class _TimedWithVRAM(_TimedPolicy):
    """Timed policy + CUDA peak-VRAM readings (0 when running off-GPU)."""

    def __init__(self, inner, torch):
        super().__init__(inner)
        self._torch = torch

    def _cuda(self) -> bool:
        return self._torch is not None and self._torch.cuda.is_available()

    def sync(self) -> None:
        """CUDA work is async; without this every timing below is a lie."""
        if self._cuda():
            self._torch.cuda.synchronize()

    def start_window(self) -> None:
        if self._cuda():
            self._torch.cuda.reset_peak_memory_stats()

    def peak_gb(self) -> float:
        return round(self._torch.cuda.max_memory_allocated() / 1e9, 3) if self._cuda() else 0.0


def bench_one(timed: _TimedWithVRAM, *, group: int, horizon: int, curriculum: str,
              seed: int, chunk: int, repeats: int, temperature: float,
              skip_train_step: bool):
    """Time `repeats` full training steps at one chunk size; return the aggregate."""
    rows = []
    for r in range(repeats):
        timed.reset()
        timed.start_window()
        timed.sync()
        t0 = time.perf_counter()
        roll = collect_step(
            timed, G=group, curriculum=curriculum, horizon=horizon,
            seed=seed + r, temperature=temperature, rollout_batch=chunk)
        timed.sync()
        collect_s = time.perf_counter() - t0
        train_s = 0.0
        if not skip_train_step:
            t1 = time.perf_counter()
            timed.grpo_step(roll.prompts, roll.completions, roll.advantages)
            timed.sync()
            train_s = time.perf_counter() - t1
        rows.append({
            "repeat": r,
            "collect_wall_s": round(collect_s, 4),
            "train_step_wall_s": round(train_s, 4),
            "step_wall_s": round(collect_s + train_s, 4),
            "generate_calls": timed.calls,
            "generate_wall_s": round(timed.generate_s, 4),
            "generated_tokens": timed.tokens,
            "peak_vram_gb": timed.peak_gb(),
        })
    decisions = group * 4 * horizon
    med = lambda key: statistics.median([row[key] for row in rows])  # noqa: E731
    gen_s = med("generate_wall_s")
    calls = med("generate_calls")
    tokens = med("generated_tokens")
    return {
        "rollout_batch": chunk,
        "episodes_per_step": group,
        "decisions_per_step": decisions,
        "step_wall_s": round(med("step_wall_s"), 4),
        "collect_wall_s": round(med("collect_wall_s"), 4),
        "train_step_wall_s": round(med("train_step_wall_s"), 4),
        "generate_calls": calls,
        "generate_wall_s": round(gen_s, 4),
        "us_per_generate_call": round(1e6 * gen_s / calls, 1) if calls else None,
        "us_per_decision": round(1e6 * gen_s / decisions, 1),
        "generated_tokens": tokens,
        "tokens_per_s": round(tokens / gen_s, 1) if gen_s else None,
        "peak_vram_gb": med("peak_vram_gb"),
        "repeats": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="batched vs sequential rollout benchmark")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--curriculum", default="poisson")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batches", default="1,4,16,32",
                    help="comma-separated rollout_batch values (1 = sequential baseline)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--skip-train-step", action="store_true",
                    help="time only the rollout collection (no grpo_step)")
    ap.add_argument("--stub", action="store_true",
                    help="torch-free synthetic policy (harness self-test)")
    ap.add_argument("--out", required=True, help="JSON output path")
    args = ap.parse_args(argv)

    batches = [int(b) for b in str(args.batches).split(",") if b.strip() != ""]
    if not batches:
        ap.error("--batches must list at least one chunk size")

    torch = None
    if args.stub:
        policy, env_note = _StubPolicy(), "stub policy: synthetic timings (no torch, no model)"
    else:
        policy, torch = _model_factory(args.model)
        env_note = "real model; harden_compile_path() applied before import torch"

    timed = _TimedWithVRAM(policy, torch)
    runs = []
    for chunk in batches:
        row = bench_one(timed, group=args.group, horizon=args.horizon,
                        curriculum=args.curriculum, seed=args.seed, chunk=chunk,
                        repeats=args.repeats, temperature=args.temperature,
                        skip_train_step=args.skip_train_step)
        runs.append(row)
        print(f"rollout_batch={chunk:>3}  step={row['step_wall_s']:>7.2f}s  "
              f"calls={row['generate_calls']:>5}  {row['us_per_decision']:>8.1f} us/decision  "
              f"{row['tokens_per_s']} tok/s", flush=True)

    baseline = min(runs, key=lambda r: r["rollout_batch"])
    speedups = {}
    for row in runs:
        if baseline["rollout_batch"] == row["rollout_batch"] or row["step_wall_s"] == 0:
            continue
        speedups[str(row["rollout_batch"])] = round(baseline["step_wall_s"] / row["step_wall_s"], 2)

    torch_ver = transformers_ver = gpu = None
    if torch is not None:
        torch_ver = getattr(torch, "__version__", None)
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        try:
            import transformers  # noqa: PLC0415
            transformers_ver = transformers.__version__
        except ImportError:
            transformers_ver = None

    payload = {
        "tool": "bench_rollout.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "user": os.environ.get("USER"),
        "model": args.model,
        "stub": bool(args.stub),
        "notes": env_note,
        "group": args.group,
        "horizon": args.horizon,
        "curriculum": args.curriculum,
        "repeats": args.repeats,
        "skip_train_step": bool(args.skip_train_step),
        "torch": torch_ver,
        "transformers": transformers_ver,
        "gpu": gpu,
        "baseline_rollout_batch": baseline["rollout_batch"],
        "speedup_vs_baseline": speedups,
        "runs": runs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
