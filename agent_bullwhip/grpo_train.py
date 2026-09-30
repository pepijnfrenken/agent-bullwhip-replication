"""Wave 6 GRPO training driver (desktop 3060 Ti, or a TU/e Umbrella L4; D3: Modal OUT).

D1: TRL `GRPOTrainer` with a custom rollout source. The env side lives in
`grpo_env.py`; this file owns the training loop wiring:

  per step:
    1. sample G curriculum episodes under the CURRENT policy (plain
       transformers generate inside torch.no_grad for rollouts — 0.6B short
       integer completions are cheap enough that vLLM is not worth the
       dependency for the pilot; revisit for 1.2B+). Rollouts are **batched**:
       all decisions of one week (G episodes x 4 roles) go through one
       `generate()` call per `--rollout-batch` prompts, because those decisions
       are causally independent (see `grpo_env.run_episodes_batched`)
    2. compute system-level return-to-go rewards + group-normalized advantages
       per (agent, week) across the G episodes (grpo_env.group_normalize)
    3. one policy-gradient step on advantage-weighted logprobs of the emitted
       integer tokens, with KL-to-frozen-reference penalty and grad clipping

This is the hand-rolled inner loop *shaped exactly like TRL's* so that swapping
in `GRPOTrainer` later (or falling back to it) touches only this file. Kept
explicit and auditable per repo rules — no silent defaults.

The loop itself is torch-free and policy-agnostic: it drives any `PolicyLike`
(duck-typed: `.rollout(list[str]) -> (seqs, texts)` and `.grpo_step(...)`), so
the hub can unit-test and dry-run the full step loop with a stub policy. All
torch imports stay inside `_Policy` (module imports on a torch-less box).

Usage (desktop / Umbrella compute node, inside the wave6 venv):
    python -m agent_bullwhip.grpo_train --model Qwen/Qwen3-0.6B \
        --steps 300 --group 8 --out results/grpo_a0 --resume

Success criteria / eval protocol: WAVE6_SPEC.md (never trained on step demand).
Slurm run kit (probe/config/sbatch/runbook): `~/hpc-umbrella/OPERATOR.md`.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable, Protocol

from .engine import SimConfig
from .grpo_env import (group_normalize, run_episodes_batched,
                       sample_curriculum_demand)
from .grpo_trl import ep_result_to_rewards


# --------------------------------------------------------------------------- #
# Policy contract. The training loop never touches torch itself; it only needs
# these two operations. `_Policy` (below) is the real desktop implementation;
# tests/dry-runs inject a stub.
# --------------------------------------------------------------------------- #
class PolicyLike(Protocol):
    """Duck-typed policy the loop drives (one shared policy across all agents)."""

    def rollout(self, prompts: list[str], temperature: float = 1.0,
                max_new: int = 8) -> tuple[object, list[str]]:
        """Generate integer completions for `prompts`; returns (token seqs, texts)."""
        ...

    def grpo_step(self, batch_prompts: list[str], batch_completions: list[str],
                  advantages: list[float], **kwargs) -> float:
        """One policy-gradient step on the flat per-decision batch; returns loss."""
        ...


# --------------------------------------------------------------------------- #
# Prompt-token bookkeeping: TRL-free path needs (prompt_ids, completion_ids,
# completion_logprobs) per decision. We tokenize once per unique prompt prefix
# (the big constant role/instruction block) — see _PromptCache.
# --------------------------------------------------------------------------- #
class _Policy:
    """Thin autoregressive wrapper: generate integer completions + logprobs.

    Torch/peft/transformers imports are lazy (method scope) so this module
    imports cleanly on the hub, where the tests run without torch.
    """

    def __init__(self, model_name: str, device: str = "auto", lora_r: int = 16):
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.lora = LoraConfig(r=lora_r, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "v_proj"])
        self.model.to(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    # -- rollout: sample completions under current policy -------------------
    def rollout(self, prompts: list[str], temperature: float = 1.0, max_new: int = 8):
        import torch
        tok, model = self.tok, self.model
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=max(temperature, 1e-4),
                max_new_tokens=max_new, min_new_tokens=1,
                output_scores=False, return_dict_in_generate=True, pad_token_id=tok.pad_token_id,
            )
        seqs = out.sequences[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(seqs, skip_special_tokens=True)
        return seqs, texts

    # -- score: logprob of given completions under current policy -----------
    def token_logprobs(self, prompt_ids, completion_ids):
        import torch
        tok, model = self.tok, self.model
        input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids).logits
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = input_ids[:, 1:]
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0, prompt_ids.shape[0] - 1:]

    # -- one gradient step ---------------------------------------------------
    def grpo_step(self, batch_prompts, batch_completions, advantages,
                  lr=1e-5, beta_kl=0.02, clip=0.2, max_grad_norm=0.5,
                  micro_bs: int = 4):
        """ advantages: per-decision, aligned with the two batch lists. """
        import torch
        opt = getattr(self, "_opt", None)
        if opt is None:
            from peft import PeftModel, get_peft_model
            # resume path hands us an already-wrapped PeftModel; never wrap twice
            # (double LoRA would stack two adapters on every linear).
            if not isinstance(self.model, PeftModel):
                self.model = get_peft_model(self.model, self.lora)
                self.model.print_trainable_parameters()
            opt = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad], lr=lr)
            self._opt = opt

        model, tok = self.model, self.tok
        model.train()
        total_loss, n = 0.0, 0
        for i in range(0, len(batch_prompts), micro_bs):
            ps = batch_prompts[i:i + micro_bs]
            cs = batch_completions[i:i + micro_bs]
            advs = advantages[i:i + micro_bs]
            enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
            comp = tok(cs, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            # aligned per-row advantages as a tensor
            a = torch.tensor(advs, dtype=torch.float32, device=model.device)

            full = torch.cat([enc["input_ids"], comp["input_ids"]], dim=1)
            attn = torch.cat([enc["attention_mask"], comp["attention_mask"]], dim=1)
            logits = model(input_ids=full, attention_mask=attn).logits
            logp_all = torch.log_softmax(logits[:, :-1, :], dim=-1)
            # completion token positions: prompt_len-1 ... prompt_len+comp_len-2
            plen = enc["input_ids"].shape[1]
            clen = comp["input_ids"].shape[1]
            tgt = full[:, 1:]
            token_logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(token_logp)
            rows = torch.arange(token_logp.shape[0], device=model.device).unsqueeze(1)
            # per-row completion span (right-padded completions)
            comp_lens = comp["attention_mask"].sum(dim=1)
            for r in range(token_logp.shape[0]):
                s = plen - 1
                e = s + int(comp_lens[r])
                mask[r, s:e] = 1.0
            seq_logp = (token_logp * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # policy-gradient loss with PPO-style clip (no old-logprob storage:
            # single inner epoch per batch = ratio is 1 on first pass; the clip
            # guards the KL-trust region across optimizer noise instead)
            loss_pg = -(a * seq_logp).mean()

            # KL to the FROZEN BASE reference (paper Eq. 25's pi_ref): beam-game
            # base weights never train (only LoRA adapters do), so the reference
            # forward must EXCLUDE the LoRA deltas. `peft.PeftModel.disable_adapter()`
            # (context manager; peft>=0.5, verified against peft main + v0.11+ docs)
            # temporarily disables adapter layers — do NOT use
            # `model.get_base_model()(...)` here: the LoRA layers live INSIDE the
            # base modules, so unwrapping still runs adapters -> a post-LoRA
            # reference that would silently train against the policy itself.
            with torch.no_grad():
                if not hasattr(model, "disable_adapter"):
                    raise RuntimeError(
                        "KL reference needs peft.PeftModel.disable_adapter() (peft>=0.5). "
                        f"Installed peft lacks it; upgrade the desktop venv.")
                with model.disable_adapter():
                    was_training = model.training
                    model.eval()
                    ref_logits = model(input_ids=full, attention_mask=attn).logits
                    model.train(was_training)
            ref_logp_all = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_seq_logp = (ref_logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            loss_kl = (seq_logp.exp() * (seq_logp - ref_seq_logp)).mean()

            loss = loss_pg + beta_kl * loss_kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_grad_norm)
            opt.step()
            total_loss += float(loss.detach())
            n += 1
        return total_loss / max(n, 1)


# --------------------------------------------------------------------------- #
# Rollout collection for one training step
# --------------------------------------------------------------------------- #
def collect_step(policy: PolicyLike, G: int, curriculum: str, horizon: int = 36,
                 seed: int | None = None, temperature: float = 1.0,
                 rollout_batch: int = 16):
    """Run G episodes under `policy`; returns flat (prompts, completions, advantages).

    Rollouts are **batched**: the decisions of one week (4 roles) across all G
    episodes are causally independent — the engine builds each role's ctx from
    state at the start of the week and never exposes a same-week order to another
    agent (AUDIT2 leak fix) — so they are submitted in a single
    `policy.rollout()` call, chunked to at most `rollout_batch` prompts per call.
    For the default G=8 that is 32 prompts/call and 36 calls per step instead of
    the previous 2,304 single-prompt calls. See `grpo_env.run_episodes_batched`
    for the invariance argument and the test that pins it against the engine.

    `rollout_batch=1` dispatches one prompt per call (the tiny-VRAM escape
    hatch); `rollout_batch<=0` puts every pending decision of a week in one call.

    Returns three parallel lists, each of length G * 4 * horizon, ordered by
    episode then (t, agent_id) — the same row order `grpo_step` consumes.
    """
    rng = random.Random(seed)
    demands = [sample_curriculum_demand(rng, curriculum, horizon) for _ in range(G)]

    def batch_policy(prompts: list[str]) -> list[str]:
        _, texts = policy.rollout(prompts, temperature=temperature)
        return list(texts)

    episodes = run_episodes_batched(batch_policy, demands,
                                    SimConfig(horizon=horizon),
                                    chunk=rollout_batch)
    rewards = [ep_result_to_rewards(e) for e in episodes]
    adv = group_normalize(rewards)
    prompts_all: list[str] = []
    completions_all: list[str] = []
    flat_adv: list[float] = []
    for e, a in zip(episodes, adv):
        for rec in e.records:
            prompts_all.append(rec.prompt)
            completions_all.append(rec.completion)
            flat_adv.append(a[rec.agent_id][rec.t])
    return prompts_all, completions_all, flat_adv


# --------------------------------------------------------------------------- #
# Main loop with checkpoint/resume (desktop powers off nightly — D3 guard).
# The step loop itself lives in `run_loop(policy, args, ...)` so tests and the
# dry-run driver can drive it with a stub policy and no torch.
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="agent_bullwhip.grpo_train")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=36)
    ap.add_argument("--curricula", default="poisson,truncnorm")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta-kl", type=float, default=0.02)
    ap.add_argument("--rollout-batch", type=int, default=16,
                    help="max prompts per generate() call during rollouts; "
                         "1 = one prompt per call (tiny-VRAM path), <=0 = unbounded")
    ap.add_argument("--out", default="results/grpo_a0")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def run_loop(policy: PolicyLike, args: argparse.Namespace, log_path: Path,
             ckpt_path: Path, start_step: int = 0, rng: random.Random | None = None,
             has_torch: Callable[[], bool] = _torch_available) -> None:
    """Core training loop: curriculum rotation -> collect_step -> grpo_step -> log.

    `has_torch` gates checkpoint saves so the hub (no torch) can exercise the
    full loop with a stub policy; tests inject a fake. Checkpointing (step %
    25 == 0 or last step) writes the PEFT adapter + tokenizer to `ckpt_path`.
    """
    rng = rng or random.Random(12345)
    curricula = args.curricula.split(",")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for step in range(start_step, args.steps):
        curriculum = curricula[step % len(curricula)]
        prompts, completions, advs = collect_step(
            policy, G=args.group, curriculum=curriculum, horizon=args.horizon,
            seed=rng.randint(0, 2**31), rollout_batch=args.rollout_batch)
        loss = policy.grpo_step(prompts, completions, advs,
                                lr=args.lr, beta_kl=args.beta_kl)
        rec = {
            "step": step, "curriculum": curriculum, "loss": loss,
            "mean_adv": sum(advs) / len(advs),
            "min_adv": min(advs), "max_adv": max(advs),
            "elapsed_s": round(time.time() - t0, 1),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step == args.steps - 1:
            print(rec, flush=True)
        if (step % 25 == 0 or step == args.steps - 1) and has_torch():
            policy.model.save_pretrained(ckpt_path)  # type: ignore[attr-defined]
            tok = getattr(policy, "tok", None)
            if tok is not None:
                tok.save_pretrained(ckpt_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= 0:
        print("steps<=0: nothing to do")
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_log.jsonl"
    ckpt_path = outdir / "checkpoint"

    policy = _Policy(args.model)
    start_step = 0
    if args.resume and ckpt_path.exists():
        from peft import PeftModel
        policy.model = PeftModel.from_pretrained(
            policy.model.get_base_model() if hasattr(policy.model, "get_base_model")
            else policy.model, ckpt_path)
        done = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        start_step = len(done)
        print(f"resumed from step {start_step}")

    run_loop(policy, args, log_path, ckpt_path, start_step=start_step)
    print("done:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())