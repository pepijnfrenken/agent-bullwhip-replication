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
(duck-typed: `.rollout(list[str]) -> (texts, ids, old_logprobs)` and
`.grpo_step(...)`), so
the hub can unit-test and dry-run the full step loop with a stub policy. All
torch imports stay inside `_Policy` (module imports on a torch-less box).

Usage (desktop / Umbrella compute node, inside the wave6 venv):
    python -m agent_bullwhip.grpo_train --model Qwen/Qwen3-0.6B \
        --steps 300 --group 8 --out results/grpo_a0 --resume

Success criteria / eval protocol: WAVE6_SPEC.md (never trained on step demand).
Slurm run kit (probe/config/sbatch/runbook): `hpc-umbrella/OPERATOR.md` (in this repo).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .engine import SimConfig
from .grpo_env import (group_normalize, has_int, run_episodes_batched,
                       sample_curriculum_demand)
from .grpo_trl import ep_result_to_rewards


# --------------------------------------------------------------------------- #
# Compile-path hardening (D2). The parked desktop stall: main thread inside
# torch/_tensor.py backward with 21 inductor `compile_worker` processes stuck in
# `_recv_msg`, GPU at 100% util but ~80 W, VRAM oscillating, 0 cached `.py` under
# /tmp/torchinductor_*, and no explicit dynamo compile call anywhere in this repo.
# Suspected cause: per-input-shape compilation from the transformers 5.17.0 /
# torch 2.5.1+cu121 pairing (evidence: /home/pino/diag2.log, PARKED.md).
# Fix step 1 is to disable the compile path outright; the Umbrella module stack
# (PyTorch 2.1.2) is the independent second test of the version hypothesis.
#
# Two knobs, for two different reasons:
#   * TORCHDYNAMO_DISABLE must be in the environment BEFORE `import torch`
#     (torch._dynamo reads it at import time) -> set in _Policy.__init__ first line.
#   * `generation_config.compile` must be cleared AFTER the model is loaded
#     (transformers compiles the decoding loop when the flag is set) -> same call,
#     with the loaded model.
# --------------------------------------------------------------------------- #
def harden_compile_path(model=None) -> dict:
    """Disable the compile path; returns the resolved knob values (for the log).

    Both env knobs use `setdefault`: an explicit operator override (e.g.
    `TORCHDYNAMO_DISABLE=0 python -m agent_bullwhip.grpo_train ...` to test the
    compile hypothesis deliberately) always wins.
    """
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    applied: dict = {
        "TORCHDYNAMO_DISABLE": os.environ["TORCHDYNAMO_DISABLE"],
        "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"],
    }
    if model is not None:
        gen_cfg = getattr(model, "generation_config", None)
        # transformers >=4.50 compiles the decoding loop when this flag is set.
        # If 5.x renamed it, the value shown here is the pilot's confirmation:
        # "absent" means the knob moved and the pilot must re-check (print below).
        if gen_cfg is not None and hasattr(gen_cfg, "compile"):
            gen_cfg.compile = False
            applied["generation_config.compile"] = False
        else:
            applied["generation_config.compile"] = "absent"
    return applied


# --------------------------------------------------------------------------- #
# Policy contract. The training loop never touches torch itself; it only needs
# these two operations. `_Policy` (below) is the real desktop implementation;
# tests/dry-runs inject a stub.
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    """Outcome of one `grpo_step` — the loss plus everything the audit's §3.4
    per-step metrics need, and the F1 nonfinite report.

    The hub's stub policies return `StepResult(self.loss)`; `run_loop` also
    accepts a bare float for legacy callers. `first_nonfinite` holds the FIRST
    offending micro-batch's diagnostics (micro index, loss_pg, loss_kl, s.min(),
    max|logit|) so the runner can save the still-finite weights and abort.
    """
    loss: float
    loss_pg: float | None = None
    loss_kl: float | None = None
    s_min: float | None = None
    s_mean: float | None = None
    frac_s_gt_m001: float | None = None   # fraction of rows with mean token logp > -0.01
    frac_s_lt_m10: float | None = None    # fraction of rows with mean token logp < -10
    entropy_first: float | None = None    # mean first-completion-token entropy (nats)
    grad_norm: float | None = None        # total grad norm returned by clip_grad_norm_ (pre-clip)
    lora_norm: float | None = None        # L2 norm of all trainable (LoRA) params
    n_nonfinite: int = 0                  # micro-batches whose loss/grad was non-finite
    first_nonfinite: dict | None = None
    policy_push: float | None = None      # B4: detached -(a * seq_logp).mean() — the pairable push signal
    max_abs_logratio: float | None = None # D6: max |cur_logp - old_logp| over scored tokens
    clip_frac: float | None = None        # D6: fraction of scored ratios outside [0.8, 1.2]


@dataclass
class StepRollout:
    """One training step's rollouts, flat per decision (episode-major, then
    (t, agent_id)) — the rows `grpo_step` consumes. `completion_ids` and
    `old_logprobs` are F2/F4's sampled identities; `orders` is the parsed order
    (separate field per the audit), `costs` the G episode total system costs.
    """
    prompts: list[str]
    completions: list[str]                     # emitted text (decoded, pads stripped)
    completion_ids: list[list[int] | None]     # raw sampled token ids (F2)
    orders: list[int]                          # parsed order per decision
    costs: list[float]                         # total_sys_cost per episode (G values)
    advantages: list[float]
    old_logprobs: list[list[float] | None]     # per-token rollout logp per decision (F4)


class PolicyLike(Protocol):
    """Duck-typed policy the loop drives (one shared policy across all agents)."""

    def rollout(self, prompts: list[str], temperature: float = 1.0,
                max_new: int = 8) -> tuple[list[str], list[list[int] | None],
                                           list[list[float] | None]]:
        """Generate integer completions; returns (texts, ids, old_logprobs).

        `texts` = decoded completions (special tokens stripped), `ids` = the raw
        sampled token ids (pads stripped — what grpo_step must SCORE), and
        `old_logprobs` = per-token logprobs of those ids under the rollout policy
        (the PPO ratio reference; None when the caller has no torch path).
        """
        ...

    def grpo_step(self, batch_prompts: list[str], batch_completions: list[str],
                  advantages: list[float], **kwargs) -> StepResult:
        """One optimizer step on the flat per-decision batch; returns StepResult."""
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

    def __init__(self, model_name: str, device: str = "auto", lora_r: int = 16,
                 seed: int | None = None):
        # must run BEFORE `import torch`: torch._dynamo reads TORCHDYNAMO_DISABLE
        # at import time (see harden_compile_path).
        harden_compile_path()
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # F1: fp16 is where the audit's divergence dies (Qwen3-0.6B ships bf16
        # weights; fp16 overflow -> inf -> NaN through RMSNorm and the LoRA cast).
        # Load bf16 on CUDA and REFUSE to run fp16: a silent fp16 fallback
        # re-creates the crash, and the tripwire alone cannot rescue a poisoned
        # optimizer state.
        if torch.cuda.is_available():
            assert torch.cuda.is_bf16_supported(), (
                "bf16 required for GRPO training (audit F1): this GPU does not "
                "advertise bfloat16 support — refusing to fall back to fp16")
            dtype = torch.bfloat16
        else:
            dtype = torch.float32   # CPU-only dry runs; not training
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        except TypeError:
            # transformers >= 5 renamed the torch_dtype kwarg -> dtype
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        self.lora = LoraConfig(r=lora_r, lora_alpha=32, lora_dropout=0.05,
                               target_modules=["q_proj", "v_proj"])
        self.model.to(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        # F4: score/rollout always in eval (no dropout) so `old` logprobs at
        # rollout and `current` inside grpo_step are the SAME distribution —
        # within one training step the PPO ratio is then exactly 1 and the clip
        # is a guard, not a noise source. (LoRA dropout is never sampled during
        # scoring; grpo_step does not call .train().)
        self.model.eval()
        # F12: one --seed drives the loop RNG (run_loop) AND the torch RNG, so a
        # replicate config no longer silently clones the seed-12345 run. torch
        # sampling itself is not checkpointed across a resume (only the loop RNG
        # is) — see FIXES-2026-10-01.md.
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.seed = seed
        # second knob: transformers' own decode-path compile flag, now that the
        # model object exists. Printed so a pilot log proves the state it ran in.
        self.compile_knobs = harden_compile_path(self.model)
        print("compile-path knobs:", json.dumps(self.compile_knobs, sort_keys=True), flush=True)

    # -- rollout: sample completions under current policy -------------------
    def rollout(self, prompts: list[str], temperature: float = 1.0, max_new: int = 8):
        """Sample completions; returns (texts, ids, old_logprobs).

        F9: sampling uses the FULL softmax (`top_k=0, top_p=1.0`). The model's
        generation_config ships `top_k=20, top_p=0.95`, so `generate()` without
        explicit values samples from a truncated+renormalized distribution while
        grpo_step scores the full one — the sampled and scored distributions
        must be identical.
        F2: per-row ids (pads stripped) and per-token "old" logprobs under the
        rollout policy are returned so grpo_step scores exactly what was sampled
        and can apply a real PPO ratio (F4).
        """
        import torch
        tok, model = self.tok, self.model
        # Left-pad for generation. The tokenizer default (right padding) makes a
        # batched generate() continue from position-shifted sequences -- transformers
        # warns "right-padding was detected" -- so the batched arm would NOT reproduce
        # the per-prompt (rollout_batch=1) path waves 1-5 ran. Restored in finally;
        # scoring (grpo_step/_old_logprobs) uses its own contiguous-row packing.
        prev_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=True).to(model.device)
            with torch.no_grad():
                # eval: same dropout-free distribution grpo_step scores under
                model.eval()
                out = model.generate(
                    **enc, do_sample=True, temperature=max(temperature, 1e-4),
                    max_new_tokens=max_new, min_new_tokens=1,
                    top_k=0, top_p=1.0,
                    output_scores=False, return_dict_in_generate=True,
                    pad_token_id=tok.pad_token_id,
                )
        finally:
            tok.padding_side = prev_side
        seqs = out.sequences[:, enc["input_ids"].shape[1]:]   # completion tokens only
        texts = tok.batch_decode(seqs, skip_special_tokens=True)
        pad_id = tok.pad_token_id
        ids = [[int(t) for t in row if t != pad_id] for row in seqs.tolist()]
        logps = self._old_logprobs(prompts, ids)
        return texts, ids, logps

    # -- F8: contiguous [prompt|completion] rows, padded only at the end -----
    def _pack_rows(self, prompt_ids: list[list[int]],
                   completion_ids: list[list[int]]):
        """Build a micro-batch of contiguous [prompt|completion] rows so the
        completion starts at its own row's prompt length and arange positions
        match generation. The audit's §3.3: the old layout concatenated a
        right-padded prompt batch with padded completions, so a short row's
        first completion token was read from a PAD position (790/1152 rows)
        and RoPE positions shifted by the pad width.
        """
        import torch
        tok, model = self.tok, self.model
        rows = [p + c for p, c in zip(prompt_ids, completion_ids)]
        plens = [len(p) for p in prompt_ids]
        clens = [len(c) for c in completion_ids]
        maxlen = max((len(r) for r in rows), default=0)
        full = torch.full((len(rows), maxlen), tok.pad_token_id,
                          dtype=torch.long, device=model.device)
        attn = torch.zeros((len(rows), maxlen), dtype=torch.long,
                           device=model.device)
        for r, row in enumerate(rows):
            full[r, :len(row)] = torch.as_tensor(row, dtype=torch.long,
                                                 device=model.device)
            attn[r, :len(row)] = 1
        return full, attn, plens, clens

    # -- score: logprob of given completions under current policy -----------
    def _extract_token_logps(self, logits, full, plens, clens):
        """Per-row per-token full-softmax logprobs of the completion tokens.

        Completion token j sits at row position plen+j, predicted at logits row
        position plen+j-1 with target full[r, plen+j]. Returns plain CPU floats
        (for the rollout-time "old" logprobs; grpo_step does its own tensor
        gather so gradients flow).
        """
        import torch
        logp = torch.log_softmax(logits, dim=-1)
        out: list[list[float]] = []
        for r, (plen, clen) in enumerate(zip(plens, clens)):
            if clen == 0:
                out.append([])
                continue
            tgt = full[r, plen:plen + clen]
            pos = torch.arange(plen - 1, plen + clen - 1, device=logits.device)
            vals = logp[r].index_select(0, pos).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append(vals.detach().cpu().tolist())
        return out

    def _old_logprobs(self, prompts: list[str], completion_ids: list[list[int]]):
        """Per-token logprobs of `completion_ids` under the ROLLOUT policy:
        the F4 "old" distribution for the PPO ratio. Eval mode, no dropout, so
        within one step old == current (ratio == 1; the clip is a guard)."""
        import torch
        tok = self.tok
        prompt_ids = [tok(p, add_special_tokens=True)["input_ids"] for p in prompts]
        full, attn, plens, clens = self._pack_rows(prompt_ids, completion_ids)
        with torch.no_grad():
            self.model.eval()
            logits = self.model(input_ids=full, attention_mask=attn).logits
        return self._extract_token_logps(logits, full, plens, clens)

    # -- optimizer (LoRA params only) ---------------------------------------
    def _ensure_optimizer(self, lr: float):
        """Create the AdamW over LoRA params once; resume reuses it."""
        opt = getattr(self, "_opt", None)
        if opt is not None:
            return opt
        import torch
        from peft import PeftModel, get_peft_model
        # resume path hands us an already-wrapped PeftModel; never wrap twice
        # (double LoRA would stack two adapters on every linear).
        if not isinstance(self.model, PeftModel):
            self.model = get_peft_model(self.model, self.lora)
            self.model.print_trainable_parameters()
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self._opt = opt
        return opt

    # -- optimizer state persistence (walltime resume) ----------------------
    def save_optimizer(self, path) -> bool:
        """Save AdamW moments next to the adapter; False if never created."""
        opt = getattr(self, "_opt", None)
        if opt is None:
            return False
        import torch
        torch.save(opt.state_dict(), Path(path) / "optimizer.pt")
        return True

    def load_optimizer(self, path, lr: float) -> bool:
        """Restore AdamW moments; False when no optimizer.pt exists."""
        src = Path(path) / "optimizer.pt"
        if not src.exists():
            return False
        import torch
        self._ensure_optimizer(lr).load_state_dict(torch.load(src, map_location="cpu"))
        return True

    def optimizer_lr(self) -> float | None:
        """lr of the first optimizer param group (None before the optimizer
        exists). Resume's `load_state_dict` restores the SAVED lr into
        param_groups, so this can differ from the --lr the caller passed (D8):
        the saved value wins, and main() warns about the mismatch."""
        opt = getattr(self, "_opt", None)
        if opt is None:
            return None
        groups = opt.param_groups
        return float(groups[0]["lr"]) if groups else None


    # -- one gradient step (F3: ONE opt.step per training step) --------------
    def grpo_step(self, batch_prompts, batch_completions, advantages,
                  lr=1e-5, beta_kl=0.02, clip=0.2, max_grad_norm=0.5,
                  micro_bs: int = 4, completion_ids=None, old_logprobs=None,
                  n_micro: int | None = None) -> StepResult:
        """One optimizer step per training step (F3): gradients accumulate over
        the micro-batches as loss/n_micro, then a SINGLE clip + AdamW step.

        F1 tripwire: a non-finite loss (any micro-batch) or a non-finite total
        grad norm aborts the update — opt.step() is SKIPPED, the event counted
        and reported in `first_nonfinite` so run_loop can save the still-finite
        weights and exit 3. Never skip silently.

        F4: per-token PPO ratio against the rollout-time old logprobs (stored at
        rollout, `old_logprobs`) and a per-token k3 KL to the FROZEN base,
        exp(r-s) - (r-s) - 1 >= 0 (the old exp(s)(s-r) was minimised at s=r-1
        and its gradient ∝e^s vanishes when s<<0 — the unbounded-push mechanism
        the audit traced). With the audit's chosen F3 variant (one step per
        training step) old == current logprobs, so the ratio is exactly 1 and
        the clip is wired-but-inert; it engages if a multi-epoch loop ever uses
        the same rollout batch for several updates.

        F8: scoring rows are contiguous [prompt|completion] padded only at the
        end (_pack_rows), so the completion starts at its own prompt length and
        positions match generation.
        """
        import torch
        opt = self._ensure_optimizer(lr)
        model, tok = self.model, self.tok

        n = len(batch_prompts)
        if n == 0:
            return StepResult(loss=0.0, n_nonfinite=0)
        if n_micro is None:
            n_micro = max(1, (n + micro_bs - 1) // micro_bs)
        opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        k_finite = 0
        loss_pg_acc = 0.0
        loss_kl_acc = 0.0
        policy_push_acc = 0.0  # B4: detached -(a * seq_logp).mean(), per micro
        max_abs_logratio = 0.0  # D6
        clip_frac_num = 0       # D6: scored tokens outside [1-clip, 1+clip]
        clip_frac_den = 0       # D6: scored tokens in finite micro-batches
        s_vals: list[float] = []
        ent_vals: list[float] = []
        n_nonfinite = 0
        first_bad: dict | None = None
        cur_micro_idx = 0
        m_stat: dict = {}
        grad_norm = lora_norm = None

        for i in range(0, n, micro_bs):
            ps = batch_prompts[i:i + micro_bs]
            cs = batch_completions[i:i + micro_bs]
            advs = advantages[i:i + micro_bs]
            row_ids = completion_ids[i:i + micro_bs] if completion_ids is not None else None
            row_old = old_logprobs[i:i + micro_bs] if old_logprobs is not None else None

            a = torch.tensor(advs, dtype=torch.float32, device=model.device)
            prompt_ids = [tok(p, add_special_tokens=True)["input_ids"] for p in ps]
            comp_ids: list[list[int]] = []
            for c, cids in zip(cs, row_ids if row_ids is not None else [None] * len(cs)):
                comp_ids.append(list(cids) if cids is not None
                                else tok(c, add_special_tokens=False)["input_ids"])
            full, attn, plens, clens = self._pack_rows(prompt_ids, comp_ids)

            model.eval()   # same dropout-free distribution the rollout scored
            logits = model(input_ids=full, attention_mask=attn).logits
            logp_all = torch.log_softmax(logits[:, :-1, :], dim=-1)
            tgt_all = full[:, 1:]
            token_logp = logp_all.gather(-1, tgt_all.unsqueeze(-1)).squeeze(-1)
            # per-row completion span in the contiguous layout: predictor
            # positions [plen-1, plen+clen-1)
            cmask = torch.zeros_like(token_logp)
            for r in range(token_logp.shape[0]):
                s = plens[r] - 1
                e = s + clens[r]
                cmask[r, s:e] = 1.0
            cmask_sum = cmask.sum(dim=1).clamp(min=1)
            seq_logp = (token_logp * cmask).sum(dim=1) / cmask_sum

            # F4: per-token PPO ratio against the rollout-time old logprobs.
            old_logp = torch.zeros_like(token_logp)
            for r in range(token_logp.shape[0]):
                s = plens[r] - 1
                e = s + clens[r]
                if row_old is not None and row_old[r] is not None \
                        and len(row_old[r]) == clens[r]:
                    old_logp[r, s:e] = torch.as_tensor(
                        row_old[r], dtype=token_logp.dtype, device=model.device)
                else:
                    old_logp[r, s:e] = token_logp[r, s:e].detach()   # ratio == 1
            logratio = token_logp - old_logp
            ratio = logratio.exp()
            clip_r = torch.clamp(ratio, 1 - clip, 1 + clip)
            # D6: per-token ratio statistics over SCORED tokens only. The mask
            # is zero outside the completion span — an unmasked pad position
            # carries old_logp=0 and a meaningless ratio. With ratio==1 exactly
            # these are 0/0%; on GPU they measure the bf16 noise floor between
            # the rollout-chunk forward and the micro-batch forward (F4a).
            n_scored = int(cmask_sum.sum())
            if n_scored:
                max_abs_logratio = max(
                    max_abs_logratio, float((logratio * cmask).abs().amax()))
                clipped = ((ratio < 1 - clip) | (ratio > 1 + clip)) * cmask
                clip_frac_num += int(clipped.sum())
                clip_frac_den += n_scored
            # Per-row mean of the per-token clipped surrogate, then mean over
            # rows. With ratio==1 this is -mean(a) with the old loss' per-token
            # gradient (-a_r / (n*m_r)); the clip engages only once the ratio
            # actually deviates (multi-epoch reuse of the batch).
            row_pg = (torch.min(ratio * a.unsqueeze(1), clip_r * a.unsqueeze(1))
                      * cmask).sum(dim=1) / cmask_sum
            loss_pg = -row_pg.mean()

            # KL to the FROZEN BASE reference (paper Eq. 25's pi_ref): base
            # weights never train (only LoRA adapters do), so the reference
            # forward must EXCLUDE the LoRA deltas. `peft.PeftModel.disable_adapter()`
            # (context manager; peft>=0.5) — NOT `model.get_base_model()(...)`:
            # the LoRA layers live INSIDE the base modules, so unwrapping still
            # runs adapters -> a post-LoRA reference would silently train
            # against the policy itself.
            with torch.no_grad():
                if not hasattr(model, "disable_adapter"):
                    raise RuntimeError(
                        "KL reference needs peft.PeftModel.disable_adapter() (peft>=0.5). "
                        f"Installed peft lacks it; upgrade the desktop venv.")
                with model.disable_adapter():
                    ref_logits = model(input_ids=full, attention_mask=attn).logits
            ref_logp_all = torch.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_token_logp = ref_logp_all.gather(-1, tgt_all.unsqueeze(-1)).squeeze(-1)
            # F4: per-token k3 KL, exp(r-s) - (r-s) - 1 >= 0, minimised at s=r;
            # gradient 1 - exp(r-s) never vanishes at extreme s (r = base logp,
            # s = policy logp). bf16's fp32 range keeps exp() from overflowing
            # where fp16 did.
            d = ref_token_logp - token_logp
            kl_per_token = d.exp() - d - 1.0
            loss_kl = ((kl_per_token * cmask).sum(dim=1) / cmask_sum).mean()

            loss = loss_pg + beta_kl * loss_kl
            cur_micro_idx = i // micro_bs
            m_stat = {
                "micro_idx": cur_micro_idx,
                "loss_pg": float(loss_pg.detach()),
                "loss_kl": float(loss_kl.detach()),
                "s_min": float(seq_logp.detach().min()),
                "max_abs_logit": float(logits.detach().abs().amax()),
            }
            if not torch.isfinite(loss):
                # F1 tripwire, loss side. Weights are still finite (no step has
                # been taken), so the pre-NaN checkpoint is saveable.
                n_nonfinite += 1
                if first_bad is None:
                    first_bad = dict(m_stat)
                break   # do not accumulate any further

            (loss / n_micro).backward()   # F3: accumulate; opt.step() once below
            total_loss += float(loss.detach())
            k_finite += 1
            loss_pg_acc += float(loss_pg.detach())
            loss_kl_acc += float(loss_kl.detach())
            # B4: detached policy-push diagnostic. With ratio==1, loss_pg is
            # -mean(advantages) and group normalization pins that to ~0 (the
            # raw field is unevaluable); this covariance term
            # -(a * seq_logp).mean() is the pairable push signal. no_grad via
            # .detach() on the scored sequence logps.
            policy_push_acc += float(-(a * seq_logp.detach()).mean())
            s_vals.extend(float(v) for v in seq_logp.detach().cpu().tolist())
            for r, plen in enumerate(plens):      # F7: first-token entropy
                if clens[r] == 0:
                    continue
                p = logp_all[r, plen - 1].exp()
                ent_vals.append(-(p * torch.log(p.clamp(min=1e-12))).sum().item())

        # F1 tripwire, gradient side: one clip for the whole accumulated step.
        if n_nonfinite == 0:
            params = [p for p in model.parameters() if p.requires_grad]
            lora_norm = float(sum(p.detach().pow(2).sum() for p in params).sqrt())
            gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if not torch.isfinite(gn):
                # loss was finite but the grads were inf/NaN (clip_grad_norm_
                # turns inf grads into inf*0 = NaN): same tripwire as a bad loss.
                n_nonfinite = 1
                if first_bad is None:
                    first_bad = dict(m_stat)
            else:
                opt.step()
                grad_norm = float(gn)
        opt.zero_grad(set_to_none=True)

        k = max(k_finite, 1)
        return StepResult(
            loss=(total_loss / k if k_finite else float("nan")),
            loss_pg=(loss_pg_acc / k if k_finite else None),
            loss_kl=(loss_kl_acc / k if k_finite else None),
            policy_push=(policy_push_acc / k if k_finite else None),
            max_abs_logratio=(max_abs_logratio if k_finite else None),
            clip_frac=(clip_frac_num / clip_frac_den if k_finite else None),
            s_min=min(s_vals) if s_vals else None,
            s_mean=(sum(s_vals) / len(s_vals)) if s_vals else None,
            frac_s_gt_m001=(sum(1 for v in s_vals if v > -0.01) / len(s_vals)) if s_vals else None,
            frac_s_lt_m10=(sum(1 for v in s_vals if v < -10) / len(s_vals)) if s_vals else None,
            entropy_first=(sum(ent_vals) / len(ent_vals)) if ent_vals else None,
            grad_norm=grad_norm,
            lora_norm=lora_norm,
            n_nonfinite=n_nonfinite,
            first_nonfinite=first_bad,
        )

    def check_finite(self) -> bool:
        """F13: True iff every trainable (LoRA) param is finite. save_checkpoint
        refuses to persist otherwise (a NaN checkpoint would silently resume a
        dead run)."""
        import torch
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return False
        return all(bool(torch.isfinite(p).all().item()) for p in params)


# --------------------------------------------------------------------------- #
# Rollout collection for one training step
# --------------------------------------------------------------------------- #


def warn_resume_lr_mismatch(cli_lr: float, saved_lr: float) -> str | None:
    """D8: the warning text when `--lr` on `--resume` differs from the SAVED
    optimizer lr (the saved `param_groups` value wins; the CLI value is
    silently overridden by `load_state_dict`). Returns None when they agree.

    Extracted from main() so the mismatch rule is unit-testable on the hub
    (main's resume path needs peft/torch)."""
    if abs(saved_lr - cli_lr) > 1e-12:
        return (f"WARNING: --lr {cli_lr} differs from the saved optimizer lr "
                f"{saved_lr}; the SAVED value wins on resume (param_groups) — "
                "use a fresh --out for a different-lr arm")
    return None


def collect_step(policy: PolicyLike, G: int, curriculum: str, horizon: int = 36,
                 seed: int | None = None, temperature: float = 1.0,
                 rollout_batch: int = 16, shared_demand: bool = True) -> StepRollout:
    """Run G episodes under `policy`; returns one StepRollout (flat per-decision).

    Rollouts are **batched**: the decisions of one week (4 roles) across all G
    episodes are causally independent — the engine builds each role's ctx from
    state at the start of the week and never exposes a same-week order to another
    agent (AUDIT2 leak fix) — so they are submitted together, chunked to at most
    `rollout_batch` prompts per call (one call per week when `rollout_batch` covers
    the week: 4·G prompts). For the default G=8 a week holds 32 prompts: 72 calls
    per step with the default `rollout_batch=16`, 36 with `rollout_batch<=0` —
    versus the previous 1,152 single-prompt calls (measured on the hub,
    2026-09-30). See `grpo_env.run_episodes_batched`
    for the invariance argument and the test that pins it against the engine.

    F17: all G episodes of a step share ONE demand path (common random numbers),
    so the per-(agent, week) z-score measures policy-sample variance, not demand
    luck (with constant actions the old code measured only demand luck). Pass
    `shared_demand=False` for the pre-fix per-episode draws.

    `rollout_batch=1` dispatches one prompt per call (the tiny-VRAM escape
    hatch); `rollout_batch<=0` puts every pending decision of a week in one call.

    Flat rows are ordered by episode then (t, agent_id) — length G * 4 * horizon —
    the same row order `grpo_step` consumes. `completion_ids`/`old_logprobs` are
    the F2/F4 sampled identities threaded through DecisionRecord.
    """
    rng = random.Random(seed)
    if shared_demand:
        demand = sample_curriculum_demand(rng, curriculum, horizon)
        demands = [demand] * G
    else:
        demands = [sample_curriculum_demand(rng, curriculum, horizon) for _ in range(G)]

    def batch_policy(prompts: list[str]):
        texts, ids, logps = policy.rollout(prompts, temperature=temperature)
        if ids is None:                       # hub stubs return (texts, None, None)
            ids = [None] * len(texts)
        if logps is None:
            logps = [None] * len(texts)
        return list(zip(texts, ids, logps))

    episodes = run_episodes_batched(batch_policy, demands,
                                    SimConfig(horizon=horizon),
                                    chunk=rollout_batch)
    rewards = [ep_result_to_rewards(e) for e in episodes]
    adv = group_normalize(rewards)
    prompts_all: list[str] = []
    completions_all: list[str] = []
    ids_all: list[list[int] | None] = []
    orders_all: list[int] = []
    logps_all: list[list[float] | None] = []
    flat_adv: list[float] = []
    for e, a in zip(episodes, adv):
        for rec in e.records:
            prompts_all.append(rec.prompt)
            completions_all.append(rec.completion)
            ids_all.append(rec.completion_ids)
            orders_all.append(rec.order)
            logps_all.append(rec.token_logprobs)
            flat_adv.append(a[rec.agent_id][rec.t])
    return StepRollout(prompts=prompts_all, completions=completions_all,
                       completion_ids=ids_all, orders=orders_all,
                       costs=[e.total_sys_cost for e in episodes],
                       advantages=flat_adv, old_logprobs=logps_all)


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
    ap.add_argument("--save-every", type=int, default=25,
                    help="checkpoint every N completed steps (0 = only on the "
                         "last step / SIGTERM)")
    ap.add_argument("--seed", type=int, default=None,
                    help="reproducibility seed for the loop RNG and torch (F12); "
                         "recorded in state.json. Default 12345 — the audit's "
                         "a11_seed2 was a replicate, not a seed variant, because "
                         "no seed ever reached the RNG")
    ap.add_argument("--out", default="results/grpo_a0")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


def _resolve_seed(args: argparse.Namespace) -> int:
    """Effective training seed: --seed when given, else the historical 12345."""
    return args.seed if args.seed is not None else 12345


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _git_stamp() -> dict:
    """HEAD + dirty flag for run_info.json (audit F6 + "first rerun must log":
    the sweep once ran pre-fix code with nothing in any log saying so)."""
    root = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        lines = [ln for ln in dirty.stdout.splitlines() if ln.strip()]
        return {
            "git_head": head.stdout.strip() or None,
            "git_dirty": bool(lines),
            "git_dirty_files": len(lines),
        }
    except Exception as exc:   # no git binary / not a repo -> say so, don't crash
        return {"git_head": None, "git_dirty": None, "git_error": str(exc)}


# --------------------------------------------------------------------------- #
# Walltime-safe checkpoint/resume (D3). Slurm sends SIGTERM before the kill
# grace period, so an 18h job survives walltime only if SIGTERM -> checkpoint ->
# clean exit. Every write is atomic (tmp + rename): a SIGKILL mid-write can
# never leave a half-written checkpoint that the next run would trust.
#
# What is restored on --resume:
#   * LoRA adapter + tokenizer  (torch path; `adapter_saved` in state.json)
#   * AdamW moments             (optimizer.pt, when the run had created them)
#   * step index and RNG state  (state.json, always — so the hub can dry-run it)
# The KL reference is the FROZEN base weights (adapters only are trained), so it
# carries no state; there are no other running buffers in the loop.
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def save_checkpoint(policy: PolicyLike, ckpt_path: Path, state_path: Path, *,
                    step: int, rng: random.Random, args: argparse.Namespace,
                    has_torch: Callable[[], bool] = _torch_available,
                    versioned: bool = True) -> dict:
    """Checkpoint the run: adapter dir (torch) first, then the resume pointer.

    F11: the adapter lands in a VERSIONED dir `checkpoint-<step>/` (no rmtree
    of a shared dir, so a walltime kill can never leave the previous checkpoint
    missing and a resume silently falls back to "step/RNG only"). The optimizer
    moments are written into the TMP dir BEFORE the rename (the old code wrote
    optimizer.pt after the rename — a half-written or missing optimizer.pt next
    to a completed rename). state.json is written LAST, atomically: a resume
    never points at a non-durable adapter.

    F13: refuse (raise) when any trainable param is non-finite — a NaN
    checkpoint would silently resume a dead run.

    Returns the payload written to state.json.
    """
    check = getattr(policy, "check_finite", None)
    if check is not None and not check():
        raise RuntimeError(
            f"refusing to checkpoint at step {step}: non-finite trainable params (F13)")
    adapter_saved = False
    optimizer_saved = False
    final_dir = ckpt_path if not versioned \
        else ckpt_path.with_name(f"{ckpt_path.name}-{step}")
    if has_torch():
        tmp = final_dir.with_name(final_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        policy.model.save_pretrained(tmp)  # type: ignore[attr-defined]
        tok = getattr(policy, "tok", None)
        if tok is not None:
            tok.save_pretrained(tmp)
        save_opt = getattr(policy, "save_optimizer", None)
        if callable(save_opt):
            optimizer_saved = bool(save_opt(tmp))   # BEFORE the rename (F11)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp, final_dir)
        adapter_saved = True
    payload = {
        "step": step,                        # last COMPLETED step; resume at step + 1
        "rng_state": list(rng.getstate()),   # json turns the inner tuple into a list
        "adapter_saved": adapter_saved,
        "optimizer_saved": optimizer_saved,
        "model": args.model,
        "group": args.group,
        "horizon": args.horizon,
        "curricula": args.curricula,
        "lr": args.lr,
        "beta_kl": args.beta_kl,
        "rollout_batch": args.rollout_batch,
        "seed": _resolve_seed(args),         # F12: effective seed, always recorded
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if adapter_saved:
        payload["checkpoint_dir"] = final_dir.name   # F11: how to find the adapter
    _atomic_write_json(state_path, payload)
    return payload


def load_resume_state(outdir: Path) -> tuple[int, dict | None]:
    """Resume pointer -> (next step to run, payload). (0, None) if no checkpoint."""
    path = outdir / "state.json"
    if not path.exists():
        return 0, None
    payload = json.loads(path.read_text())
    return int(payload["step"]) + 1, payload


def rng_from_state(payload: dict) -> random.Random:
    """Rebuild the loop RNG from a checkpoint payload (exact continuation)."""
    rng = random.Random()
    version, state, gauss_next = payload["rng_state"]
    rng.setstate((version, tuple(state), gauss_next))
    return rng


def sigterm_handler(stop: threading.Event) -> Callable[[int, object], None]:
    """SIGTERM -> set the stop event. The loop checkpoints and exits at the next
    step boundary (Slurm's grace period is minutes, not hours, so this is the
    difference between an 18h job resuming and an 18h job lost)."""
    def _handler(signum, frame):
        stop.set()
    return _handler


def run_loop(policy: PolicyLike, args: argparse.Namespace, log_path: Path,
             ckpt_path: Path, start_step: int = 0, rng: random.Random | None = None,
             has_torch: Callable[[], bool] = _torch_available,
             stop: threading.Event | None = None) -> None:
    """Core training loop: curriculum rotation -> collect_step -> grpo_step -> log.

    `has_torch` gates the adapter save so the hub (no torch) can exercise the
    full loop with a stub policy; tests inject a fake. A checkpoint (adapter +
    state.json) is written every `args.save_every` completed steps, on the last
    step, and immediately when `stop` is set (SIGTERM). `stop` is checked at step
    boundaries, so a kill mid-step costs at most that step.

    F1: on the FIRST non-finite step (any micro-batch's loss or the total grad
    norm), save `checkpoint-prenan/` (the weights are still finite — the update
    was skipped), append `{"event":"nonfinite",...}` to the log, mark
    `state.json` "aborted":"nonfinite", and exit 3. Abort, never skip silently.
    """
    rng = rng or random.Random(_resolve_seed(args))
    curricula = args.curricula.split(",")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_path.parent / "state.json"
    t0 = time.time()
    seen_nonfinite = False
    for step in range(start_step, args.steps):
        curriculum = curricula[step % len(curricula)]
        roll = collect_step(
            policy, G=args.group, curriculum=curriculum, horizon=args.horizon,
            seed=rng.randint(0, 2**31), rollout_batch=args.rollout_batch)
        out = policy.grpo_step(roll.prompts, roll.completions, roll.advantages,
                               lr=args.lr, beta_kl=args.beta_kl,
                               completion_ids=roll.completion_ids,
                               old_logprobs=roll.old_logprobs)
        if isinstance(out, float):          # legacy stub shield
            out = StepResult(loss=out)
        n = len(roll.advantages)
        orders = roll.orders
        om = sum(orders) / n if n else 0.0
        n_unique_orders = len(set(orders))
        n_unique_completions = len(set(roll.completions))
        parse_failures = sum(1 for c in roll.completions if not has_int(c))
        samples = [
            {"prompt_hash": hashlib.sha256(p.encode("utf-8")).hexdigest()[:12],
             "text": c, "order": o, "ids": i}
            for p, c, o, i in zip(roll.prompts, roll.completions, orders,
                                  roll.completion_ids)][:3]
        rec = {
            "step": step, "curriculum": curriculum, "loss": out.loss,
            "loss_pg": out.loss_pg, "loss_kl": out.loss_kl,
            "policy_push": out.policy_push,
            "s_min": out.s_min, "s_mean": out.s_mean,
            "frac_s_gt_m001": out.frac_s_gt_m001, "frac_s_lt_m10": out.frac_s_lt_m10,
            "entropy_first": out.entropy_first,
            "grad_norm": out.grad_norm, "lora_norm": out.lora_norm,
            "max_abs_logratio": out.max_abs_logratio,
            "clip_frac": out.clip_frac,
            "n_nonfinite": out.n_nonfinite,
            "mean_adv": sum(roll.advantages) / n if n else 0.0,
            "min_adv": min(roll.advantages) if n else 0.0,
            "max_adv": max(roll.advantages) if n else 0.0,
            "n_unique_orders": n_unique_orders,
            "n_unique_completions": n_unique_completions,
            "parse_failure_rate": (parse_failures / n) if n else 0.0,
            "order_mean": round(om, 3),
            "order_std": round((sum((o - om) ** 2 for o in orders) / n) ** 0.5, 3) if n else None,
            "order_max": max(orders) if n else None,
            "mean_episode_cost": (sum(roll.costs) / len(roll.costs)) if roll.costs else None,
            "samples": samples,
            "elapsed_s": round(time.time() - t0, 1),
        }
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if step % 10 == 0 or step == args.steps - 1:
            print(rec, flush=True)
        if out.n_nonfinite > 0 and not seen_nonfinite:
            seen_nonfinite = True
            event = {"event": "nonfinite", "step": step, **(out.first_nonfinite or {})}
            with log_path.open("a") as f:
                f.write(json.dumps(event) + "\n")
            prenan = ckpt_path.parent / "checkpoint-prenan"
            payload = save_checkpoint(policy, prenan, state_path, step=step,
                                      rng=rng, args=args, has_torch=has_torch,
                                      versioned=False)
            payload["aborted"] = "nonfinite"
            _atomic_write_json(state_path, payload)
            print(f"NON-FINITE at step {step} ({out.n_nonfinite} micro-batch(es)): "
                  f"weights saved to {prenan}, state.json aborted=nonfinite, "
                  f"exiting 3 (resume is refused)", flush=True)
            raise SystemExit(3)
        stopped = stop is not None and stop.is_set()
        due = (args.save_every > 0 and (step + 1) % args.save_every == 0) \
            or step == args.steps - 1
        if stopped or due:
            payload = save_checkpoint(policy, ckpt_path, state_path, step=step,
                                      rng=rng, args=args, has_torch=has_torch)
            if stopped:
                with log_path.open("a") as f:
                    f.write(json.dumps({"step": step, "event": "sigterm",
                                        "adapter_saved": payload["adapter_saved"]}) + "\n")
                print(f"SIGTERM: checkpointed at step {step}, exiting cleanly "
                      f"(resume with the same --out)", flush=True)
                break


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.steps <= 0:
        print("steps<=0: nothing to do")
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_log.jsonl"
    ckpt_path = outdir / "checkpoint"
    eff_seed = _resolve_seed(args)

    # SIGTERM = Slurm's walltime warning. Set the flag; the loop checkpoints and
    # exits at the next step boundary (SIGKILL after the grace period is then safe).
    stop = threading.Event()
    signal.signal(signal.SIGTERM, sigterm_handler(stop))

    policy = _Policy(args.model, seed=eff_seed)
    start_step, state, rng = 0, None, None
    resume_info: dict | None = None
    if args.resume:
        start_step, state = load_resume_state(outdir)
        if state is None:
            print(f"--resume: no {outdir / 'state.json'}; starting from step 0")
        else:
            if state.get("aborted") == "nonfinite":
                print(f"--resume refused: {outdir} aborted NON-FINITE at step "
                      f"{state['step']} (see train_log.jsonl for the event); "
                      "inspect checkpoint-prenan/ and start a fresh --out", flush=True)
                return 3
            rng = rng_from_state(state)
            if state.get("adapter_saved"):
                # F11: the adapter dir is versioned; state.json says where.
                ckpt_dir = outdir / state.get("checkpoint_dir", "checkpoint")
                if not ckpt_dir.exists():
                    print(f"FATAL: state.json says the adapter was saved but "
                          f"{ckpt_dir} is MISSING — refusing to resume "
                          "(F11: never continue from a silently-fresh LoRA)", flush=True)
                    return 1
                from peft import PeftModel
                base = policy.model.get_base_model() if hasattr(
                    policy.model, "get_base_model") else policy.model
                # F5: from_pretrained defaults to is_trainable=False -> the
                # resumed adapter is FROZEN and AdamW gets an empty param list.
                policy.model = PeftModel.from_pretrained(base, ckpt_dir,
                                                         is_trainable=True)
                n_trainable = sum(1 for p in policy.model.parameters()
                                  if p.requires_grad)
                print(f"resume drill: adapter {ckpt_dir} -> {n_trainable} "
                      f"trainable params", flush=True)
                if n_trainable <= 0:
                    print(f"FATAL: resumed adapter has zero trainable params (F5); "
                          "refusing to continue", flush=True)
                    return 1
                opt_loaded = policy.load_optimizer(ckpt_dir, args.lr)
                if opt_loaded:
                    saved_lr = policy.optimizer_lr()
                    if saved_lr is not None:
                        # D8: load_state_dict restores the saved param-group lr
                        # over the constructor value, so --lr is silently
                        # ignored on resume. Say so instead of mislabeling the
                        # run (a reused --out for a different-lr arm).
                        warning = warn_resume_lr_mismatch(args.lr, saved_lr)
                        if warning:
                            print(warning, flush=True)
                resume_info = {"adapter_loaded": True, "n_trainable": n_trainable,
                               "optimizer_loaded": opt_loaded, "step": start_step,
                               "checkpoint_dir": str(ckpt_dir)}
                print(f"resumed from step {start_step} (adapter + tokenizer"
                      f"{' + optimizer' if opt_loaded else ''})", flush=True)
            else:
                print(f"resumed step counter from {start_step} (no adapter on disk; "
                      "step/RNG only)", flush=True)
                resume_info = {"adapter_loaded": False, "step": start_step}
            # Append-only log hazard (§3.7): records for steps that will RE-RUN
            # must be dropped, else the jsonl carries two versions of each step.
            if log_path.exists():
                try:
                    lines = log_path.read_text().splitlines()
                    keep = [ln for ln in lines
                            if int(json.loads(ln).get("step", -1)) < start_step]
                    if len(keep) != len(lines):
                        tmp = log_path.with_name(log_path.name + ".tmp")
                        tmp.write_text("\n".join(keep) + ("\n" if keep else ""))
                        os.replace(tmp, log_path)
                        print(f"log trimmed: dropped {len(lines) - len(keep)} stale "
                              f"record(s) for steps >= {start_step}", flush=True)
                except (json.JSONDecodeError, ValueError, OSError) as exc:
                    print(f"WARN: could not trim {log_path}: {exc}", flush=True)

    run_info = {
        **_git_stamp(),                          # F6: HEAD + dirty flag in the log
        "seed": eff_seed,                        # F12: the torch + loop seed
        "resume": resume_info,                   # "First rerun must log": the drill
        "model": args.model,
        "group": args.group, "horizon": args.horizon,
        "curricula": args.curricula, "lr": args.lr, "beta_kl": args.beta_kl,
        "rollout_batch": args.rollout_batch,
        "compile_knobs": getattr(policy, "compile_knobs", None),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _atomic_write_json(outdir / "run_info.json", run_info)
    print("run_info:", json.dumps(run_info, sort_keys=True), flush=True)

    run_loop(policy, args, log_path, ckpt_path, start_step=start_step, rng=rng,
             stop=stop)
    print("done:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())