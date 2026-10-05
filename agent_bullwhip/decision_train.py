"""Wave-7 decision-native training: soft-target SFT then an RLCD refinement arm.

WAVE7_SPEC.md D4 + WAVE7-AUDIT-1 fixes (binding fix-list, re-read at P3 freeze):

- LoRA r=8 / alpha=16, attention projections ONLY (q/k/v/o_proj), base frozen
  (peft; cluster pin peft==0.21.1 — LoraConfig API used here is stable across
  0.21.1 and current).
- Stage 1: soft-target cross-entropy over label logits. Default data
  augmentation: PERMUTED OPTION ORDERS (audit P0 #1) — each sample renders the
  option grid under a per-(epoch,row) permutation and permutes the target the
  same way, so letter priors (A/H/L) cannot be a shortcut; the permutation
  rng derives from (seed, epoch, row), deterministic across resume.
- Stage 2 (RLCD arm): exact expected-cost policy gradient over ALL 21 actions
  (audit P1: K-of-21 sampling is the wrong estimator — exact scoring has zero
  sampling variance; 13-220 ms/state), loss = sum_a pi(a) * J(a)/std_a(J) +
  beta_kl * KL(pi || pi_sft), with pi_sft frozen per state (computed once from
  the SFT adapter). Sign: minimize cost (audit P1: 'advantage = J - mean'
  ascends cost). `--rlcd-sample-k K` restores the public-recipe sampled arm
  (REINFORCE with LOO baseline excluding self, per-state std normalization) for
  R1 comparison; K=0 (default) = exact scoring.
- `--labels` = a decision_labeler jsonl (states already carry CRN E[J]; the
  on-policy relabel loop is the labeler's job — DAgger-style round 2, audit
  P2 fix 8).
- Checkpoint/resume/logging mirror grpo_train.py (versioned adapter dir,
  optimizer saved pre-rename, state.json written LAST atomically, F13
  non-finite refusal, SIGTERM -> checkpoint at next boundary). Data cursor:
  batch composition is derived from (seed, epoch) only, so a resume after
  walltime cannot skip or re-see rows (audit §4 P2: grpo's torch-RNG/data-
  cursor gap) — no dataloader cursor to checkpoint. RLCD sampling uses a
  torch.Generator seeded from (seed, step), also resume-exact.

The step loop is torch-free (PolicyLike stub, like grpo_train), so the hub
exercises the full loop with a stub; torch imports stay inside `_Policy`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .decision_labeler import _seeded_rng
from .decision_readout import LABELS, ORDER_GRID, PINNED_REVISION, render_prompt
from .engine import ROLES


# --------------------------------------------------------------------------- #
# Data loading + cursor (deterministic, resume-safe)
# --------------------------------------------------------------------------- #
def load_labels(path: str | Path) -> tuple[list[dict], str]:
    """Rows from a decision_labeler jsonl + a SHA-256 of the raw bytes (the
    checkpoint records it so a resume can spot a changed label file)."""
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    for r in rows:
        for key in ("role", "week", "ctx", "p"):
            if key not in r:
                raise ValueError(f"labels row missing {key!r} (bad labeler output?)")
    return rows, hashlib.sha256(raw).hexdigest()


def train_val_split(rows: list[dict], val_frac: float) -> list[int]:
    """Deterministic split: every ~1/val_frac-th row (by index) is validation."""
    if val_frac <= 0:
        return []
    stride = max(1, int(round(1 / val_frac)))
    return list(range(0, len(rows), stride))



def permute_grid_and_target(p: list[float], rng: random.Random
                            ) -> tuple[list[int], list[float]]:
    """Option-permutation augmentation (audit P0 #1): render label i with value
    ORDER_GRID[perm[i]] and target label i = p[perm[i]]."""
    perm = list(range(len(ORDER_GRID)))
    rng.shuffle(perm)
    target = [p[i] for i in perm]
    return perm, target


# --------------------------------------------------------------------------- #
# Policy contract (hub stub runs the whole loop without torch)
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    loss: float = float("nan")
    loss_kl: float | None = None
    grad_norm: float | None = None
    n_nonfinite: int = 0
    mean_target_entropy: float | None = None   # nats over the batch targets
    train_top1: float | None = None            # fraction argmax(pi) == argmin(J)
    # full-vocab probability mass on the 21 option letters (WAVE7_SPEC A1:
    # the untrained-readout >= 0.99 invariant erodes under SFT; logged, not
    # asserted, for trained adapters)
    min_label_mass: float | None = None
    mean_label_mass: float | None = None


class DecisionPolicy(Protocol):
    model: object
    tok: object

    def sft_step(self, rows: list[dict], targets: list[list[float]],
                 lr: float, seed: int, epoch: int, step: int) -> StepResult: ...
    def rlcd_step(self, rows: list[dict], j_vals: list[list[float]],
                  lr: float, beta_kl: float, sample_k: int, seed: int,
                  step: int) -> StepResult: ...
    def evaluate(self, rows: list[dict]) -> dict: ...   # val: nll/top1/regret
    def check_finite(self) -> bool: ...
    def save_optimizer(self, path: Path) -> bool: ...
    def load_optimizer(self, path: Path, lr: float) -> bool: ...


# --------------------------------------------------------------------------- #
# Checkpoint / resume / logging (grpo_train.py shape)
# --------------------------------------------------------------------------- #
def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def save_checkpoint(policy: DecisionPolicy, ckpt_path: Path, state_path: Path,
                    *, step: int, seed: int, args: argparse.Namespace,
                    labels_sha: str, phase: str,
                    has_torch: Callable[[], bool] | None = None) -> dict:
    """Adapter dir (torch) first, optimizer pre-rename, state.json LAST and
    atomically (grpo_train.save_checkpoint F11/F13 shape)."""
    has_torch = has_torch or _torch_available
    check = getattr(policy, "check_finite", None)
    if check is not None and not check():
        raise RuntimeError(f"refusing to checkpoint at step {step}: "
                           f"non-finite trainable params (F13)")
    adapter_saved = optimizer_saved = False
    final_dir = ckpt_path.with_name(f"{ckpt_path.name}-{step}")
    if has_torch():
        tmp = final_dir.with_name(final_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        policy.model.save_pretrained(tmp)
        tok = getattr(policy, "tok", None)
        if tok is not None:
            tok.save_pretrained(tmp)
        save_opt = getattr(policy, "save_optimizer", None)
        if callable(save_opt):
            optimizer_saved = bool(save_opt(tmp))
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp, final_dir)
        adapter_saved = True
    payload = {
        "step": step,                    # last completed step; resume at step+1
        "seed": seed,
        "phase": phase,
        "labels_sha": labels_sha,
        "adapter_saved": adapter_saved,
        "optimizer_saved": optimizer_saved,
        "model": args.model,
        "lr": args.lr,
        "batch": args.batch,
        "epochs": getattr(args, "epochs", None),
        "permute_options": getattr(args, "permute_options", None),
        "beta_kl": getattr(args, "beta_kl", None),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if adapter_saved:
        payload["checkpoint_dir"] = final_dir.name
    _atomic_write_json(state_path, payload)
    return payload


def pi_sft_key(row: dict) -> str:
    """Stable identity of a label row across processes (not id())."""
    return f"{row['episode']}:{row['week']}:{row['role']}"


def prepare_pi_sft(policy, rows: list[dict], outdir: Path, *, labels_sha: str,
                   batch: int, resuming: bool, log_path: Path
                   ) -> dict[str, list[float]]:
    """The RLCD KL anchor, frozen (AUDIT-2 P1-4). Fresh start: computed once
    from the policy as loaded (= the SFT adapter, before any update) and
    persisted to outdir/pi_sft.json with the labels sha. Resume: loaded from
    that file — rebuilding it from the live (RLCD-trained) adapter would make
    the anchor follow the policy, so a missing or mismatched file is fatal."""
    path = outdir / "pi_sft.json"
    if resuming:
        if not path.exists():
            raise SystemExit(f"RLCD resume: {path} missing — the frozen pi_sft "
                             f"anchor cannot be rebuilt from the trained "
                             f"adapter; refusing to resume")
        saved = json.loads(path.read_text())
        if saved.get("labels_sha") != labels_sha:
            raise SystemExit(f"RLCD resume: {path} labels_sha "
                             f"{str(saved.get('labels_sha'))[:12]} != "
                             f"{labels_sha[:12]} — refusing to resume")
        table = saved["pi"]
        missing = [pi_sft_key(r) for r in rows if pi_sft_key(r) not in table]
        if missing:
            raise SystemExit(f"RLCD resume: {path} lacks {len(missing)} rows "
                             f"(e.g. {missing[0]}) — refusing to resume")
        source = "loaded"
    else:
        table = policy.compute_pi_sft(rows, batch)
        _atomic_write_json(path, {"labels_sha": labels_sha, "n": len(table),
                                  "pi": table})
        source = "computed"
    with log_path.open("a") as f:
        f.write(json.dumps({"event": "pi_sft", "source": source,
                            "n": len(table), "labels_sha": labels_sha},
                           sort_keys=True) + "\n")
    return table


def load_resume_state(outdir: Path) -> tuple[int, dict | None]:
    path = outdir / "state.json"
    if not path.exists():
        return 0, None
    payload = json.loads(path.read_text())
    return int(payload["step"]) + 1, payload


def sigterm_handler(stop: threading.Event) -> Callable[[int, object], None]:
    def _handler(signum, frame):
        stop.set()
    return _handler


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _git_stamp() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        import subprocess
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        lines = [ln for ln in dirty.stdout.splitlines() if ln.strip()]
        return {"git_head": head.stdout.strip() or None,
                "git_dirty": bool(lines), "git_dirty_files": len(lines)}
    except Exception as exc:  # noqa: BLE001
        return {"git_head": None, "git_dirty": None, "git_error": str(exc)}


# --------------------------------------------------------------------------- #
# The step loop (torch-free; drives any DecisionPolicy)
# --------------------------------------------------------------------------- #
def run_loop(policy: DecisionPolicy, args: argparse.Namespace, log_path: Path,
             ckpt_path: Path, *, start_step: int = 0, labels_sha: str = "",
             stop: threading.Event | None = None,
             has_torch: Callable[[], bool] = _torch_available) -> None:
    """SFT or RLCD over the fixed label set, checkpointing every save-every
    steps, on the last step, and on SIGTERM (checked at step boundaries)."""
    stop = stop or threading.Event()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = ckpt_path.parent / "state.json"
    rows, _ = load_labels(args.labels)
    val_idx = train_val_split(rows, args.val_frac)
    val_rows = [rows[i] for i in val_idx]
    train_rows = [r for i, r in enumerate(rows) if i not in set(val_idx)]
    n_batches = (len(train_rows) + args.batch - 1) // args.batch
    seed = args.seed if args.seed is not None else 12345
    t0 = time.time()
    epochs_done = start_step // n_batches

    def _log(rec: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    if start_step == 0:
        _log({"event": "start", "phase": args.phase, "seed": seed,
              "n_train": len(train_rows), "n_val": len(val_rows),
              "n_batches": n_batches, "labels_sha": labels_sha,
              "git": _git_stamp()})

    for step in range(start_step, args.steps):
        epoch = step // n_batches
        rng = _seeded_rng(f"{seed}:sft-epoch:{epoch}")
        perm = rng.sample(range(len(train_rows)), len(train_rows))
        b_idx = perm[step % n_batches * args.batch:
                     (step % n_batches + 1) * args.batch]
        batch_rows = [train_rows[i] for i in b_idx]
        targets = [r["p"] for r in batch_rows]
        if args.phase == "rlcd":
            j_vals = [r["J"] for r in batch_rows]
            out = policy.rlcd_step(batch_rows, j_vals, lr=args.lr,
                                   beta_kl=args.beta_kl,
                                   sample_k=args.rlcd_sample_k,
                                   seed=seed, step=step)
        else:
            out = policy.sft_step(batch_rows, targets, lr=args.lr,
                                  seed=seed, epoch=epoch, step=step)
        rec = {"step": step, "phase": args.phase, "epoch": epoch,
               "loss": out.loss, "wall": round(time.time() - t0, 1)}
        for k in ("loss_kl", "grad_norm", "mean_target_entropy", "train_top1",
                  "min_label_mass", "mean_label_mass"):
            v = getattr(out, k)
            if v is not None:
                rec[k] = round(v, 6)
        _log(rec)
        if out.n_nonfinite:
            # F1-style abort: save the still-finite weights, mark abort, exit 3
            try:
                pay = save_checkpoint(
                    policy, ckpt_path.parent / "checkpoint-prenan", state_path,
                    step=step, seed=seed, args=args, labels_sha=labels_sha,
                    phase=args.phase)
                pay["aborted"] = "nonfinite"
                _atomic_write_json(state_path, pay)
            except Exception as exc:  # noqa: BLE001
                _log({"event": "nonfinite-save-failed", "step": step,
                      "error": str(exc)})
            _log({"event": "nonfinite", "step": step})
            raise SystemExit(3)
        do_save = (args.save_every > 0 and (step + 1) % args.save_every == 0) \
            or step == args.steps - 1 or stop.is_set()
        if do_save:
            pay = save_checkpoint(policy, ckpt_path, state_path, step=step,
                                  seed=seed, args=args, labels_sha=labels_sha,
                                  phase=args.phase, has_torch=has_torch)
            if stop.is_set():
                _log({"event": "sigterm", "step": step})
                return
        # validation at each epoch boundary
        if (epoch + 1) * n_batches - 1 == step and val_rows:
            v = policy.evaluate(val_rows)
            rec = {"event": "val", "epoch": epoch, "step": step,
                   "n_val": len(val_rows)}
            for k, val in v.items():
                rec[k] = round(val, 6) if isinstance(val, float) else val
            _log(rec)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="agent_bullwhip.decision_train")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=PINNED_REVISION,
                    help="hub snapshot pin (AUDIT-1 3a P2); '' = unpinned")
    ap.add_argument("--labels", required=True,
                    help="decision_labeler jsonl (rows carry ctx + p + J)")
    ap.add_argument("--phase", choices=["sft", "rlcd"], default="sft")
    ap.add_argument("--base-adapter", default=None,
                    help="SFT checkpoint dir to start RLCD from (required for "
                         "--phase rlcd)")
    ap.add_argument("--steps", type=int, default=120,
                    help="total optimizer batches (sft or rlcd)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta-kl", type=float, default=0.02,
                    help="RLCD KL(pi||pi_sft) anchor weight")
    ap.add_argument("--rlcd-sample-k", type=int, default=0,
                    help="0 = exact all-21 expected-cost PG (default); K>0 = "
                         "public-recipe sampled REINFORCE with LOO baseline")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--permute-options", type=int, default=1,
                    help="option-permutation augmentation (audit P0 #1); 0=off")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="results/wave7_sft")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="recompute activations instead of retaining them "
                         "(needed to fit batch 32 on a 22 GiB L4)")
    args = ap.parse_args(argv)
    if args.phase == "rlcd" and not args.base_adapter:
        ap.error("--phase rlcd requires --base-adapter (an SFT checkpoint dir)")
    if args.revision == "":
        args.revision = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "train_log.jsonl"
    ckpt_path = outdir / "checkpoint"
    rows, labels_sha = load_labels(args.labels)

    if args.resume:
        start_step, payload = load_resume_state(outdir)
        if payload is not None and payload.get("labels_sha") != labels_sha:
            raise SystemExit(
                f"labels hash changed since checkpoint ({payload.get('labels_sha')[:12]} "
                f"vs {labels_sha[:12]}) — refusing to resume on different data")
        seed = payload.get("seed") if payload is not None else \
            (args.seed if args.seed is not None else 12345)
        if payload is not None and payload.get("phase") != args.phase:
            raise SystemExit("resume phase mismatch")
    else:
        start_step = 0
        seed = args.seed if args.seed is not None else 12345

    stop = threading.Event()
    signal.signal(signal.SIGTERM, sigterm_handler(stop))

    adapter_dir = None
    if args.resume and payload is not None and payload.get("checkpoint_dir"):
        adapter_dir = ckpt_path.with_name(payload["checkpoint_dir"]).__str__()
    policy = _Policy(args.model, device=args.device, dtype=args.dtype,
                     revision=args.revision, phase=args.phase,
                     base_adapter=args.base_adapter, adapter_dir=adapter_dir,
                     max_length=args.max_length, seed=seed,
                     permute_options=bool(args.permute_options),
                     grad_ckpt=bool(args.grad_ckpt))
    if adapter_dir:
        opt = Path(adapter_dir) / "optimizer.pt"
        if opt.exists():
            policy.load_optimizer(opt, args.lr)
    if args.phase == "rlcd" and args.beta_kl > 0:
        resuming = args.resume and payload is not None
        policy.set_pi_sft(prepare_pi_sft(policy, rows, outdir,
                                         labels_sha=labels_sha,
                                         batch=args.batch, resuming=resuming,
                                         log_path=log_path))
    with log_path.open("a") as f:
        f.write(json.dumps({"event": "resume" if args.resume else "begin",
                            "phase": args.phase, "seed": seed,
                            "start_step": start_step,
                            "labels_sha": labels_sha,
                            "git": _git_stamp()}, sort_keys=True) + "\n")
    run_loop(policy, args, log_path, ckpt_path, start_step=start_step,
             labels_sha=labels_sha, stop=stop)
    # AUDIT-2 P1-1: a SIGTERM'd run checkpoints and returns early — it must
    # not look complete to the sbatch (which only stops on non-zero). 75 =
    # EX_TEMPFAIL: resubmit to continue.
    _, final = load_resume_state(outdir)
    if final is None or int(final["step"]) < args.steps - 1:
        return 75
    return 0


# --------------------------------------------------------------------------- #
# Real policy (torch + peft — lazy imports)
# --------------------------------------------------------------------------- #
def _enable_grad_checkpointing(model) -> None:
    """Recompute activations instead of retaining all layers' worth.

    Batch 32 x ~700-token prompts keeps ~21 GiB of activations alive on a
    0.6B model: the L4 target (22 GiB) OOMs in the first SFT step without
    this. `use_reentrant=False` is required so the frozen base still feeds
    gradients to the first checkpointed block; `enable_input_require_grads`
    makes the embedding outputs require grad for the same reason.
    """
    try:
        model.enable_input_require_grads()
    except AttributeError:
        pass
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:      # older transformers: no kwargs passthrough
        model.gradient_checkpointing_enable()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = False


def _attach_adapter(model, *, phase: str, base_adapter: str | None,
                    adapter_dir: str | None):
    """Exactly ONE adapter on the base (AUDIT-2 P1-3): the resume checkpoint,
    else the SFT anchor (rlcd), else a fresh LoRA (sft). Stacking
    PeftModel(PeftModel(base, sft), rlcd_ckpt) re-initialises the shared
    `default` adapter (lora_B == 0 -> the base model) and saves checkpoints
    with a nested key prefix that later load as the base. RLCD checkpoints
    already carry the SFT-initialised adapter, trained in place."""
    from peft import LoraConfig, PeftModel, get_peft_model
    if adapter_dir is not None:
        return PeftModel.from_pretrained(model, adapter_dir)
    if phase == "rlcd":
        return PeftModel.from_pretrained(model, base_adapter)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj",
                                     "o_proj"],   # attention only
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


class _Policy:
    def __init__(self, model_name: str, *, device: str | None = None,
                 dtype: str | None = None, revision: str | None = None,
                 phase: str = "sft", base_adapter: str | None = None,
                 adapter_dir: str | None = None, max_length: int = 1024,
                 seed: int = 12345, permute_options: bool = True,
                 grad_ckpt: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.phase = phase
        self.seed = seed
        self.max_length = max_length
        self._permute_options = permute_options
        self._pi_sft: dict[str, list[float]] | None = None   # set_pi_sft()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = "float32" if self.device == "cpu" else "bfloat16"
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # compile-path guard
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.manual_seed(seed)
        load_kwargs = {"local_files_only": True, "revision": revision,
                       "torch_dtype": getattr(torch, dtype),
                       "attn_implementation": "sdpa"}
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 local_files_only=True,
                                                 revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model = _attach_adapter(self.model, phase=phase,
                                     base_adapter=base_adapter,
                                     adapter_dir=adapter_dir)
        # peft loads from_pretrained adapters FROZEN — make the LoRA params
        # trainable for every resume/RLCD path (the base stays frozen; this is
        # idempotent for the fresh get_peft_model path). Without it a resumed
        # run would hit "optimizer got an empty parameter list".
        for name, p in self.model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
        self.model.to(self.device)
        self.model.train()          # LoRA dropout active during training
        if grad_ckpt:
            _enable_grad_checkpointing(self.model)
        from .decision_readout import resolve_label_ids
        sample = render_prompt({"role": "retailer", "t": 0, "on_hand": 0,
                                "backlog": 0, "outstanding": 0,
                                "incoming_last": 4, "incoming_now": 4,
                                "last_orders": []}, self.tok)
        self.label_ids = resolve_label_ids(self.tok, sample)
        self.optimizer = None
        self._use_logits_to_keep = self._probe_logits_to_keep()

    # -- helpers ------------------------------------------------------------ #
    def _probe_logits_to_keep(self) -> bool:
        try:
            self.model.eval()
            inp = self.tok([self.tok.eos_token or " "], return_tensors="pt",
                           padding=True).to(self.device)
            with self._torch.no_grad():
                out = self.model(**inp, use_cache=False, logits_to_keep=1)
            self.model.train()
            return tuple(out.logits.shape)[-2] == 1
        except Exception:  # noqa: BLE001 - version quirk -> logits[:, -1]
            self.model.train()
            return False

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.model.parameters())

    def _label_logits(self, inputs, *, with_mass: bool = False) -> "object":
        """Label-letter logits [B, 21] at the last position. `with_mass=True`
        also returns the detached full-vocab mass on those letters [B]
        (the quantity readout_batch asserts >= 0.99 on the untrained base)."""
        fwd = {"use_cache": False}
        if self._use_logits_to_keep:
            fwd["logits_to_keep"] = 1
        logits = self.model(**inputs, **fwd).logits
        if self._use_logits_to_keep:
            last = logits[:, 0, :].float()
        else:
            last = logits[:, -1, :].float()
        cand = last[:, list(self.label_ids.values())]
        if not with_mass:
            return cand
        with self._torch.no_grad():
            mass = (cand.logsumexp(-1) - last.logsumexp(-1)).exp()
        return cand, mass

    # -- optimizer ---------------------------------------------------------- #
    def _ensure_optimizer(self, lr: float):
        if self.optimizer is None:
            self.optimizer = self._torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr, betas=(0.9, 0.999), eps=1e-8)

    def save_optimizer(self, path: Path) -> bool:
        if self.optimizer is None:
            return False
        self._torch.save(self.optimizer.state_dict(), str(path / "optimizer.pt"))
        return True

    def load_optimizer(self, path: Path, lr: float) -> bool:
        self._ensure_optimizer(lr)
        sd = self._torch.load(str(path), map_location=self.device)
        self.optimizer.load_state_dict(sd)
        return True

    def check_finite(self) -> bool:
        import math
        for p in self.model.parameters():
            if p.requires_grad and not self._torch.isfinite(p).all():
                return False
        return True

    # -- stage 1: soft-target CE -------------------------------------------- #
    def sft_step(self, rows: list[dict], targets: list[list[float]],
                 lr: float, seed: int, epoch: int, step: int) -> StepResult:
        import torch
        if len(rows) != len(targets):
            raise ValueError("rows/targets length mismatch")
        # per-step torch seeding: LoRA dropout draws depend only on
        # (seed, epoch, step), so a resumed run reproduces every forward
        # EXACTLY (the audit's "torch RNG not checkpointed" gap closed by
        # derivation, which needs no state file entry).
        torch.manual_seed(int(hashlib.sha256(
            f"{seed}:sft:{epoch}:{step}".encode()).hexdigest()[:8], 16))
        # render with per-sample option permutation (deterministic per
        # (seed, epoch, row-identity) — the audit P0 #1 augmentation)
        prompts, tgt = [], []
        for r, t in zip(rows, targets):
            if self._permute_options:
                grid_rng = _seeded_rng(
                    f"{seed}:perm:{epoch}:{r.get('episode')}:{r.get('week')}:"
                    f"{r.get('role')}")
                perm, target = permute_grid_and_target(t, grid_rng)
                prompts.append(render_prompt(r["ctx"], self.tok,
                                             grid=[ORDER_GRID[i] for i in perm]))
                tgt.append(target)
            else:
                prompts.append(render_prompt(r["ctx"], self.tok))
                tgt.append(t)
        inputs = self._tokenize(prompts)
        logits, mass = self._label_logits(inputs, with_mass=True)
        probs = logits.softmax(-1)
        t = torch.tensor(tgt, dtype=torch.float32, device=self.device)
        loss = -(t * (probs + 1e-9).log()).sum(-1).mean()
        self._ensure_optimizer(lr)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        with torch.no_grad():
            top1 = (probs.argmax(-1) == t.argmax(-1)).float().mean().item()
        return StepResult(loss=float(loss.detach()),
                          grad_norm=float(gn),
                          mean_target_entropy=float(
                              -(t * (t + 1e-9).log()).sum(-1).mean().detach()),
                          train_top1=top1,
                          min_label_mass=float(mass.min()),
                          mean_label_mass=float(mass.mean()))

    def _tokenize(self, prompts: list[str]):
        inputs = self.tok(prompts, return_tensors="pt", padding=True,
                          truncation=False)
        if self.max_length and inputs.input_ids.shape[1] > self.max_length:
            raise ValueError(f"prompt of {inputs.input_ids.shape[1]} tokens > "
                             f"max_length={self.max_length} — refusing to "
                             f"silently truncate a decision prompt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    # -- stage 2: RLCD ------------------------------------------------------ #
    def compute_pi_sft(self, rows: list[dict], batch: int
                       ) -> dict[str, list[float]]:
        """pi_sft for EVERY row, once, from the adapter as loaded (the SFT
        anchor at a fresh RLCD start): eval mode (no dropout), no_grad,
        chunked by `batch`. Keyed by pi_sft_key (AUDIT-2 P1-4)."""
        table: dict[str, list[float]] = {}
        self.model.eval()
        try:
            with self._torch.no_grad():
                for lo in range(0, len(rows), max(1, batch)):
                    chunk = rows[lo:lo + max(1, batch)]
                    prompts = [render_prompt(r["ctx"], self.tok) for r in chunk]
                    probs = self._label_logits(self._tokenize(prompts)
                                               ).softmax(-1).tolist()
                    for r, p in zip(chunk, probs):
                        k = pi_sft_key(r)
                        if k in table:
                            raise ValueError(f"duplicate label row {k} — "
                                             f"pi_sft key must be unique")
                        table[k] = p
        finally:
            self.model.train()
        return table

    def set_pi_sft(self, table: dict[str, list[float]]) -> None:
        self._pi_sft = table

    def _sft_ref_logprobs(self, rows: list[dict]) -> list[list[float]]:
        """Frozen pi_sft per state: a lookup in the table built once at RLCD
        start (or loaded from pi_sft.json on resume) — never the live policy."""
        if self._pi_sft is None:
            raise RuntimeError("pi_sft reference not set — call set_pi_sft() "
                               "(main() does this before RLCD step 0)")
        return [self._pi_sft[pi_sft_key(r)] for r in rows]

    def rlcd_step(self, rows: list[dict], j_vals: list[list[float]],
                  lr: float, beta_kl: float, sample_k: int, seed: int,
                  step: int) -> StepResult:
        import torch
        if self.phase != "rlcd":
            raise ValueError("rlcd_step called in sft phase")
        prompts = [render_prompt(r["ctx"], self.tok) for r in rows]
        logits = self._label_logits(self._tokenize(prompts))
        log_probs = logits.log_softmax(-1)
        probs = log_probs.exp()
        j = torch.tensor(j_vals, dtype=torch.float32, device=self.device)
        jstd = j.std(-1, keepdim=True).clamp_min(1e-3)
        jn = (j - j.mean(-1, keepdim=True)) / jstd   # per-state normalize
        if sample_k > 0:
            # public-recipe arm: sample K actions, LOO baseline excluding self;
            # adv = -(J_k - mean_{j!=k} J_j)/std  (sign: J is a cost)
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(hashlib.sha256(f"{seed}:rlcd:{step}".encode())
                                .hexdigest()[:16], 16) % (2**32))
            act = torch.multinomial(probs, sample_k, replacement=True,
                                    generator=gen)          # [B, K]
            picked = jn.gather(-1, act)                      # normalized J
            b = (jn.sum(-1, keepdim=True) - picked) / (jn.shape[-1] - 1)  # LOO
            adv = -(picked - b)                              # already /std
            pg = -(adv * log_probs.gather(-1, act)).mean()
        else:
            # exact expected-cost gradient: loss = sum_a pi(a) J(a)/std (J const)
            pg = (probs * jn).sum(-1).mean()
        kl = torch.tensor(0.0, device=self.device)
        if beta_kl > 0:
            ref = torch.tensor(self._sft_ref_logprobs(rows),
                               dtype=torch.float32, device=self.device)
            ref = (ref + 1e-9)
            ref = ref / ref.sum(-1, keepdim=True)
            kl = (probs * (log_probs - ref.log())).sum(-1).mean()
        loss = pg + beta_kl * kl
        self._ensure_optimizer(lr)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        return StepResult(loss=float(loss.detach()), loss_kl=float(kl.detach()),
                          grad_norm=float(gn))

    # -- validation --------------------------------------------------------- #
    def evaluate(self, rows: list[dict]) -> dict:
        import torch
        prompts = [render_prompt(r["ctx"], self.tok) for r in rows]
        self.model.eval()
        try:
            logits = self._label_logits(self._tokenize(prompts))
            probs = logits.softmax(-1)
            t = torch.tensor([r["p"] for r in rows], dtype=torch.float32,
                             device=self.device)
            nll = float(-(t * (probs + 1e-9).log()).sum(-1).mean().detach())
            top1 = float((probs.argmax(-1) == t.argmax(-1)).float().mean().item())
            # grid regret from the labels' J vectors (no extra forwards)
            regs, tots = [], 0.0
            for r, p in zip(rows, probs.tolist()):
                b = max(range(len(ORDER_GRID)), key=p.__getitem__)
                regs.append(r["J"][b] - r["J"][r["argmin_idx"]])
            return {"nll": nll, "top1": top1,
                    "mean_regret": sum(regs) / len(regs) if regs else float("nan")}
        finally:
            self.model.train()


if __name__ == "__main__":
    raise SystemExit(main())

