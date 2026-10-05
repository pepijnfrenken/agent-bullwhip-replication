"""Decision-native readout: one forward pass, letter-logit answer over the order grid.

Wave-7 lane (WAVE7_SPEC.md, approved 2026-10-05). The policy is a *readout*, not a
generator: the prompt renders the leak-free engine state (the wave-1-5 field family
so comparisons to those arms are direct), the question, and the 21 allowed orders as
single-token letters A..U -> orders 0..40 step 2. The answer is the softmax over the
last-position logits of the 21 label token ids. Primary composition: argmax. Secondary:
expectation over the grid (both reported). No generation, no sampling, no parse step.

Invariants (spec D2), asserted per batch:
  - labels are single tokens in the loaded tokenizer, resolved on the *rendered
    continuation* (``tokenizer(prompt + label).input_ids[-1]``) — not on the bare
    letter (WAVE7-AUDIT-1 3c P2): the read position is what follows the rendered
    prompt, and a space-merged variant would silently read the wrong logit.
  - candidate-letter mass >= ``require_mass`` (default 0.99). NB: this proves format
    compliance only (AUDIT-1 3b: mass >= 0.998 even on empty material) — it is
    asserted because the spec pre-registered it, not because it measures quality.
  - option order is fixed ascending (0, 2, ..., 40).

State rendering comes from the repo's engine ``GameStepper.observations()`` ctx
dicts, with the wave-1-5 field text (agents.py build_prompt). Hand-typed states are
NOT accepted (AUDIT-1 3c P1: the probe's hand-typed states were engine-unreachable);
tests golden-check the renderer against real engine ctxs.

Prompt bytes are pinned by hash (AUDIT-1 3c P3): the chat template renders
``enable_thinking=False`` from the model's own tokenizer_config (a Jinja variable
that works on both 4.55.4 and 4.57.6), and `prompt_hash()` lets a test/sbatch assert
the cluster renders byte-identical prompts.

Torch/transformers imports are lazy and internal (same policy as grpo_train): this
module imports on a torch-less box, so the hub base suite keeps running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import string
from pathlib import Path
from typing import Callable

# ... order grid (A..U) — fixed ascending by pre-registered D2.
ORDER_GRID = list(range(0, 41, 2))            # 21 options
LABELS = list(string.ascii_uppercase[:len(ORDER_GRID)])  # A..U

QUESTION = "How many units should this stage order from its supplier this week?"

# System instruction: repo field text voice (agents.py build_system_prompt) plus
# the closed-choice letter rule. build_prompt is NOT reused verbatim — it ends
# "Reply with ONLY an integer" (AUDIT-1 3c P1), which contradicts the letter
# interface; the environment/objective/cost sentences below are its verbatim text.
INSTRUCTION = (
    "You are one stage in a four-stage beer supply chain:\n"
    "Retailer -> Wholesaler -> Distributor -> Factory.\n"
    "You manage inventory at your stage. Each week you decide how many units to "
    "order from your supplier (upstream). Goal: minimize total supply chain cost.\n"
    "Holding cost is $1 per unit per week; backlog cost is $2 per unit per week. "
    "Shipments from your supplier take 2 weeks to arrive.\n"
    "You are given your stage's state and a list of 21 possible orders labelled "
    "A..U. Choose the best order using only the supplied state. "
    "Reply with exactly one option letter, without explanation."
)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Prompt rendering (repo field text, engine ctx dicts only)
# --------------------------------------------------------------------------- #
def render_options(grid: list[int] | None = None) -> str:
    """Options block: 'A. order 0 units' ... 'U. order 40 units' (fixed ascending)."""
    grid = ORDER_GRID if grid is None else grid
    if list(grid) != sorted(grid):
        raise ValueError(f"option grid must be ascending, got {grid}")
    if len(grid) != len(LABELS):
        raise ValueError(f"grid has {len(grid)} options but {len(LABELS)} labels")
    return "\n".join(f"{LABELS[i]}. order {v} units" for i, v in enumerate(grid))


def render_state(ctx: dict) -> str:
    """Render an engine observation ctx (GameStepper.observations()) with the
    wave-1-5 field text (agents.py build_prompt). Raises on unknown/missing fields
    so a hand-typed (engine-unreachable) state cannot silently pass: every field
    name is asserted present, and the retailer-only incoming_now is enforced."""
    for key in ("role", "t", "on_hand", "backlog", "outstanding", "incoming_last",
                "incoming_now", "last_orders"):
        if key not in ctx:
            raise KeyError(f"ctx missing required observation key {key!r} (render "
                           f"only engine GameStepper.observations() dicts)")
    if ctx["role"] == "retailer":
        incoming_line = (f"- Incoming order from your customer this week: "
                         f"{ctx['incoming_now']}\n")
    elif ctx["incoming_now"] is None:
        incoming_line = ("- Incoming order from your customer this week: not yet "
                         "known (you decide on last week's information)\n")
    else:
        raise ValueError("non-retailer ctx carried incoming_now (engine leak) — "
                         "only the retailer ever sees same-week demand")
    return (
        f"Current week: {ctx['t']}\n"
        f"Your state at the start of the week:\n"
        f"- On-hand inventory: {ctx['on_hand']}\n"
        f"- Backlog (unfilled orders owed to your customer): {ctx['backlog']}\n"
        f"- Outstanding orders (ordered but not yet received): {ctx['outstanding']}\n"
        f"- Incoming order from your customer last week: {ctx['incoming_last']}\n"
        + incoming_line
        + f"- Your recent orders: {list(ctx['last_orders'])}\n"
    )


def render_body(ctx: dict, grid: list[int] | None = None) -> str:
    """Material + question + options (the user message), verbatim probe shape."""
    return (f"Material:\n{render_state(ctx)}\n\n"
            f"Question:\n{QUESTION}\n\n"
            f"Options:\n{render_options(grid)}")


def chat_messages(ctx: dict) -> list[dict]:
    """(system instruction, user body) — leaks nothing the engine does not show."""
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": render_body(ctx)},
    ]


def render_prompt(ctx: dict, tokenizer: object, *, enable_thinking: bool = False,
                  grid: list[int] | None = None) -> str:
    """Full prompt string via the model's own chat template (add_generation_prompt).

    ``enable_thinking`` is a Jinja template variable (Qwen3 tokenizer_config), so
    it works on transformers 4.55.4 and 4.57.6 alike (AUDIT-1 3a); the fallback
    documented in spec §5 (pre-rendered prompt constant) is covered by the
    ``--prompt-hash`` CLI + fixture: any template drift breaks the hash assert.
    """
    messages = list(chat_messages(ctx))
    if grid is not None:
        messages[1] = {"role": "user",
                       "content": f"Material:\n{render_state(ctx)}\n\n"
                                  f"Question:\n{QUESTION}\n\n"
                                  f"Options:\n{render_options(grid)}"}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking)


def prompt_hash(prompt: str) -> str:
    """SHA-256 of the rendered prompt bytes — the pinned template fixture
    (AUDIT-1 3c P3): hub and cluster must render byte-identical prompts."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def resolve_label_ids(tokenizer: object, sample_prompt: str,
                      labels: list[str] | None = None) -> dict[str, int]:
    """Label letter -> token id, asserted on the RENDERED CONTINUATION.

    For every label, the token that follows the rendered prompt (the read
    position) must be exactly that letter's own single-token encoding. This is
    the AUDIT-1 3c P2 fix over the seed's bare-letter ``encode(l)[0]``: if a
    tokenizer merged the letter with a leading space after the template tail, the
    naive id would read a different (dead) logit column.
    """
    labels = LABELS if labels is None else labels
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels")
    ids: dict[str, int] = {}
    for label in labels:
        continuation = tokenizer(sample_prompt + label, add_special_tokens=False)
        if len(continuation.input_ids) < 1:
            raise ValueError(f"label {label!r}: empty continuation tokenization")
        cid = int(continuation.input_ids[-1])
        matched = False
        for form in (label, " " + label):        # bare vs space-prefixed variant
            toks = tokenizer(form, add_special_tokens=False).input_ids
            if len(toks) == 1 and int(toks[0]) == cid:
                matched = True
                break
        if not matched:
            raise ValueError(
                f"label {label!r}: continuation token {cid} is not the single "
                f"token of {label!r} (bare {tokenizer.encode(label, add_special_tokens=False)}) "
                f"— the read position will not see that label")
        ids[label] = cid
    return ids


# --------------------------------------------------------------------------- #
# Batched readout
# --------------------------------------------------------------------------- #
class ReadoutModel:
    """Letter-logit readout over a frozen base (or a LoRA adapter on top).

    Batched: prompts are LEFT-padded (pad = eos), one forward pass, and the
    answer logits are read at the LAST position — valid for every row because
    left padding never puts a pad token at the end (AUDIT-1 3c P2: right-padded
    batches with logits[:, -1] read pad positions). No `logits_to_keep` (version
    gated between 4.55.4/4.57.6 per the brief): `logits[:, -1]` is read instead.
    On an L4 the full-logits materialization at batch 32 / T~400 is ~7 GB fp32
    (bf16 halves it) — the price of the version-safe path, noted for the trainer.
    """

    def __init__(self, model_name: str, *, device: str | None = None,
                 dtype: str | None = None, local_files_only: bool = True,
                 adapter_dir: str | Path | None = None, seed: int = 12345,
                 max_length: int | None = 1024):
        if not _torch_available():
            raise ImportError("decision_readout needs torch — run under the "
                              "torch venv (e.g. /home/pino/bw6-eval/.venvs/jev/bin/python)")
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # dtype: "float32" on CPU (hub smoke; the seed's verified numbers),
        # "bfloat16" on GPU unless overridden. `torch_dtype=` is the kwarg name
        # both 4.55.4 and 4.57.6 accept (4.57 renamed it `dtype`; torch_dtype
        # still works there with a deprecation warning — AUDIT-1 3c P2).
        if dtype is None:
            dtype = "float32" if self.device == "cpu" else "bfloat16"
        torch.manual_seed(seed)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict = {
            "local_files_only": local_files_only,
            "torch_dtype": getattr(torch, dtype),
            "attn_implementation": "sdpa",
        }
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if adapter_dir is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        self.model.to(self.device).eval()

        # single-token + rendered-continuation asserts on a canonical prompt
        sample = render_prompt({
            "role": "retailer", "t": 0, "on_hand": 0, "backlog": 0,
            "outstanding": 0, "incoming_last": 4, "incoming_now": 4,
            "last_orders": [],
        }, self.tokenizer)
        self.label_ids = resolve_label_ids(self.tokenizer, sample)
        if list(self.label_ids) != LABELS:
            raise AssertionError("label order must be fixed A..U (D2)")
        self.prompt_hash = prompt_hash(sample)

    # -- composition --------------------------------------------------------- #
    def compose(self, probs: "object"):  # torch 1-D tensor over the grid
        """argmax (primary) + expectation (secondary) over the grid."""
        idx = int(probs.argmax())
        grid = self._torch.tensor(ORDER_GRID, dtype=probs.dtype, device=probs.device)
        return {"argmax_index": idx, "argmax_order": ORDER_GRID[idx],
                "expect_order": float(probs @ grid)}

    def readout_batch(self, prompts: list[str], *, require_mass: float = 0.99
                      ) -> list[dict]:
        """Batched readout. Returns one dict per prompt:
        {probs, argmax_index, argmax_order, argmax_label, expect_order, mass,
         tokens}."""
        if not prompts:
            return []
        import torch
        tok = self.tokenizer
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False)
        if self.max_length is not None and inputs.input_ids.shape[1] > self.max_length:
            raise ValueError(f"prompt of {inputs.input_ids.shape[1]} tokens exceeds "
                             f"max_length={self.max_length} — raising instead of "
                             f"silently truncating a decision prompt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs, use_cache=False).logits
            last = logits[:, -1, :].float()          # left-padded: always real
            cand = last[:, list(self.label_ids.values())]
            probs = cand.softmax(-1)
            masses = (cand.logsumexp(-1) - last.logsumexp(-1)).exp()
        out = []
        for i, p in enumerate(prompts):
            row = probs[i]
            mass = float(masses[i])
            if mass < require_mass:
                raise AssertionError(
                    f"label mass {mass:.4f} < {require_mass} (prompt {i}): "
                    f"candidate-letter invariant violated")
            comp = self.compose(row)
            out.append({
                "probs": [float(x) for x in row.tolist()],
                "argmax_index": comp["argmax_index"],
                "argmax_order": comp["argmax_order"],
                "argmax_label": LABELS[comp["argmax_index"]],
                "expect_order": comp["expect_order"],
                "mass": mass,
                "tokens": inputs["attention_mask"][i].sum().item(),
            })
        return out

    def readout(self, prompt: str, **kw) -> dict:
        return self.readout_batch([prompt], **kw)[0]

    def as_agent(self, role: str) -> Callable[[dict], int]:
        """run_game-compatible callable: ctx -> argmax order (grid value)."""
        def _agent(ctx: dict) -> int:
            if ctx.get("role") is None:
                ctx = dict(ctx, role=role)
            prompt = render_prompt(ctx, self.tokenizer)
            with self._torch.inference_mode():
                rec = self.readout(prompt)
            return int(rec["argmax_order"])
        return _agent


# --------------------------------------------------------------------------- #
# CLI: readout a JSONL of engine ctxs (or plain prompts)
# --------------------------------------------------------------------------- #
def _jsonl_lines(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_bullwhip.decision_readout",
                                 description="letter-logit readout over engine states")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--cases", default="cases.jsonl",
                    help="JSONL: one {'role':..,'ctx':{engine observation}} per line, "
                         "or {'prompt': <pre-rendered>} to skip rendering")
    ap.add_argument("--out", default="-")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--prompt-hash", action="store_true",
                    help="print SHA-256 of one canonical rendered prompt and exit "
                         "(template-drift fixture for hub/cluster pinning)")
    args = ap.parse_args(argv)

    rm = ReadoutModel(args.model, device=args.device, dtype=args.dtype,
                      adapter_dir=args.adapter)
    if args.prompt_hash:
        print(rm.prompt_hash)
        return 0
    out_lines = []
    prompts, metas = [], []
    for line in _jsonl_lines(Path(args.cases)):
        if "prompt" in line:
            prompts.append(line["prompt"])
            metas.append({"rendered": True})
        else:
            ctx = line["ctx"]
            prompts.append(render_prompt(ctx, rm.tokenizer))
            metas.append({"role": line["role"], "ctx": ctx, "rendered": False})
        if len(prompts) >= args.batch:
            for rec, meta in zip(rm.readout_batch(prompts), metas):
                rec.update(meta)
                out_lines.append(rec)
            prompts, metas = [], []
    if prompts:
        for rec, meta in zip(rm.readout_batch(prompts), metas):
            rec.update(meta)
            out_lines.append(rec)
    text = "\n".join(json.dumps(r, sort_keys=True) for r in out_lines) + "\n"
    if args.out == "-":
        print(text, end="")
    else:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())