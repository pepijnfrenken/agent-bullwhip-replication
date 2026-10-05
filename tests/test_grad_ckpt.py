"""Gradient-checkpointing wiring for decision_train (torch hosts only).

The wave-7 SFT OOMs a 22 GiB L4 at batch 32: ~21.5 GiB of retained
activations across 28 layers, fatal on the first forward. `--grad-ckpt`
recomputes them instead. This locks the enable path against
transformers/peft API drift — the classic failure mode is a frozen base
plus checkpointing raising "element 0 of tensors does not require grad"
on the first backward, which only shows up on a real training step.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from agent_bullwhip.decision_train import (  # noqa: E402
    _enable_grad_checkpointing,
)


def _tiny_lora_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    model = Qwen3ForCausalLM(cfg)
    lcfg = peft.LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                           target_modules=["q_proj", "k_proj", "v_proj",
                                           "o_proj"],
                           task_type="CAUSAL_LM")
    m = peft.get_peft_model(model, lcfg)
    for name, p in m.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
    return m


def test_grad_ckpt_forward_backward_on_frozen_base():
    model = _tiny_lora_model()
    _enable_grad_checkpointing(model)
    assert model.config.use_cache is False
    ids = torch.randint(0, 256, (4, 16))
    out = model(input_ids=ids, use_cache=False)
    loss = out.logits.float().mean()
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_" in n]
    assert grads, "no lora params found"
    assert all(g is not None and torch.isfinite(g).all() for g in grads), \
        "checkpointing broke gradient flow to the LoRA params"


def test_policy_sft_step_with_grad_ckpt():
    """One real SFT step through _Policy with grad_ckpt on the actual
    Qwen3-0.6B; skips when this host has no model cache (offline hub)."""
    from agent_bullwhip.decision_train import _Policy
    try:
        pol = _Policy("Qwen/Qwen3-0.6B", device="cpu", phase="sft",
                      max_length=512, seed=5, permute_options=True,
                      grad_ckpt=True)
    except Exception as exc:  # noqa: BLE001 - no cache / not offline-ready
        pytest.skip(f"model unavailable on this host: {exc}")
    try:
        from test_decision_train import synth_rows   # sibling test module
    except ImportError:  # pragma: no cover - packaging quirk
        pytest.skip("sibling test module not importable")
    rows = synth_rows(4)
    for r in rows:
        k = r["argmin_idx"]
        r["p"] = [0.6 / 20 if i != k else 0.6 for i in range(21)]
    out = pol.sft_step(rows, [r["p"] for r in rows], lr=1e-4, seed=5,
                       epoch=0, step=0)
    assert out.loss == out.loss and out.loss > 0, f"non-finite loss {out.loss}"
