"""Adapter handoff for the wave-7 RLCD resume (AUDIT-2 P1-3), torch hosts only.

A tiny from-config Qwen3 (no model cache): the RLCD resume must load the
RLCD checkpoint ALONE. The old stack PeftModel(PeftModel(base, sft), ckpt)
re-initialised the shared adapter (every lora_B == 0, i.e. the base model)
and saved checkpoints with a nested key prefix.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from agent_bullwhip.decision_train import _attach_adapter  # noqa: E402


def _tiny_base():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    return Qwen3ForCausalLM(cfg).eval()


def _lora(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if "lora_" in n}


def _perturb(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.add_(torch.randn(p.shape, generator=g) * 0.05)


def test_rlcd_resume_loads_the_checkpoint_only(tmp_path):
    ids = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(1))
    sft = _attach_adapter(_tiny_base(), phase="sft", base_adapter=None,
                          adapter_dir=None)
    _perturb(sft, 1)                       # "trained" SFT adapter
    sft.save_pretrained(tmp_path / "sft")
    rl = _attach_adapter(_tiny_base(), phase="rlcd",
                         base_adapter=str(tmp_path / "sft"), adapter_dir=None)
    assert all(torch.equal(a, b) for a, b in
               zip(_lora(rl).values(), _lora(sft).values())), \
        "fresh RLCD start must carry the SFT adapter"
    _perturb(rl, 2)                        # "trained" RLCD adapter
    rl.save_pretrained(tmp_path / "rlcd-ck")
    want = _lora(rl)
    with torch.no_grad():
        want_logits = rl(input_ids=ids).logits

    res = _attach_adapter(_tiny_base(), phase="rlcd",
                          base_adapter=str(tmp_path / "sft"),
                          adapter_dir=str(tmp_path / "rlcd-ck"))
    # single PeftModel over the base, not PeftModel -> LoraModel -> PeftModel
    assert not isinstance(res.base_model.model, peft.PeftModel)
    got = _lora(res)
    assert list(got) == list(want)
    assert all(torch.equal(got[k], want[k]) for k in want)
    assert any(("lora_B" in k) and got[k].abs().sum() > 0 for k in got)
    with torch.no_grad():
        assert torch.allclose(res(input_ids=ids).logits, want_logits, atol=1e-6)
    # a checkpoint saved after the resume keeps the flat key prefix
    res.save_pretrained(tmp_path / "rlcd-ck2")
    from safetensors.torch import load_file
    keys = load_file(str(tmp_path / "rlcd-ck2" / "adapter_model.safetensors"))
    assert keys and all(k.startswith("base_model.model.model.") for k in keys)
