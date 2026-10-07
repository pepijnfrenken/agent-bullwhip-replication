"""WAVE8 `_Policy` behaviour that needs torch + transformers + peft + the cached
Qwen3-0.6B snapshot. Skipped on the torch-less hub (and wherever peft is missing);
it runs in the cluster venv during `stage.sh`'s pytest pass. UNOBSERVED locally in the
mission that wrote it: no peft install existed on the hub host."""
from __future__ import annotations

import pytest

from test_decision_train import synth_rows


@pytest.fixture(scope="module")
def torch_ok():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return True


def _rows(n=4):
    rows = synth_rows(n)
    for i, r in enumerate(rows):
        r["J"] = [float((k * 7 + i * 3) % 26) * 10 + 100 for k in range(26)]
        r["p"] = [1 / 26] * 26
    return rows


@pytest.fixture(scope="module")
def fresh_policy(torch_ok):
    from agent_bullwhip.decision_train import _Policy
    try:
        return _Policy("Qwen/Qwen3-0.6B", device="cpu", dtype="float32", phase="rlcd",
                       init="fresh", anchor="base", grid_id="w8_g26", permute_p=1.0,
                       micro_batch=2, max_length=1024, seed=5)
    except Exception as exc:  # noqa: BLE001 - snapshot not cached on this host
        pytest.skip(f"model unavailable on this host: {exc}")


def test_fresh_init_step0_policy_is_the_untrained_readout(fresh_policy):
    import torch
    from agent_bullwhip.decision_readout import render_prompt
    pol = fresh_policy
    prompts = [render_prompt(r["ctx"], pol.tok, grid_id="w8_g26") for r in _rows(3)]
    pol.model.eval()
    with torch.no_grad():
        with_adapter = pol._label_logits(pol._tokenize(prompts)).softmax(-1)
        with pol.model.disable_adapter():
            base = pol._label_logits(pol._tokenize(prompts)).softmax(-1)
    assert with_adapter.shape[-1] == 26
    assert (with_adapter - base).abs().max().item() < 1e-5      # lora_B = 0


def test_base_anchor_table_never_follows_the_policy(fresh_policy):
    import torch
    pol = fresh_policy
    rows = _rows(4)
    t0 = pol.compute_pi_sft(rows, 2)
    with torch.no_grad():
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    t1 = pol.compute_pi_sft(rows, 2)
    assert t0 == t1                       # anchor = adapter-disabled readout: policy-independent
    with torch.no_grad():
        for n, p in pol.model.named_parameters():
            if "lora_B" in n:
                p.sub_(0.05)


def test_permuted_micro_batched_rlcd_step_runs_and_moves_the_adapter(fresh_policy):
    import torch
    pol = fresh_policy
    rows = _rows(4)
    pol.set_pi_sft(pol.compute_pi_sft(rows, 2))
    before = {n: p.detach().clone() for n, p in pol.model.named_parameters() if "lora_B" in n}
    out = pol.rlcd_step(rows, [r["J"] for r in rows], lr=1e-3, beta_kl=0.02, sample_k=0,
                        seed=5, step=0)
    assert out.n_nonfinite == 0 and out.loss == out.loss and out.loss_kl is not None
    moved = any((p.detach() - before[n]).abs().max().item() > 0
                for n, p in pol.model.named_parameters() if "lora_B" in n)
    assert moved
