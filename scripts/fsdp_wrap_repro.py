#!/usr/bin/env python3
"""fsdp_wrap_repro.py — reproduce the FSDP wrap failure on CPU/gloo, no GPU needed.

Job 796505 died inside FullyShardedDataParallel.__init__ ->
_init_param_handle_from_module, and the exception text was lost when the NCCL
watchdog aborted the process. FSDP construction does not need CUDA, so the same
call is made here on 2 gloo ranks with a small model: either the same error
reappears with its message, or the GPU path differs and we know that too.

  torchrun --nproc_per_node=2 --standalone scripts/fsdp_wrap_repro.py
"""
from __future__ import annotations

import functools
import os
import sys
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy  # noqa: E402

MODEL = os.environ.get("REPRO_MODEL", "Qwen/Qwen3-0.6B")
REV = {"Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
       "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218"}.get(MODEL)


def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, local_files_only=True)
    if rank == 0:
        print(f"[repro] tie_word_embeddings={base.config.tie_word_embeddings} "
              f"torch={torch.__version__}", flush=True)
        lm = base.get_output_embeddings().weight
        emb = base.get_input_embeddings().weight
        print(f"[repro] lm_head is embed_tokens: {lm.data_ptr() == emb.data_ptr()}", flush=True)
    peft_model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM"))

    layer_cls = type(peft_model.get_base_model().model.layers[0])
    policy = functools.partial(transformer_auto_wrap_policy,
                               transformer_layer_cls={layer_cls})
    try:
        wrapped = FSDP(peft_model, auto_wrap_policy=policy, use_orig_params=True,
                       sync_module_states=True)
        if rank == 0:
            print(f"[repro] WRAP OK: {len(list(wrapped.modules()))} modules", flush=True)
        return 0
    except Exception:
        if rank == 0:
            print("[repro] WRAP FAILED — message follows", flush=True)
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    rc = main()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)
