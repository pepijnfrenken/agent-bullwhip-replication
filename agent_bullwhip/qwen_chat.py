"""Pure-string Qwen3 chat-template rendering (torch/transformers-free).

Reproduces `tokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True, enable_thinking=False)` for the plain system/user
message lists this repo sends (no tools, no assistant turns). The hub has no
transformers, so the hub-side prompt-byte fixtures (wave 6-fix contract prompt,
wave 8 readout prompt) are rendered with this function; equality with the real
`apply_chat_template` is asserted wherever transformers is importable.
"""
from __future__ import annotations

ASSISTANT_NOTHINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render_nothink(messages: list[dict]) -> str:
    """Qwen3 template, thinking disabled, generation prompt appended."""
    parts: list[str] = []
    for m in messages:
        role = m["role"]
        if role not in ("system", "user"):
            raise ValueError(f"render_nothink supports system/user only, got {role!r}")
        parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>\n")
    parts.append(ASSISTANT_NOTHINK)
    return "".join(parts)
