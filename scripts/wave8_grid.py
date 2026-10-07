#!/usr/bin/env python3
"""wave8_grid.py — generate the WAVE8 choice-set artifact (results/wave8/grid_<id>.json).

The grid is produced by code (`decision_readout.GRIDS`), never hand-typed into a doc.
Default mode is torch/transformers-free: values, labels, the instruction text and the
canonical-prompt SHA-256 from the pure-string renderer (validated against the wave-7
fixture b96fb847… in tests). `--tokenizer-check` (needs transformers + a local snapshot)
additionally fills `token_ids`, asserts all labels are single tokens on the RENDERED
continuation with contiguous ids, and asserts the real `apply_chat_template` prompt SHA
equals the string-renderer SHA (WAVE8 pre-flight F2).

  python3 scripts/wave8_grid.py [--grid w8_g26] [--out results/wave8/grid_w8_g26.json]
  python3 scripts/wave8_grid.py --tokenizer-check [--model Qwen/Qwen3-0.6B] [--revision SHA]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_bullwhip.decision_eval import CANONICAL_CTX                      # noqa: E402
from agent_bullwhip.decision_readout import (GRIDS, PINNED_REVISION, instruction,  # noqa: E402
                                             prompt_hash, render_prompt,
                                             render_prompt_str, resolve_label_ids)


def build(grid_id: str, tokenizer_check: bool, model: str, revision: str | None,
          local_only: bool = True) -> dict:
    g = GRIDS[grid_id]
    sha_str = prompt_hash(render_prompt_str(CANONICAL_CTX, grid_id=grid_id))
    out = {"grid_id": grid_id, "values": list(g.values), "labels": list(g.labels),
           "instruction": instruction(grid_id), "prompt_sha256_string_renderer": sha_str,
           "token_ids": None, "prompt_sha256_apply_chat_template": None,
           "tokenizer_checked_with": None, "generated_by": "scripts/wave8_grid.py",
           "git_head": _head()}
    if tokenizer_check:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, revision=revision,
                                            local_files_only=local_only)
        sample = render_prompt(CANONICAL_CTX, tok, grid_id=grid_id)
        ids = resolve_label_ids(tok, sample, grid_id=grid_id)
        vals = list(ids.values())
        assert vals == list(range(vals[0], vals[0] + len(vals))), f"label ids not contiguous: {vals}"
        sha_tpl = prompt_hash(sample)
        assert sha_tpl == sha_str, (f"apply_chat_template SHA {sha_tpl} != string renderer "
                                    f"{sha_str}")
        import transformers
        out.update(token_ids=ids, prompt_sha256_apply_chat_template=sha_tpl,
                   tokenizer_checked_with={"model": model, "revision": revision,
                                           "transformers": transformers.__version__})
    return out


def _head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]),
                               "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="w8_g26b", choices=sorted(GRIDS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer-check", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=PINNED_REVISION)
    a = ap.parse_args(argv)
    res = build(a.grid, a.tokenizer_check, a.model, a.revision or None)
    out = Path(a.out or f"results/wave8/grid_{a.grid}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(f"grid {a.grid}: {len(res['values'])} options {res['labels'][0]}..{res['labels'][-1]} "
          f"values={res['values']}")
    print(f"prompt sha256 (string renderer): {res['prompt_sha256_string_renderer']}")
    if res["token_ids"]:
        print(f"token ids {list(res['token_ids'].values())[0]}..{list(res['token_ids'].values())[-1]} "
              f"(contiguous); apply_chat_template sha equal: True")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
