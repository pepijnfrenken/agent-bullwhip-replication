#!/usr/bin/env python3
"""wave8_gguf_serve.py — the WAVE8 CPU/GGUF step (docs/waves/WAVE8_SPEC.md §9.2).

The specialist decision model as a real CPU artifact: merge a trained adapter, convert to GGUF
(f16, Q8_0, Q4_K_M), serve with llama.cpp's llama-server and read the full 26-vector through
`/completion` (n_predict 1, temperature -1 = greedy, n_probs 64: token probabilities are a plain
softmax of the logits). A label absent from the returned top-64 counts as 0; probs are renormalised
over the label set and `mass` is the raw label mass (asserted against torch).

  convert     merge_and_unload the adapter -> HF dir -> GGUF f16 -> Q8_0 / Q4_K_M + sha256 manifest
              [needs torch+peft+transformers for the merge; llama.cpp for the rest]
  torch-probs reference 26-vectors from the torch readout for a states file   [torch]
  probs       the same states through a running llama-server
  equivalence mandatory gate: argmax agreement f16 >= 0.99 (|d top-1 prob| <= 0.02), Q8_0 >= 0.98,
              Q4_K_M reported; |mass_gguf - mass_torch| < 0.01
  eval        the standard protocol (step-argmax n=1, noisy-argmax n=30) through the server;
              MEASURES prefill latency, decisions/s, wall time (no number is pre-written anywhere)

Equivalence + serving logic is exercised against a fake llama-server in tests/test_w8_gguf.py;
the real convert/serve path is UNEXERCISED until a trained adapter exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_bullwhip.decision_readout import (PINNED_REVISION, W8_STAGED_GRID, get_grid,  # noqa: E402
                                             prompt_hash, render_prompt, render_prompt_str)
from agent_bullwhip.qwen_chat import render_nothink                                 # noqa: E402

LLAMA_CPP = "/home/pino/bw6-eval/llama.cpp"
EQUIV = {"f16": {"argmax": 0.99, "top1_abs": 0.02}, "q8_0": {"argmax": 0.98, "top1_abs": None},
         "q4_k_m": {"argmax": None, "top1_abs": None}}
MASS_TOL = 0.01


class _StrTok:
    """Tokenizer stand-in so decision_readout.render_prompt works without transformers."""
    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, **kw):
        if tokenize or not add_generation_prompt or enable_thinking:
            raise ValueError("pure-string renderer: tokenize=False, generation prompt, thinking off only")
        return render_nothink(messages)


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def label_probs(resp: dict, labels: list[str]) -> list[float]:
    """Raw full-vocab probability of each label token from a llama-server /completion reply.
    Handles the `top_logprobs` (logprob) and the older `probs` (prob) shapes; absent -> 0."""
    cp = resp.get("completion_probabilities") or []
    if not cp:
        raise ValueError("llama-server reply carries no completion_probabilities (n_probs ignored?)")
    item = cp[0]
    entries = item.get("top_logprobs") or item.get("probs") or []
    got: dict[str, float] = {}
    for e in entries:
        tok = e.get("token", e.get("tok_str"))
        if "logprob" in e:
            p = math.exp(e["logprob"])
        else:
            p = float(e.get("prob", 0.0))
        if tok in labels:
            got[tok] = max(got.get(tok, 0.0), p)
    return [got.get(l, 0.0) for l in labels]


class GgufReadout:
    """Duck-types ReadoutModel for decision_eval.run_batched_eval (readout_batch + tokenizer)."""

    def __init__(self, url: str, grid_id: str = W8_STAGED_GRID, n_probs: int = 64,
                 timeout: float = 600.0, post=_http_post):
        spec = get_grid(grid_id)
        self.url, self.grid_id, self.n_probs, self.timeout, self._post = url.rstrip("/"), grid_id, n_probs, timeout, post
        self.grid, self.labels = list(spec.values), list(spec.labels)
        self.tokenizer = _StrTok()
        self.model_name, self.revision = f"gguf@{url}", None
        self.latencies_ms: list[float] = []

    def readout_batch(self, prompts: list[str], *, require_mass: float = 0.99) -> list[dict]:
        out = []
        for p in prompts:
            t0 = time.perf_counter()
            resp = self._post(self.url + "/completion",
                              {"prompt": p, "n_predict": 1, "temperature": -1,
                               "n_probs": self.n_probs, "cache_prompt": True}, self.timeout)
            self.latencies_ms.append((time.perf_counter() - t0) * 1000)
            raw = label_probs(resp, self.labels)
            mass = sum(raw)
            if mass <= 0:
                raise AssertionError("no label token among the returned top-n_probs")
            if mass < require_mass:
                raise AssertionError(f"label mass {mass:.4f} < {require_mass}")
            probs = [x / mass for x in raw]
            idx = max(range(len(probs)), key=probs.__getitem__)
            out.append({"probs": probs, "argmax_index": idx, "argmax_order": self.grid[idx],
                        "argmax_label": self.labels[idx], "mass": mass,
                        "expect_order": sum(q * v for q, v in zip(probs, self.grid)),
                        "tokens": int(resp.get("tokens_evaluated", 0))})
        return out


# ------------------------------------------------------------------ equivalence ----
def equivalence_gate(torch_rows: list[dict], gguf_rows: list[dict], quant: str) -> dict:
    """rows = [{id, probs, mass?}]. Gate per quant level from W8 §9.2.3."""
    t = {r["id"]: r for r in torch_rows}
    g = {r["id"]: r for r in gguf_rows}
    ids = sorted(set(t) & set(g))
    if not ids:
        raise ValueError("no common ids between torch and gguf rows")
    am = lambda p: max(range(len(p)), key=p.__getitem__)
    agree = sum(1 for i in ids if am(t[i]["probs"]) == am(g[i]["probs"])) / len(ids)
    d = sorted(abs(max(t[i]["probs"]) - max(g[i]["probs"])) for i in ids)
    mass_d = [abs(t[i]["mass"] - g[i]["mass"]) for i in ids if "mass" in t[i] and "mass" in g[i]]
    thr = EQUIV[quant]
    ok = True
    if thr["argmax"] is not None:
        ok &= agree >= thr["argmax"]
    if thr["top1_abs"] is not None:
        ok &= d[-1] <= thr["top1_abs"]
    mass_ok = (max(mass_d) < MASS_TOL) if mass_d else None
    if mass_ok is False:
        ok = False
    return {"quant": quant, "n": len(ids), "argmax_agreement": agree,
            "top1_abs_diff_max": d[-1], "top1_abs_diff_p99": d[min(len(d) - 1, math.ceil(0.99 * len(d)) - 1)],
            "top1_abs_diff_mean": sum(d) / len(d), "mass_abs_diff_max": max(mass_d) if mass_d else None,
            "mass_ok": mass_ok, "thresholds": thr,
            "gated": thr["argmax"] is not None, "pass": bool(ok) if thr["argmax"] is not None else None}


# ----------------------------------------------------------------------- commands ----
def _states(paths: list[str], n: int) -> list[dict]:
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows[:n]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_commands(adapter: str, out: str, model: str, revision: str | None,
                     llama_cpp: str = LLAMA_CPP) -> list[list[str]]:
    """The exact command list (printed by --dry-run; run in order by `convert`)."""
    o = Path(out)
    merge = ("import sys,torch;from transformers import AutoModelForCausalLM,AutoTokenizer;"
             "from peft import PeftModel;m,rev,ad,out=sys.argv[1:5];rev=rev or None;"
             "b=AutoModelForCausalLM.from_pretrained(m,revision=rev,torch_dtype=torch.bfloat16,local_files_only=True);"
             "x=PeftModel.from_pretrained(b,ad).merge_and_unload();x.save_pretrained(out);"
             "AutoTokenizer.from_pretrained(m,revision=rev,local_files_only=True).save_pretrained(out)")
    f16 = str(o / "model-f16.gguf")
    return [["python3", "-c", merge, model, revision or "", adapter, str(o / "merged")],
            ["python3", f"{llama_cpp}/convert_hf_to_gguf.py", str(o / "merged"),
             "--outtype", "f16", "--outfile", f16],
            [f"{llama_cpp}/build/bin/llama-quantize", f16, str(o / "model-q8_0.gguf"), "Q8_0"],
            [f"{llama_cpp}/build/bin/llama-quantize", f16, str(o / "model-q4_k_m.gguf"), "Q4_K_M"]]


def cmd_convert(a) -> int:
    cmds = convert_commands(a.adapter, a.out, a.model, a.revision, a.llama_cpp)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in cmds:
        print("+", " ".join(c[:3] if c[1] == "-c" else c), flush=True)
        if not a.dry_run:
            subprocess.run(c, check=True)
    if not a.dry_run:
        man = {f.name: {"bytes": f.stat().st_size, "sha256": _sha(f)}
               for f in sorted(Path(a.out).glob("*.gguf"))}
        (Path(a.out) / "manifest.json").write_text(json.dumps(
            {"adapter": a.adapter, "model": a.model, "revision": a.revision, "files": man}, indent=1))
        print(json.dumps(man, indent=1))
    return 0


def cmd_torch_probs(a) -> int:
    from agent_bullwhip.decision_readout import ReadoutModel
    rows = _states(a.states, a.n)
    rm = ReadoutModel(a.model, adapter_dir=a.adapter, revision=a.revision or None, grid_id=a.grid)
    out = []
    for lo in range(0, len(rows), a.batch):
        prompts = [render_prompt(r["ctx"], rm.tokenizer, grid_id=a.grid) for r in rows[lo:lo + a.batch]]
        for i, rec in enumerate(rm.readout_batch(prompts, require_mass=0.0)):
            out.append({"id": lo + i, "probs": rec["probs"], "mass": rec["mass"]})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"torch probs: {len(out)} states -> {a.out}")
    return 0


def cmd_probs(a) -> int:
    rows = _states(a.states, a.n)
    gr = GgufReadout(a.url, a.grid)
    out = []
    for i, r in enumerate(rows):
        rec = gr.readout_batch([render_prompt_str(r["ctx"], grid_id=a.grid)], require_mass=0.0)[0]
        out.append({"id": i, "probs": rec["probs"], "mass": rec["mass"]})
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"gguf probs: {len(out)} states, mean latency {sum(gr.latencies_ms)/len(gr.latencies_ms):.1f} ms -> {a.out}")
    return 0


def cmd_equivalence(a) -> int:
    rd = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    res = equivalence_gate(rd(a.torch), rd(a.gguf), a.quant)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return 0 if res["pass"] in (True, None) else 1


def cmd_eval(a) -> int:
    from agent_bullwhip import decision_eval as de
    gr = GgufReadout(a.url, a.grid)
    arms = [("argmax", "step", 1), ("argmax", "noisy", a.runs)]
    summ = {}
    for mode, pattern, runs in arms:
        tag = f"{a.tag_prefix}-{mode}-{pattern}"
        t0 = time.time()
        n0 = len(gr.latencies_ms)
        demands = de.make_demands(pattern, mode, runs, 36, 1000)
        recs = de.run_batched_eval(gr, demands, mode, 1000, 36, tag, a.model_name,
                                   require_mass=0.0, grid_id=a.grid, traces=True, adapter=a.gguf_name)
        wall = time.time() - t0
        lat = gr.latencies_ms[n0:]
        outdir = Path(a.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{tag}.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
        costs = [r["total_cost"] for r in recs]
        summ[tag] = {"n_runs": len(recs), "mean_cost": sum(costs) / len(costs), "wall_sec": wall,
                     "n_decisions": len(lat), "cpu_latency_ms_mean": sum(lat) / len(lat),
                     "cpu_latency_ms_max": max(lat),
                     "decisions_per_sec": len(lat) / (sum(lat) / 1000), "gguf": a.gguf_name,
                     "parse_fail": 0, "min_mass": min(t["mass"] for r in recs for v in r["traces"].values() for t in v)}
        (outdir / f"{tag}.summary.json").write_text(json.dumps(summ[tag], indent=1, sort_keys=True) + "\n")
        print(tag, json.dumps(summ[tag]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--adapter", required=True); c.add_argument("--out", required=True)
    c.add_argument("--model", default="Qwen/Qwen3-0.6B"); c.add_argument("--revision", default=PINNED_REVISION)
    c.add_argument("--llama-cpp", default=LLAMA_CPP); c.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("torch-probs")
    t.add_argument("--states", nargs="+", required=True); t.add_argument("--n", type=int, default=2000)
    t.add_argument("--adapter", default=None); t.add_argument("--model", default="Qwen/Qwen3-0.6B")
    t.add_argument("--revision", default=PINNED_REVISION); t.add_argument("--grid", default=W8_STAGED_GRID)
    t.add_argument("--batch", type=int, default=32); t.add_argument("--out", required=True)
    p = sub.add_parser("probs")
    p.add_argument("--states", nargs="+", required=True); p.add_argument("--n", type=int, default=2000)
    p.add_argument("--url", default="http://127.0.0.1:8080"); p.add_argument("--grid", default=W8_STAGED_GRID)
    p.add_argument("--out", required=True)
    e = sub.add_parser("equivalence")
    e.add_argument("--torch", required=True); e.add_argument("--gguf", required=True)
    e.add_argument("--quant", choices=sorted(EQUIV), required=True); e.add_argument("--out", default=None)
    v = sub.add_parser("eval")
    v.add_argument("--url", default="http://127.0.0.1:8080"); v.add_argument("--grid", default=W8_STAGED_GRID)
    v.add_argument("--runs", type=int, default=30); v.add_argument("--outdir", default="results/wave8_gguf")
    v.add_argument("--tag-prefix", default="w8-gguf"); v.add_argument("--model-name", default="qwen3-gguf")
    v.add_argument("--gguf-name", default="model-q8_0.gguf")
    a = ap.parse_args(argv)
    return {"convert": cmd_convert, "torch-probs": cmd_torch_probs, "probs": cmd_probs,
            "equivalence": cmd_equivalence, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
