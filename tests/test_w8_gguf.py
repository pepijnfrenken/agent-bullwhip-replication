"""wave8_gguf_serve against a fake llama-server (real HTTP, llama.cpp-shaped replies) +
equivalence gate math. The real convert/serve path is not exercised here (no adapter exists)."""
from __future__ import annotations

import importlib.util
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_bullwhip.decision_readout import get_grid, prompt_hash, render_prompt_str

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wave8_gguf_serve", ROOT / "scripts" / "wave8_gguf_serve.py")
gs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gs)
LABELS = list(get_grid("w8_g26b").labels)
CTX = {"role": "retailer", "t": 0, "on_hand": 0, "backlog": 0, "outstanding": 0,
       "incoming_last": 4, "incoming_now": 4, "last_orders": []}


def _reply(peak: int, shape: str = "new", drop: tuple = (), mass: float = 0.98):
    """llama.cpp-shaped /completion reply: label probs peak at `peak`; the rest of the mass sits
    on non-label tokens. `drop` = labels missing from the returned top-n."""
    w = [1.0 + (3.0 if i == peak else 0.0) for i in range(26)]
    z = sum(w)
    probs = {l: mass * x / z for l, x in zip(LABELS, w)}
    ent = [{"token": l, "logprob": math.log(p)} for l, p in probs.items() if l not in drop]
    ent.append({"token": " the", "logprob": math.log(1 - mass)})
    if shape == "new":
        item = {"id": 32, "logprob": ent[0]["logprob"], "token": LABELS[0], "top_logprobs": ent}
    else:
        item = {"content": LABELS[0], "probs": [{"tok_str": e["token"], "prob": math.exp(e["logprob"])} for e in ent]}
    return {"content": LABELS[peak], "tokens_evaluated": 417, "completion_probabilities": [item]}


class _Server:
    def __init__(self, fn):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(n))
                outer.requests.append((self.path, body))
                data = json.dumps(fn(body)).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass
        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    s = _Server(lambda body: _reply(7))
    yield s
    s.close()


def test_readout_through_real_http_and_request_shape(server):
    gr = gs.GgufReadout(server.url, "w8_g26b")
    prompt = render_prompt_str(CTX, grid_id="w8_g26b")
    rec = gr.readout_batch([prompt], require_mass=0.0)[0]
    path, body = server.requests[0]
    assert path == "/completion" and body["n_predict"] == 1 and body["temperature"] == -1
    assert body["n_probs"] == 64 and body["prompt"] == prompt
    assert rec["argmax_index"] == 7 and rec["argmax_order"] == get_grid("w8_g26b").values[7]
    assert sum(rec["probs"]) == pytest.approx(1.0) and rec["mass"] == pytest.approx(0.98)
    assert len(rec["probs"]) == 26 and rec["tokens"] == 417 and len(gr.latencies_ms) == 1


def test_legacy_probs_shape_and_missing_label_counts_as_zero():
    s = _Server(lambda body: _reply(3, shape="old", drop=("Z", "Y")))
    try:
        rec = gs.GgufReadout(s.url, "w8_g26b").readout_batch(["p"], require_mass=0.0)[0]
    finally:
        s.close()
    assert rec["probs"][25] == 0.0 and rec["probs"][24] == 0.0 and rec["argmax_index"] == 3
    assert rec["mass"] < 0.98            # the dropped labels' mass is gone (they are not in the top-n)


def test_mass_assertion_and_empty_reply():
    s = _Server(lambda body: _reply(1, mass=0.5))
    try:
        gr = gs.GgufReadout(s.url, "w8_g26b")
        with pytest.raises(AssertionError, match="label mass"):
            gr.readout_batch(["p"], require_mass=0.99)
        assert gr.readout_batch(["p"], require_mass=0.0)[0]["mass"] == pytest.approx(0.5)
    finally:
        s.close()
    s = _Server(lambda body: {"content": "x"})
    try:
        with pytest.raises(ValueError, match="completion_probabilities"):
            gs.GgufReadout(s.url, "w8_g26b").readout_batch(["p"])
    finally:
        s.close()


def test_string_tokenizer_stands_in_for_apply_chat_template():
    from agent_bullwhip.decision_readout import render_prompt
    tok = gs._StrTok()
    assert render_prompt(CTX, tok, grid_id="w8_g26b") == render_prompt_str(CTX, grid_id="w8_g26b")
    with pytest.raises(ValueError):
        tok.apply_chat_template([], tokenize=True)


def _rows(peaks, mass=0.98, top=0.6):
    out = []
    for i, k in enumerate(peaks):
        p = [(1 - top) / 25] * 26
        p[k] = top
        out.append({"id": i, "probs": p, "mass": mass})
    return out


def test_equivalence_gate_thresholds():
    t = _rows([i % 26 for i in range(200)])
    same = gs.equivalence_gate(t, _rows([i % 26 for i in range(200)]), "f16")
    assert same["pass"] and same["argmax_agreement"] == 1.0 and same["top1_abs_diff_max"] == 0
    # 2 of 200 argmax flips = 0.99 -> passes f16 (>= 0.99); 3 flips fails f16 but passes Q8_0 (>= 0.98)
    flip = lambda n: _rows([(i + 1) % 26 if i < n else i % 26 for i in range(200)])
    assert gs.equivalence_gate(t, flip(2), "f16")["pass"]
    assert not gs.equivalence_gate(t, flip(3), "f16")["pass"]
    assert gs.equivalence_gate(t, flip(3), "q8_0")["pass"]
    assert not gs.equivalence_gate(t, flip(5), "q8_0")["pass"]
    assert gs.equivalence_gate(t, flip(40), "q4_k_m")["pass"] is None       # reported, no gate
    # top-1 probability drift > 0.02 fails f16 even with identical argmax
    assert not gs.equivalence_gate(t, _rows([i % 26 for i in range(200)], top=0.63), "f16")["pass"]
    # mass must agree within 0.01
    bad_mass = gs.equivalence_gate(t, _rows([i % 26 for i in range(200)], mass=0.95), "f16")
    assert bad_mass["mass_ok"] is False and not bad_mass["pass"]


def test_convert_command_list_is_complete_and_ordered():
    cmds = gs.convert_commands("results/x/checkpoint-169", "/tmp/gg", "Qwen/Qwen3-0.6B", "rev")
    assert len(cmds) == 4 and "merge_and_unload" in cmds[0][2]
    assert cmds[1][1].endswith("convert_hf_to_gguf.py") and "f16" in cmds[1]
    assert cmds[2][-1] == "Q8_0" and cmds[3][-1] == "Q4_K_M"
    assert all(c[0].endswith("llama-quantize") for c in cmds[2:])
    assert gs.main(["convert", "--adapter", "a", "--out", "/tmp/gg_dry", "--dry-run"]) == 0


def test_eval_protocol_through_the_fake_server(tmp_path, server):
    gs.main(["eval", "--url", server.url, "--grid", "w8_g26b", "--runs", "2",
             "--outdir", str(tmp_path), "--gguf-name", "fake.gguf"])
    s1 = json.loads((tmp_path / "w8-gguf-argmax-step.summary.json").read_text())
    s2 = json.loads((tmp_path / "w8-gguf-argmax-noisy.summary.json").read_text())
    assert s1["n_runs"] == 1 and s2["n_runs"] == 2 and s1["n_decisions"] == 4 * 36
    assert s1["cpu_latency_ms_mean"] > 0 and s1["decisions_per_sec"] > 0 and s1["parse_fail"] == 0
    rec = json.loads((tmp_path / "w8-gguf-argmax-step.jsonl").read_text().splitlines()[0])
    assert rec["readout"]["schema"] == "w8-readout-1" and rec["readout"]["adapter"] == "fake.gguf"
    assert all(o == get_grid("w8_g26b").values[7] for v in rec["orders"].values() for o in v)
