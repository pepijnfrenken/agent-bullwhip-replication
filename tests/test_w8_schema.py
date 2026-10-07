"""WAVE8 §7.2: traces/readout schema extension is backwards compatible by construction."""
from __future__ import annotations

import importlib.util
import json
import math
import random
import re
from pathlib import Path

import pytest

import agent_bullwhip.decision_eval as de
from agent_bullwhip.decision_readout import get_grid

ROOT = Path(__file__).resolve().parents[1]
W7_RECORD_FILE = ROOT / "results" / "eval" / "w7-rlcd-seed0-argmax-noisy.jsonl"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeReadout26:
    """Duck-typed ReadoutModel over the 26-letter grid: probs depend on the state
    (on_hand) so the letter-lock and diffuse cases are distinguishable."""
    revision = "fake-rev"
    model_name = "fake/model"

    class _Tok:
        @staticmethod
        def apply_chat_template(messages, **kw):
            return "prompt"
    tokenizer = _Tok()

    def readout_batch(self, prompts, *, require_mass=0.99):
        out = []
        for i, _ in enumerate(prompts):
            w = [1.0 + ((j * 7 + i) % 5) for j in range(26)]
            z = sum(w)
            p = [x / z for x in w]
            out.append({"probs": p, "argmax_index": max(range(26), key=p.__getitem__),
                        "argmax_order": get_grid("w8_g26").values[max(range(26), key=p.__getitem__)],
                        "mass": 0.99, "tokens": 10})
        return out


def test_w8_record_traces_and_readout_key():
    demands = [[4] * 6, [5] * 6]
    recs = de.run_batched_eval(FakeReadout26(), demands, "argmax", 1000, 6, "t", "m",
                               grid_id="w8_g26", traces=True, adapter="ad")
    grid = list(get_grid("w8_g26").values)
    r = recs[0]
    assert set(r["traces"]) == {"retailer", "wholesaler", "distributor", "factory"}
    for role, tr in r["traces"].items():
        assert [t["t"] for t in tr] == list(range(6))
        for t in tr:
            assert len(t["probs"]) == 26 and abs(sum(t["probs"]) - 1) < 1e-5
            assert t["confidence"] == max(t["probs"])
            assert t["order"] == grid[t["idx"]]
            assert abs(t["entropy"] - (-sum(q * math.log(q) for q in t["probs"] if q > 0))) < 1e-4
            assert set(t["ctx"]) == {"on_hand", "backlog", "outstanding", "incoming_last",
                                     "incoming_now", "last_orders"}
    ro = r["readout"]
    assert ro["schema"] == "w8-readout-1" and ro["grid_id"] == "w8_g26" and ro["grid"] == grid
    assert ro["labels"][0] == "A" and ro["labels"][-1] == "Z" and ro["adapter"] == "ad"
    assert ro["n_decisions"] == 4 * 6 and re.fullmatch(r"[0-9a-f]{64}", ro["prompt_sha"])
    s = de.trace_summary(recs, "w8_g26")
    assert s["grid_id"] == "w8_g26" and 0 <= s["top_letter_share"] <= 1 and s["min_mass"] == 0.99


def test_default_records_keep_the_wave7_shape():
    class Fake21(FakeReadout26):
        def readout_batch(self, prompts, *, require_mass=0.99):
            return [{"argmax_order": 4, "tokens": 10, "mass": 0.95, "probs": [1 / 21] * 21}
                    for _ in prompts]
    rec = de.run_batched_eval(Fake21(), [[4] * 4], "argmax", 1000, 4, "t", "m")[0]
    assert "readout" not in rec and rec["traces"] == {r: [] for r in rec["traces"]}
    assert set(rec) == {"run", "config", "model", "tag", "total_cost", "orders", "backlogs",
                        "failures", "calls", "tokens", "prompt_hash_dups", "traces", "tool_traces"}


@pytest.mark.skipif(not W7_RECORD_FILE.exists(), reason="wave-7 eval JSONL not in checkout")
def test_analyzers_read_wave7_and_wave8_records_identically(tmp_path):
    w7 = json.loads(W7_RECORD_FILE.read_text().splitlines()[0])
    recs = de.run_batched_eval(FakeReadout26(), [[4] * 36], "argmax", 1000, 36, "w7-x-argmax-noisy", "m",
                               grid_id="w8_g26", traces=True)
    w8 = recs[0]
    p7, p8 = tmp_path / "w7-a-argmax-noisy.jsonl", tmp_path / "w8-a-argmax-noisy.jsonl"
    p7.write_text(json.dumps(w7) + "\n")
    p8.write_text(json.dumps(w8) + "\n")
    wave7 = _load(ROOT / "scripts" / "wave7_analysis.py")
    a7, a8 = wave7.load_arm(p7), wave7.load_arm(p8)
    for a, rec in ((a7, w7), (a8, w8)):
        assert len(a["complete"]) == 1 and not a["failed"]
        assert a["complete"][0]["total_cost"] == rec["total_cost"]
        assert a["complete"][0]["orders"] == rec["orders"]
    ea = _load(ROOT / "hpc-umbrella" / "eval-analyze.py")
    r7, r8 = ea.load_arm(p7, 36, 4), ea.load_arm(p8, 36, 4)
    assert len(r7[0]) == len(r8[0]) == 1 and r7[1] == r8[1] == []
    assert r7[0][0]["total_cost"] == w7["total_cost"] and r8[0][0]["total_cost"] == w8["total_cost"]


def test_no_analyzer_reads_traces():
    # wave8_gguf_serve.py PRODUCES wave-8 records and summarises its own min_mass; it is not an analyzer
    for f in list((ROOT / "scripts").glob("*.py")) + [ROOT / "hpc-umbrella" / "eval-analyze.py"]:
        if f.name == "wave8_gguf_serve.py":
            continue
        txt = f.read_text()
        assert not re.search(r"""\[["']traces["']\]|\.get\(["']traces["']""", txt), f.name


def test_curriculum_pattern_uses_the_validation_band():
    d = de.make_demands("curriculum", "argmax", 4, 36, 9500)
    assert len(d) == 4 and all(len(x) == 36 for x in d)
    assert d == de.make_demands("curriculum", "argmax", 4, 36, 9500)
    assert d != de.make_demands("curriculum", "argmax", 4, 36, 9600)
