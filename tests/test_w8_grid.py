"""WAVE8 choice-set registry: w8_g26 constants, wave-7 default unchanged, labeler --grid."""
from __future__ import annotations

import json
import random
import string
from pathlib import Path

import pytest

from agent_bullwhip import decision_labeler as L
from agent_bullwhip.decision_readout import (
    DEFAULT_GRID_ID, GRIDS, INSTRUCTION, LABELS, ORDER_GRID, GridSpec, get_grid, instruction,
    prompt_hash, render_options, render_prompt_str)

W7_PROMPT_SHA = "b96fb847317aed3ce670a015a0e4bc269b5573299ee19f1f577499803b99ebb7"
CANONICAL_CTX = {"role": "retailer", "t": 0, "on_hand": 0, "backlog": 0, "outstanding": 0,
                 "incoming_last": 4, "incoming_now": 4, "last_orders": []}
G26 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
       34, 38, 44, 52, 62, 76, 94, 116, 140, 160]


def test_w8_g26_equals_spec_literal_and_is_valid():
    g = get_grid("w8_g26")
    assert list(g.values) == G26 and list(g.labels) == list(string.ascii_uppercase)
    assert len(set(g.values)) == 26 and all(b > a for a, b in zip(g.values, g.values[1:]))
    # letters A..P keep the wave-7 values (the untrained letter-E = 8 prior is the same decision)
    assert list(g.values[:16]) == ORDER_GRID[:16]


def test_wave7_defaults_unchanged():
    assert DEFAULT_GRID_ID == "w7_u21"
    assert ORDER_GRID == list(range(0, 41, 2)) and LABELS == list(string.ascii_uppercase[:21])
    assert "21 possible orders labelled A..U." in INSTRUCTION
    assert INSTRUCTION == instruction("w7_u21")


def test_string_renderer_reproduces_wave7_prompt_fixture():
    # the hub-side proof that qwen_chat.render_nothink + the templated instruction are
    # byte-identical to what apply_chat_template rendered when the fixture was recorded
    assert prompt_hash(render_prompt_str(CANONICAL_CTX)) == W7_PROMPT_SHA


def test_w8_instruction_and_options():
    ins = instruction("w8_g26")
    assert "a list of 26 possible orders labelled A..Z." in ins
    lines = render_options(grid_id="w8_g26").splitlines()
    assert len(lines) == 26 and lines[0] == "A. order 0 units" and lines[-1] == "Z. order 160 units"
    assert prompt_hash(render_prompt_str(CANONICAL_CTX, grid_id="w8_g26")) != W7_PROMPT_SHA
    # the w7 grid still rejects a 22-option list and non-ascending grids
    with pytest.raises(ValueError):
        render_options(list(range(0, 44, 2)))
    with pytest.raises(ValueError):
        render_options(list(reversed(G26)), grid_id="w8_g26")


def test_gridspec_validation_and_unknown_id():
    with pytest.raises(ValueError):
        GridSpec("bad", (0, 2, 2), ("A", "B", "C"))
    with pytest.raises(ValueError):
        GridSpec("bad", tuple(range(27)), tuple(string.ascii_uppercase) + ("a",))
    with pytest.raises(ValueError, match="unknown grid"):
        get_grid("nope")
    assert set(GRIDS) == {"w7_u21", "w8_g26", "w8_g26b", "w8_r3b", "w8_r3c"}


def test_behavior_random_grid_is_frozen_and_stream_unchanged():
    assert L.BEHAVIOR_RANDOM_GRID == list(range(0, 41, 2))
    rng = random.Random("t")
    # sequence recorded from the pre-change `rng.choice(ORDER_GRID)` on the same seed
    assert [rng.choice(L.BEHAVIOR_RANDOM_GRID) for _ in range(12)] == \
        [38, 40, 16, 10, 24, 2, 14, 36, 10, 28, 20, 22]


def test_labeler_scores_26_candidates_and_reports_edge_metrics(tmp_path):
    out = tmp_path / "lbl"
    m = L.run_labeler(40, 2, horizon=8, m_futures=3, out_prefix=str(out), grid_id="w8_g26",
                      score_reference=True)
    rows = [json.loads(l) for l in out.with_suffix(".jsonl").read_text().splitlines()]
    assert all(len(r["J"]) == 26 == len(r["p"]) and "ref_min" in r for r in rows)
    assert all(r["argmin_order"] == G26[r["argmin_idx"]] for r in rows)
    assert m["grid"] == G26 and m["grid_id"] == "w8_g26" and len(m["labels"]) == 26
    for k in ("top_edge_rate", "zero_rate", "target_mass_top_letter",
              "regret_vs_reference_mean", "regret_vs_reference_p90", "edge_argmin_rate"):
        assert k in m
    # regret >= 0 by construction: the 26-point grid is a subset of the reference
    assert all(min(r["J"]) >= r["ref_min"] - 1e-9 for r in rows)
    assert m["regret_vs_reference_mean"] >= 0
    # the default grid is untouched
    m7 = L.run_labeler(40, 1, horizon=6, m_futures=2)
    assert m7["grid"] == ORDER_GRID and all(len(r["J"]) == 21 for r in m7["rows"])


def test_labeler_cli_rejects_unknown_grid():
    with pytest.raises(ValueError):
        L.main(["--grid", "nope", "--episodes", "1", "--horizon", "4"])


def _snapshot():
    snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    snaps = sorted(snap.glob("*")) if snap.exists() else []
    return snaps[0] if snaps else None


def test_w8_labels_are_single_tokens_with_contiguous_ids():
    transformers = pytest.importorskip("transformers")
    snap = _snapshot()
    if snap is None:
        pytest.skip("no local Qwen3-0.6B snapshot")
    from agent_bullwhip.decision_readout import render_prompt, resolve_label_ids
    tok = transformers.AutoTokenizer.from_pretrained(str(snap))
    sample = render_prompt(CANONICAL_CTX, tok, grid_id="w8_g26")
    ids = resolve_label_ids(tok, sample, grid_id="w8_g26")
    assert list(ids) == list(string.ascii_uppercase)
    assert list(ids.values()) == list(range(32, 58))
    assert prompt_hash(sample) == prompt_hash(render_prompt_str(CANONICAL_CTX, grid_id="w8_g26"))


def test_committed_grid_artifact_matches_code():
    art = Path(__file__).resolve().parents[1] / "results" / "wave8" / "grid_w8_g26.json"
    if not art.exists():
        pytest.skip("artifact not generated in this checkout (python3 scripts/wave8_grid.py)")
    a = json.loads(art.read_text())
    assert a["values"] == G26 and a["labels"] == list(string.ascii_uppercase)
    assert a["prompt_sha256_string_renderer"] == prompt_hash(
        render_prompt_str(CANONICAL_CTX, grid_id="w8_g26"))
    assert a["instruction"] == instruction("w8_g26")


G26B = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        36, 44, 54, 64, 78, 94, 114, 136, 166, 200]


def test_w8_g26b_is_the_generated_f0_revision_and_the_staged_grid():
    from agent_bullwhip.decision_readout import W8_STAGED_GRID, make_w8_g26b
    g = get_grid("w8_g26b")
    assert list(g.values) == G26B and make_w8_g26b().values == g.values
    assert W8_STAGED_GRID == "w8_g26b"
    # generation rule: geometric ladder 30 -> 200, even integers, strictly increasing, last = top
    tail = G26B[16:]
    assert all(b > a for a, b in zip([30] + tail, tail)) and tail[-1] == 200
    assert tail[:3] == [2 * round(30 * (200 / 30) ** (k / 10) / 2) for k in (1, 2, 3)]
    # same head as w8_g26 (letters A..P keep the wave-7 values); only the tail differs
    assert list(g.values[:16]) == list(get_grid("w8_g26").values[:16]) != G26B[16:]
    assert prompt_hash(render_prompt_str(CANONICAL_CTX, grid_id="w8_g26b")) != prompt_hash(
        render_prompt_str(CANONICAL_CTX, grid_id="w8_g26"))
    assert make_w8_g26b(160).values[-1] == 160


def test_prereg_grid_f0_failure_is_on_record_and_reproducible_from_the_saved_J():
    """The pre-registered w8_g26 FAILED F0 on 1,000 states per regime (2026-10-07); the saved
    reference-grid J rows reproduce the verdict and the g26b revision's selection-set reads."""
    from agent_bullwhip import decision_gates as G
    root = Path(__file__).resolve().parents[1] / "results" / "wave8"
    rec, Jf = root / "f0_w8_g26_prereg.json", root / "f0_w8_g26_prereg.json.J.json"
    if not (rec.exists() and Jf.exists()):
        pytest.skip("F0 record not in this checkout")
    saved = json.loads(rec.read_text())
    assert saved["verdict"]["pass"] is False
    J = json.loads(Jf.read_text())
    ref = list(range(0, 201, 2))
    st = {r: G.f0_stats([x["J"] for x in J[r]], list(get_grid("w8_g26").values), ref) for r in J}
    assert not G.f0_gate(st)["pass"]
    for r in st:
        assert st[r]["top_argmax_share"] == pytest.approx(saved["stats"][r]["top_argmax_share"], abs=2e-3)
    st_b = {r: G.f0_stats([x["J"] for x in J[r]], G26B, ref) for r in J}
    assert G.f0_gate(st_b)["pass"]          # selection-set read only; the gate is the fresh-seed F0


def test_r3_variant_grids_follow_the_spec_descriptions():
    b, c = list(get_grid("w8_r3b").values), list(get_grid("w8_r3c").values)
    assert len(b) == len(c) == 26
    assert b[:2] == [0, 2] and b[2:19] == list(range(4, 21)) and b[-1] == 200      # resolution 1 on 4..20
    assert c[:24] == G26B[:24] and c[-2:] == [190, 240]                            # widened tail
    for g in (b, c):
        assert all(y > x for x, y in zip(g, g[1:]))
    assert prompt_hash(render_prompt_str(CANONICAL_CTX, grid_id="w8_r3b")).startswith(
        prompt_hash(render_prompt_str(CANONICAL_CTX, grid_id="w8_r3b"))[:8])
