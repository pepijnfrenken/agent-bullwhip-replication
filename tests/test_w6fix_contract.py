"""WAVE6-FIX (docs/waves/WAVE6-FIX_SPEC.md): contract prompt, strict parser,
parse-aware reward masking, arm-config diff, legacy bit-identity, stub loop."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from agent_bullwhip import agents, contract
from agent_bullwhip.contract import (CONTRACT_TAIL, LEGACY_REPLY, build_contract_prompt,
                                     count_digit_tokens, has_loose_json, parse_contract,
                                     render_chat_nothink)
from agent_bullwhip.grpo_env import combine_parse_aware, group_normalize
from agent_bullwhip.grpo_train import StepResult, collect_step, parse_args, run_loop
from agent_bullwhip.qwen_chat import render_nothink

FIX = Path(__file__).parent / "fixtures" / "w6fix_collect_legacy_golden.json"
CTX = {"role": "wholesaler", "t": 3, "on_hand": 5, "backlog": 0, "outstanding": 8,
       "incoming_last": 4, "incoming_now": None, "last_orders": [4, 4, 5]}


# ----------------------------- parser ------------------------------------- #
@pytest.mark.parametrize("text,value,exact", [
    ('{"order_quantity": 12}', 12, True),
    ('{"order_quantity": 5}', 5, True),                       # the paper's own example
    ('{"order_quantity": 12}\nThanks', 12, False),
    ('  {"order_quantity":7}  ', 7, True),
    ('\n{"order_quantity": 0}\n', 0, True),
    ('{"order_quantity": 9999}', 9999, True),
    ('{"order_quantity": 007}', 7, True),                      # pinned: leading zeros parse
])
def test_parse_contract_accepts(text, value, exact):
    r = parse_contract(text)
    assert (r.value, r.exact) == (value, exact)


@pytest.mark.parametrize("text", [
    '{"order_quantity": -1}', '{"order_quantity": 1.5}', '{"order_quantity": 12345}',
    '```json\n{"order_quantity": 5}\n```', 'Sure! {"order_quantity": 5}', "", None,
    '{"order_quantity": \u0663}',                              # Arabic-Indic digit
    '{"order_quantity": 7} thanks', '{"order_quantity": 7}.', "12", '"order_quantity": 7',
    '{"order_qty": 7}', '{"order_quantity": "7"}', '\n\n7\n{"order_quantity": 7}',
])
def test_parse_contract_rejects(text):
    assert parse_contract(text) == contract.ParseResult(value=None, exact=False)


def test_loose_json_is_wider_than_contract():
    assert has_loose_json('Sure! {"order_quantity": 5}')
    assert not has_loose_json("order 5")


# ----------------------------- prompt ------------------------------------- #
def test_legacy_reply_is_the_real_tail_of_build_prompt():
    assert agents.build_prompt("wholesaler", CTX).endswith(LEGACY_REPLY)


def test_build_contract_prompt_shape():
    p = build_contract_prompt("wholesaler", CTX)
    assert p.startswith("<|im_start|>user\nYou are the WHOLESALER")
    assert p.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert p.count(CONTRACT_TAIL) == 1
    assert LEGACY_REPLY not in p
    assert CONTRACT_TAIL.startswith("\u2014" * 9 + "\nYour Task:")
    # only the answer request changed: state text before the tail is build_prompt's
    base = agents.build_prompt("wholesaler", CTX)
    assert base[: -len(LEGACY_REPLY)] in p


def test_render_chat_nothink_matches_transformers_template():
    transformers = pytest.importorskip("transformers")
    snap = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    snaps = list(snap.glob("*")) if snap.exists() else []
    if not snaps:
        pytest.skip("no local Qwen3-0.6B snapshot")
    tok = transformers.AutoTokenizer.from_pretrained(str(snaps[0]))
    user = agents.build_prompt("wholesaler", CTX)[: -len(LEGACY_REPLY)] + CONTRACT_TAIL
    ref = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    assert render_chat_nothink(user) == ref


def test_render_nothink_rejects_unsupported_roles():
    with pytest.raises(ValueError):
        render_nothink([{"role": "assistant", "content": "x"}])


def test_count_digit_tokens():
    vocab = {1: '{"', 2: " ", 3: "1", 4: "2", 5: "}"}
    d, tot = count_digit_tokens([[1, 2, 3, 4, 5], [3, 3], None], [True, False, True],
                                lambda i: vocab[i])
    assert (d, tot) == (2, 7)


# ----------------------------- A2 advantage -------------------------------- #
def test_combine_parse_aware_hand_built_grid():
    # 3 episodes x 2 agents x 2 weeks. Costs differ per episode (cost adv != 0).
    rewards = [{k: [-10.0, -4.0] for k in (0, 1)},
               {k: [-20.0, -8.0] for k in (0, 1)},
               {k: [-30.0, -12.0] for k in (0, 1)}]
    adv = group_normalize(rewards)
    parsed = [  # f[g][agent][t]
        {0: [True, True], 1: [True, False]},     # ep0
        {0: [True, False], 1: [True, False]},    # ep1
        {0: [False, False], 1: [True, False]},   # ep2
    ]
    comb, fmt = combine_parse_aware(adv, parsed, w=1.0)
    # cell (agent0, week0): f = [1,1,0] -> z-scores with population std + 1e-4
    m = 2 / 3
    sd = ((2 * (1 - m) ** 2 + m ** 2) / 3) ** 0.5 + 1e-4
    assert fmt[0][0][0] == pytest.approx((1 - m) / sd)
    assert fmt[2][0][0] == pytest.approx((0 - m) / sd)
    # all-parsed cell (agent1, week0) and all-unparsed cell (agent*, week1 agent1) -> 0
    assert [fmt[g][1][0] for g in range(3)] == [0.0, 0.0, 0.0]
    assert [fmt[g][1][1] for g in range(3)] == [0.0, 0.0, 0.0]
    # mask: unparsed row loses its cost advantage, keeps only the format term
    assert comb[2][0][0] == pytest.approx(0.0 + 1.0 * fmt[2][0][0])
    # parsed row keeps cost advantage + format term
    assert comb[0][0][0] == pytest.approx(adv[0][0][0] + fmt[0][0][0])
    # fully parsed cell: pure cost advantage
    assert comb[1][1][0] == pytest.approx(adv[1][1][0])
    # fully unparsed cell (agent1, week1): masked cost + zero format term = 0
    assert [comb[g][1][1] for g in range(3)] == [0.0, 0.0, 0.0]
    # w scales only the format term
    comb2, _ = combine_parse_aware(adv, parsed, w=2.0)
    assert comb2[0][0][0] == pytest.approx(adv[0][0][0] + 2.0 * fmt[0][0][0])


# ----------------------------- arm config ---------------------------------- #
BASE = ["--out", "results/grpo_a0"]
ARMS = {
    "A0": ["--steps", "50", "--dump-completions-every", "10"],
    "A0b": ["--steps", "25", "--max-new", "24", "--dump-completions-every", "10"],
    "A1": ["--steps", "100", "--prompt-format", "chat_contract", "--max-new", "24",
           "--dump-completions-every", "10"],
    "A2": ["--steps", "100", "--prompt-format", "chat_contract", "--max-new", "24",
           "--parse-aware-reward", "1", "--fmt-weight", "1.0",
           "--dump-completions-every", "10"],
}


def _diff(a, b):
    va, vb = vars(a), vars(b)
    return {k: (va[k], vb[k]) for k in va if va[k] != vb[k]}


def test_arm_config_diff_against_a0_base():
    a0_base = parse_args(BASE)
    assert a0_base.prompt_format == "raw" and a0_base.max_new == 8
    assert a0_base.parse_aware_reward == 0 and a0_base.dump_completions_every == 0
    exp = {
        "A0": {"steps", "dump_completions_every"},
        "A0b": {"steps", "max_new", "dump_completions_every"},
        "A1": {"steps", "prompt_format", "max_new", "dump_completions_every"},
        "A2": {"steps", "prompt_format", "max_new", "parse_aware_reward",
               "dump_completions_every"},   # fmt_weight 1.0 == default
    }
    for arm, flags in ARMS.items():
        assert set(_diff(a0_base, parse_args(BASE + flags))) == exp[arm], arm
    # A1 and A2 differ from each other only in the reward flag
    assert set(_diff(parse_args(BASE + ARMS["A1"]), parse_args(BASE + ARMS["A2"]))) == {
        "parse_aware_reward"}


def test_parse_aware_requires_contract_prompt():
    with pytest.raises(SystemExit):
        parse_args(BASE + ["--parse-aware-reward", "1"])
    with pytest.raises(SystemExit):
        parse_args(BASE + ["--max-new", "0"])


# ----------------------------- legacy bit-identity ------------------------- #
class _Stub8:
    def rollout(self, prompts, temperature=1.0, max_new=8):
        return ["8"] * len(prompts), None, None


class _Alt:
    def __init__(self):
        self.i = 0

    def rollout(self, prompts, temperature=1.0, max_new=8):
        out = []
        for _ in prompts:
            out.append("8" if (self.i // 4) % 2 == 0 else "9")
            self.i += 1
        return out, None, None


def test_collect_step_defaults_are_bit_identical_to_pre_change_golden():
    gold = json.loads(FIX.read_text())
    for pname, mk in (("stub8", _Stub8), ("alt", _Alt)):
        for s in (12345, 2):
            for c in ("poisson", "truncnorm"):
                r = collect_step(mk(), G=8, curriculum=c, seed=s, rollout_batch=16)
                g = gold[f"{pname}|{s}|{c}"]
                assert hashlib.sha256("\x00".join(r.prompts).encode()).hexdigest() == g["prompts_sha256"]
                assert r.completions == g["completions"] and r.orders == g["orders"]
                assert r.costs == g["costs"] and r.advantages == g["advantages"]
                assert r.completion_ids == g["ids"] and r.old_logprobs == g["old_logprobs"]
                assert r.adv_cost == r.advantages and r.adv_fmt == []


def test_golden_first_prompt_shas():
    gold = json.loads(FIX.read_text())
    rng = random.Random(12345)
    seed = rng.randint(0, 2**31)
    a0 = collect_step(_Stub8(), G=8, curriculum="poisson", seed=seed)
    assert hashlib.sha256(a0.prompts[0].encode()).hexdigest() == gold["A0_FIRST_PROMPT_SHA"]
    a1 = collect_step(_Stub8(), G=8, curriculum="poisson", seed=seed,
                      prompt_format="chat_contract", max_new=24)
    assert hashlib.sha256(a1.prompts[0].encode()).hexdigest() == gold["A1_FIRST_PROMPT_SHA"]


# ----------------------------- stub loop ----------------------------------- #
class _ComplianceStub:
    """Compliant (contract JSON) for episodes with (i//4)%2==0 when rate=.5, all/none else."""

    def __init__(self, rate):
        self.rate, self.i, self.max_new_seen = rate, 0, set()
        self.tok = None

    def rollout(self, prompts, temperature=1.0, max_new=8):
        self.max_new_seen.add(max_new)
        out = []
        for _ in prompts:
            ok = self.rate == 1.0 or (self.rate == 0.5 and (self.i // 4) % 2 == 0)
            out.append('{"order_quantity": 8}' if ok else "well, let me think about it")
            self.i += 1
        return out, None, None

    def grpo_step(self, *a, **kw):
        return StepResult(loss=0.0)


def _run(tmp_path, rate, extra):
    args = parse_args(["--steps", "3", "--group", "8", "--out", str(tmp_path)] + extra)
    pol = _ComplianceStub(rate)
    run_loop(pol, args, tmp_path / "train_log.jsonl", tmp_path / "checkpoint",
             has_torch=lambda: False)
    recs = [json.loads(l) for l in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    return pol, [r for r in recs if "step" in r and "event" not in r]


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_stub_loop_a2_telemetry(tmp_path, rate):
    pol, recs = _run(tmp_path, rate, ["--prompt-format", "chat_contract", "--max-new", "24",
                                      "--parse-aware-reward", "1", "--dump-completions-every", "2"])
    assert pol.max_new_seen == {24}
    assert len(recs) == 3
    for r in recs:
        assert r["contract_parse_rate"] == rate
        assert r["contract_exact_rate"] == rate
        assert r["parse_lenient_rate"] == pytest.approx(rate if rate else 0.0)
        if rate in (0.0, 1.0):
            assert r["fmt_adv_share"] == 0.0
        else:
            assert r["fmt_adv_share"] > 0
        if rate == 0.0:
            assert r["cost_rows_frac"] == 0.0
        assert re.fullmatch(r"[0-9a-f]{16}", r["demand_sha"])
        assert len(r["first_prompt_sha256"]) == 64
    d0 = tmp_path / "completions_step0.jsonl"
    rows = [json.loads(l) for l in d0.read_text().splitlines()]
    assert len(rows) == 8 * 4 * 36
    assert set(rows[0]) == {"i", "e", "agent_id", "t", "text", "ids", "order", "parsed",
                            "adv_cost", "adv_fmt", "adv"}
    assert not (tmp_path / "completions_step1.jsonl").exists()
    assert (tmp_path / "completions_step2.jsonl").exists()
    # A2 mask: unparsed rows carry no cost advantage in the combined adv unless fmt != 0
    for row in rows:
        if not row["parsed"]:
            assert row["adv"] == pytest.approx(row["adv_fmt"])


def test_stub_loop_a1_has_no_format_term_and_demand_sha_is_reproducible(tmp_path):
    from agent_bullwhip.grpo_env import sample_curriculum_demand
    pol, recs = _run(tmp_path, 0.5, ["--prompt-format", "chat_contract", "--max-new", "24",
                                     "--seed", "2"])
    assert all(r["fmt_adv_share"] is None for r in recs)
    assert all(r["cost_rows_frac"] > 0 for r in recs)      # junk rows still carry cost adv
    rng = random.Random(2)
    for r in recs:
        d = sample_curriculum_demand(random.Random(rng.randint(0, 2**31)), r["curriculum"], 36)
        assert r["demand_sha"] == hashlib.sha256(",".join(map(str, d)).encode()).hexdigest()[:16]


def test_stub_loop_legacy_flags_keep_wave6_parse_failure_counter(tmp_path):
    pol, recs = _run(tmp_path, 0.0, [])
    assert pol.max_new_seen == {8}
    for r in recs:
        assert r["parse_failure_rate"] == 1.0 and r["parse_lenient_rate"] == 0.0
        assert r["contract_parse_rate"] == 0.0 and r["fmt_adv_share"] is None
    state = json.loads((tmp_path / "run_info.json").read_text()) if (tmp_path / "run_info.json").exists() else {}
    assert state.get("contract_version", "w6fix-1") == "w6fix-1"
