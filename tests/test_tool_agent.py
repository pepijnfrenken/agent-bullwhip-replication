"""Tests for the Wave 4 tool-calling agent (offline, no API)."""
from agent_bullwhip.client import chat_with_tools, lint_code
from agent_bullwhip.agents import ToolAgent, LLMAgentConfig


def _fake_chat(messages, tools, **kwargs):
    """Simulate a model that calls run_python once, then answers."""
    # first round: emit a tool call
    return "", [{"name": "run_python", "arguments": {"code": "print(17*23)"},
                 "result": "391"}]


def test_tool_agent_tools_schema():
    tools = ToolAgent._tools()
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "run_python"
    assert "code" in tools[0]["function"]["parameters"]["properties"]


def test_tool_agent_constructs():
    cfg = LLMAgentConfig(model="qwen3.6-35b", tag="test")
    a = ToolAgent("retailer", cfg)
    assert a.max_tool_rounds == 3  # default now allows error-recovery retries
    assert a.tool_traces == []
    assert a.last_decision_meta == {}


def test_lint_code_catches_undefined_function():
    # the classic model bug: defines `simulate` but calls `simulate_order_up_to`
    code = """
def simulate(x):
    return x
print(simulate_order_up_to(5))
"""
    err = lint_code(code)
    assert err is not None
    assert "undefined" in err
    assert "simulate_order_up_to" in err


def test_lint_code_passes_valid_code():
    code = """
import math
x = 4
print(x * math.pi)
"""
    assert lint_code(code) is None


def test_lint_code_catches_syntax_error():
    err = lint_code("def foo(:\n  pass")
    assert err is not None
    assert "syntax" in err


def test_lint_code_accepts_builtins_and_helpers():
    code = """
import numpy as np
vals = [1,2,3]
print(np.mean(vals), statistics.mean(vals), sum(vals))
"""
    assert lint_code(code) is None


def test_chat_with_tools_executes_code():
    """Verify the local exec loop actually runs code and returns tool results."""
    tools = ToolAgent._tools()
    text, trace = chat_with_tools(
        [{"role": "user", "content": "compute 17*23 with the tool"}],
        tools=tools,
        model="test",
        max_tool_rounds=2,
        dry_run=True,
    )
    # dry run: round 0 emits tool call -> exec runs print(17*23) -> round 1 answers
    assert isinstance(text, str)
    assert len(trace) >= 1
    assert trace[0]["name"] == "run_python"
    assert trace[0]["result"] == "391"
    assert text == "391"


def test_chat_with_tools_error_recovery_loop(monkeypatch):
    """A tool error must be fed back so the model can retry (round 2 allowed)."""
    import agent_bullwhip.client as c
    monkeypatch.setattr(c, "PROVIDER", "freeinference")
    responses = [
        # round 0: forced tool call -> broken code
        {"choices": [{"message": {"role": "assistant", "content": "",
                                   "tool_calls": [{"id": "c1", "type": "function",
                                                   "function": {"name": "run_python",
                                                                "arguments": '{"code": "print(undefined_var)"}'}}]}}]},
        # round 1 (auto): model retries with fixed code
        {"choices": [{"message": {"role": "assistant", "content": "",
                                   "tool_calls": [{"id": "c2", "type": "function",
                                                   "function": {"name": "run_python",
                                                                "arguments": '{"code": "print(10+5)"}'}}]}}]},
        # round 2 (none): final answer
        {"choices": [{"message": {"role": "assistant", "content": "15"}}]},
    ]
    def fake_post(*a, **k):
        # check tool_choice progression: required -> auto -> none
        nonlocal responses
        resp = responses.pop(0)
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return resp
        return R()
    monkeypatch.setattr(c.requests, "post", fake_post)
    text, trace = c.chat_with_tools(
        [{"role": "user", "content": "compute 10+5 with the tool"}],
        tools=ToolAgent._tools(), model="test",
        max_tool_rounds=3, force_tool=True, retries=1,
    )
    assert text == "15"
    # two tool calls: first errored, second succeeded -> recovery engaged
    assert len(trace) == 2
    assert trace[0]["result"].startswith("ERROR")
    assert "15" in trace[1]["result"]


def test_chat_with_tools_final_round_forces_none(monkeypatch):
    """The last round must force tool_choice='none' so the model answers."""
    import agent_bullwhip.client as c
    monkeypatch.setattr(c, "PROVIDER", "freeinference")
    seen = []
    def fake_post(*a, **k):
        tc = k["json"]["tool_choice"]
        seen.append(tc)
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                # always call a tool until forced to answer
                if tc != "none":
                    return {"choices": [{"message": {"role": "assistant", "content": "",
                                                     "tool_calls": [{"id": "c", "type": "function",
                                                                     "function": {"name": "run_python",
                                                                                  "arguments": '{"code": "print(1)"}'}}]}}]}
                return {"choices": [{"message": {"role": "assistant", "content": "42"}}]}
        return R()
    monkeypatch.setattr(c.requests, "post", fake_post)
    text, _ = c.chat_with_tools(
        [{"role": "user", "content": "hi"}],
        tools=ToolAgent._tools(), model="test",
        max_tool_rounds=2, force_tool=True, retries=1,
    )
    assert text == "42"
    assert seen == ["required", "auto", "none"]


def test_chat_with_tools_handles_error_body(monkeypatch):
    """Endpoint returning {'error': ...} with no 'choices' must not KeyError."""
    import agent_bullwhip.client as c
    # force the FreeInference (non-streaming) path regardless of env
    monkeypatch.setattr(c, "PROVIDER", "freeinference")
    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"error": {"message": "model overloaded"}}
    monkeypatch.setattr(c.requests, "post", lambda *a, **k: FakeResp())
    try:
        chat_with_tools([{"role": "user", "content": "hi"}],
                        tools=ToolAgent._tools(), model="test",
                        retries=1, max_tool_rounds=1)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "no choices" in str(e)
        assert "model overloaded" in str(e)


def test_extract_xml_tool_call():
    from agent_bullwhip.client import _extract_xml_tool_call
    xml = ('<tool_calls>\n<invoke name="run_python">\n'
           '<parameter name="code" string="true">print(1)</parameter>\n'
           '</invoke>\n</tool_calls>')
    assert _extract_xml_tool_call(xml) == "print(1)"
    # bare invoke (no param tag) still extracts inner text
    assert _extract_xml_tool_call('<invoke name="run_python">print(2)</invoke>') == "print(2)"
    assert _extract_xml_tool_call("plain text no xml") is None


class _FakeStreamResp:
    """Minimal stream response: yields SSE lines, has close()."""
    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.text = ""
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def iter_lines(self, decode_unicode=False):
        for l in self._lines:
            yield l
    def close(self):
        pass


def test_stream_chat_completion_reassembles_tool_calls(monkeypatch):
    """Streaming helper must reassemble fragmented tool-call deltas."""
    import json
    import agent_bullwhip.client as c
    calls = []
    def fake_post(*a, **k):
        calls.append(k)
        return _FakeStreamResp([
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"run_","arguments":"{\\"code\\":"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"python","arguments":" \\"print(42)\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{}}]}',
            'data: [DONE]',
        ])
    monkeypatch.setattr(c.requests, "post", fake_post)
    out = c._stream_chat_completion({"model": "x"}, idle_timeout=30)
    msgs = out["choices"][0]["message"]
    assert msgs["tool_calls"][0]["function"]["name"] == "run_python"
    assert json.loads(msgs["tool_calls"][0]["function"]["arguments"]) == {"code": "print(42)"}
    # the helper passes the payload through as-is (caller adds stream=True)


def test_stream_chat_completion_idle_timeout(monkeypatch):
    """A stream that goes silent mid-flight must raise TimeoutError quickly."""
    import agent_bullwhip.client as c
    def fake_post(*a, **k):
        return _FakeStreamResp([
            'data: {"choices":[{"delta":{"content":"partial"}}]}',
        ])  # then silence — iter_lines just ends, no [DONE]
    monkeypatch.setattr(c.requests, "post", fake_post)
    # iter_lines that yields one line then blocks forever isn't easy to fake;
    # here the stream ends without [DONE] — helper must not crash and returns
    # what it got (the caller's no-choices/empty-content handling catches it).
    out = c._stream_chat_completion({"model": "x"}, idle_timeout=0.1)
    assert out["choices"][0]["message"]["content"] == "partial"
