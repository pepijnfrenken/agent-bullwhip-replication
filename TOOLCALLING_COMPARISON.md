# Tool-Calling Model Comparison — Results (CommandCode, 2026-09-01)

Run: `run_model_comparison.py --models z-ai/glm-5.3-flash,deepseek/deepseek-v4-flash,Qwen/Qwen3.8-Flash --decisions 6`
Endpoint: CommandCode (api.commandcode.ai/provider/v1), OpenAI-compat, streaming.
Harness: ToolAgent (run_python exec tool), force_tool first round, max 3 tool rounds.
Checkpoints: `results/toolagent_compare/<model-id-sanitized>.jsonl` (successful decisions only count as done; failed ones retried on re-run).

## Final table (6/6 decisions per model, all filled)

| Model | tool_call_rate | code_error_rate | retry_after_error_rate | parse_fail_rate | errors | avg_calls/dec |
|---|---|---|---|---|---|---|
| z-ai/glm-5.3-flash       | 1.0 | 0.071 | 1.0 | 0.333 | 1 (transient stall, recovered) | 2.33 |
| deepseek/deepseek-v4-flash | 1.0 | 0.4   | 0.5 | 0.0   | 0 | 1.67 |
| Qwen/Qwen3.8-Flash      | 1.0 | 0.3   | 1.0 | 0.5   | 1 (429, recovered) | 3.33 |

Ranking (by code-error, then parse-fail): GLM-5.3-Flash > Qwen > DeepSeek
Honest read: GLM wins on code quality + recovery; DeepSeek wins on answer reliability + speed; Qwen is the weakest (confirms paper baseline).

## Cost
Total spend for the full 18-decision run: **< $1**. GLM-5.3-Flash alone would be ~$0.36 for 576 decisions (a full game).

## Notes
- GLM's parse-fails are empty-final-text after deep thinking (4 tool calls), not broken tool calls.
- DeepSeek sometimes emits DSML-style `<DSML>` tags instead of clean JSON tool calls (2/6 in first pass), but its final answers were always parseable.
- CommandCode rate-limits bursts: use PROBE_CALL_DELAY=30 (default) between decisions.
- One stream-stall (GLM t=10) and one 429 (Qwen t=30) both recovered on the retry-once path.

## Verdict
Use **z-ai/glm-5.3-flash** for the tool-calling arm (cheapest good option: $0.15/$0.50 per M, lowest code-error, best recovery). If answer-reliability matters more than code quality, deepseek/deepseek-v4-flash is the fallback.
