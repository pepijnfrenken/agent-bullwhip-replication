# CommandCode provider setup (2026-09-01)

Key: `COMMANDCODE_API_KEY` in `.env.local` (gitignored) — user_3srz... verified live.

## Endpoint layout (important)
- `https://api.commandcode.ai/provider/v1/models` — full catalog (61 models), auth = Bearer key
- `https://api.commandcode.ai/provider/v1/chat/completions` — OpenAI-compat, works for:
  - gpt-5.6-sol / terra / luna, gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.3-codex
  - deepseek/deepseek-v4-flash(-fast, -vision-exp), deepseek/deepseek-v4-pro
  - Qwen/*, z-ai/glm-5.3-flash, zai-org/GLM-5.x, MiniMaxAI/*, moonshotai/Kimi-K*, xai/grok-4.5/4.6, google/gemini-*, etc.
- **Claude models** (`claude-*`) require the Anthropic Messages shape at `/provider/v1/messages` — they 400 on `/chat/completions`. The repo's `client.py` is OpenAI-compat only, so use the non-Claude models here.

## Verified (2026-09-01)
- `gpt-5.6-sol` via `/chat/completions` → 200, `CC-OK` ✅
- `/models` with the key → full 61-model list ✅

## Repo behavior (agent_bullwhip/client.py)
- If `COMMANDCODE_API_KEY` env is set → provider=commandcode, base=`https://api.commandcode.ai/provider/v1`, no client-side pacing.
- Otherwise → freeinference with key from `.env.local` (or env).
- Default model for commandcode runs: set `COMMANDCODE_MODEL` env or pass `--model`.

## Cost comparison (verified 2026-09-01, commandcode.ai/models)
| Model | Input $/M | Output $/M | Coding score | Verdict |
|---|---|---|---|---|
| **GLM-5.3 Flash** | **$0.15** | **$0.50** | 71.5 | ✅ **cheapest of the good** |
| GPT-5.6 Luna | $0.20 | $1.20 | 71.4 | cheap, slightly weaker |
| DeepSeek V4 Flash | $0.22 | $0.66 | 69.1 | cheap, solid |
| GLM-5.3 | $1.40 | $4.40 | 74.8 | ~10x cost, +3 pts |
| GPT-5.6 Sol | $5.00 | $30.00 | 77.4 | 30x cost, not for this |

Full 576-decision comparison on GLM-5.3-Flash ≈ **$0.36**.
