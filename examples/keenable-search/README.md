# Keenable Web Search bot

A voice bot that grounds its answers with live web search via the
[Keenable API](https://keenable.ai/docs/api).

The LLM is configured with two tools, `keenable_search` and `keenable_fetch`.
Whenever the user asks something that benefits from up-to-date information
(news, current events, recent releases, etc.), the model calls `keenable_search`,
gets back ranked search results, optionally calls `keenable_fetch` to read a
result in full, and speaks a grounded answer. The model can also narrow a
search per call with `site` and publication/index date filters.

By default the bot runs **fully on free/local services**: local Whisper (STT) and
local Kokoro (TTS) — both download a small model on first run and need no API key —
plus Keenable's keyless public endpoints. The only key you need is for the LLM.

## Required environment variables

| Variable | Where to get it |
| --- | --- |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys (or use the free Groq option below) |

`KEENABLE_API_KEY` (https://keenable.ai/console) is **optional** — without it the
bot uses Keenable's free, keyless public endpoints. Set it to use the
authenticated endpoints.

`DEEPGRAM_API_KEY` / `CARTESIA_API_KEY` are **only** needed if you switch from the
local Whisper/Kokoro defaults to premium cloud STT/TTS (see the commented block in
`bot.py`).

## Run

From the repo root, install the local STT/TTS engines + an LLM provider:

```bash
uv sync --extra openai --extra whisper --extra kokoro --extra silero
uv run python examples/keenable-search/bot.py
```

(Or `uv sync --group dev --all-extras --no-extra gstreamer --no-extra local` to get
everything, including the premium Deepgram/Cartesia services.)

The pipecat runner will print a connection URL. Open it in a browser, allow
microphone access, and ask something current, e.g. "What are the latest
pipecat releases?" The first run is slower while Whisper and Kokoro download their
models.

## Switching to an OSS-friendly model (Groq + Llama 3.3 70B)

Groq inherits OpenAI-style tool calling and runs Llama 3.3 70B with full
function support, so the only changes are the import, the class, and the
API key. In `bot.py` swap:

```python
from pipecat.services.openai.llm import OpenAILLMService
# ...
llm = OpenAILLMService(
    api_key=os.environ["OPENAI_API_KEY"],
    system_instruction=SYSTEM_INSTRUCTION,
)
```

for the commented-out Groq block already present in the file, then export
`GROQ_API_KEY` instead of `OPENAI_API_KEY`. Other tool-calling providers
(Together AI, Fireworks, Cerebras, DeepSeek, Anthropic, Ollama, etc.) work
the same way.

## API reference

See [keenable.ai/docs/api](https://keenable.ai/docs/api) for the underlying
REST endpoints. The pipecat client lives at
[`src/pipecat/services/keenable/search.py`](../../src/pipecat/services/keenable/search.py).
