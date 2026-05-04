# Keenable Web Search bot

A voice bot that grounds its answers with live web search via the
[Keenable API](https://keenable.ai/docs/api).

The LLM is configured with a single tool, `keenable_search`. Whenever the
user asks something that benefits from up-to-date information (news, current
events, recent releases, etc.), the model calls the tool, gets back ranked
search results, and speaks a grounded answer.

## Required environment variables

| Variable | Where to get it |
| --- | --- |
| `KEENABLE_API_KEY` | https://keenable.ai/console |
| `OPENAI_API_KEY` | https://platform.openai.com/api-keys |
| `DEEPGRAM_API_KEY` | https://console.deepgram.com |
| `CARTESIA_API_KEY` | https://play.cartesia.ai |

## Run

From the repo root:

```bash
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local
uv run python examples/keenable-search/bot.py
```

The pipecat runner will print a connection URL. Open it in a browser, allow
microphone access, and ask something current, e.g. "What are the latest
pipecat releases?"

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
