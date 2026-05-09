# Voice PeerCoT

A voice discussion bot where two AI agents debate a topic in real time,
grounded by live web search via the [Keenable API](https://keenable.ai/docs/api).

Inspired by **PeerCoT** (Chaturvedi et al., ICLR 2026 Workshop on Logical
Reasoning of LLMs) — a structured multi-agent Chain-of-Thought collaboration
protocol. The paper's Expert and Curious Student personas are adapted here
into two podcast-style speakers with distinct Grok voices:

| Persona | Temperature | Grok voice | Role |
| --- | --- | --- | --- |
| **Expert** | 0.3 | Rex | Precise, analytical, evidence-driven |
| **Curious Thinker** | 0.9 | Eve | Exploratory, probing, creative |

## How it works

1. If no topic is set, the bot pulls the **top trending question from
   Polymarket** as the discussion topic.
2. **Keenable searches** the topic and **fetches full article content** from
   all result URLs via `/v1/fetch`.
3. The Expert delivers an opening analysis grounded in the articles,
   naturally citing news sources by name.
4. The Curious Thinker responds — sometimes agreeing, sometimes pushing back,
   always grounded in the same articles.
5. They alternate for *N* turns (default 4), each seeing the other's
   previous responses.
6. The Expert wraps up with a closing remark.

All speech uses xAI TTS with voice switching (Rex/Eve) between speakers.
Supports Hindi (Hinglish) via `PEERCOT_LANGUAGE=hi`.

## Required environment variables

Only **two** API keys needed:

| Variable | Where to get it |
| --- | --- |
| `XAI_API_KEY` | https://console.x.ai |
| `KEENABLE_API_KEY` | https://keenable.ai/console |

The `XAI_API_KEY` is used for both the LLM (Grok) and TTS (xAI voices).

## Optional environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `PEERCOT_TOPIC` | *(Polymarket trending)* | Discussion topic. If unset, pulls from Polymarket |
| `PEERCOT_MODEL` | `grok-3-fast` | LLM model override |
| `PEERCOT_TURNS` | `4` | Number of Expert/Student exchanges |
| `PEERCOT_LANGUAGE` | `en` | Language: `en` (English) or `hi` (Hindi/Hinglish) |

## Run

```bash
git clone https://github.com/keenableai/pipecat.git
cd pipecat
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local

export XAI_API_KEY=...
export KEENABLE_API_KEY=...

# Auto-pick trending Polymarket topic:
uv run python examples/peercot-voice/bot.py -t webrtc

# Or set a specific topic:
PEERCOT_TOPIC="2026 NBA Champion" uv run python examples/peercot-voice/bot.py -t webrtc

# Hindi mode:
PEERCOT_TOPIC="Modi Indian elections 2026" PEERCOT_LANGUAGE=hi \
    uv run python examples/peercot-voice/bot.py -t webrtc
```

Open **http://localhost:7860** in your browser and listen.

## Architecture

```
Polymarket API ──(topic)──> Keenable Search + Fetch
                                    |
                              Full articles
                                    |
                                    v
    Grok (Expert, temp 0.3) <── Orchestration Loop ──> Grok (Student, temp 0.9)
                                    |
                          TTSUpdateSettings + TTSSpeakFrame
                                    |
                                    v
                             xAI TTS (Rex / Eve)
                                    |
                                    v
                          WebRTC Transport ──> Browser
```

Minimal pipecat pipeline (`TTS -> transport.output()`). The orchestration
loop calls the Grok API directly (via the `openai` SDK), queues speech
frames to alternate voices. Keenable fetches full page content from each
search result URL so agents discuss real article text, not just snippets.

## References

- Chaturvedi, I., Llewellyn-Jones, R., & Schaffer, S. R. (2026). *PeerCoT:
  Structured Multi-Agent Chain-of-Thought Collaboration for Error
  Localization in LLM Reasoning.* ICLR 2026 Workshop on Logical Reasoning
  of Large Language Models.
- [Keenable API docs](https://keenable.ai/docs/api)
- [Grok x Keenable latency lab](https://grok.keenable.ai)
- [Polymarket API](https://docs.polymarket.com)
