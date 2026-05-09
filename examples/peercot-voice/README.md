# Voice PeerCoT

A voice discussion bot where two AI agents debate a topic in real time,
grounded by live web search via the [Keenable API](https://keenable.ai/docs/api).

Inspired by **PeerCoT** (Chaturvedi et al., ICLR 2026 Workshop on Logical
Reasoning of LLMs) — a structured multi-agent Chain-of-Thought collaboration
protocol. The paper's Expert and Curious Student personas are adapted here
into two podcast-style speakers with distinct voices:

| Persona | Temperature | Cartesia voice | Role |
| --- | --- | --- | --- |
| **Expert** | 0.3 | Narrator (male) | Precise, analytical, evidence-driven |
| **Curious Thinker** | 0.9 | British Reading Lady | Exploratory, probing, creative |

## How it works

1. On client connect, Keenable searches the web for the chosen topic.
2. The Expert delivers an opening analysis grounded in search results.
3. The Curious Thinker responds — sometimes agreeing, sometimes pushing back.
4. They alternate for *N* turns (default 4), each seeing the other's
   previous responses.
5. The Expert wraps up with a closing remark.

All speech is streamed through Cartesia TTS with voice switching between
speakers.

## Required environment variables

| Variable | Where to get it |
| --- | --- |
| `KEENABLE_API_KEY` | https://keenable.ai/console |
| `CARTESIA_API_KEY` | https://play.cartesia.ai |
| One of `XAI_API_KEY`, `GROQ_API_KEY`, or `OPENAI_API_KEY` | See below |

**LLM provider priority:** xAI Grok > Groq > OpenAI. The bot auto-detects
which key is set and picks the right base URL and default model.

## Optional environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `PEERCOT_TOPIC` | `the future of AI agents` | Discussion topic |
| `PEERCOT_MODEL` | auto-detected | LLM model override |
| `PEERCOT_TURNS` | `4` | Number of Expert/Student exchanges |

## Run

From the repo root:

```bash
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local
PEERCOT_TOPIC="quantum computing breakthroughs" \
    uv run python examples/peercot-voice/bot.py
```

The pipecat runner prints a connection URL. Open it in a browser and listen.

## Switching LLM providers

**xAI Grok** (default when `XAI_API_KEY` is set):

```bash
export XAI_API_KEY=...
export PEERCOT_MODEL=grok-3-fast   # or grok-3, grok-3-mini
```

**Groq + Llama 3.3 70B** (OSS-friendly):

```bash
export GROQ_API_KEY=...
export PEERCOT_MODEL=llama-3.3-70b-versatile
```

**OpenAI**:

```bash
export OPENAI_API_KEY=...
export PEERCOT_MODEL=gpt-4o
```

## Architecture

```
                          +------------------+
                          |  Keenable Search |
                          +--------+---------+
                                   |
                                   v
+-----------+    +-------------------------------------+    +-----------+
| LLM (Expert) <--->  Orchestration Loop  <---> LLM (Student) |
+-----------+    +-------------------------------------+    +-----------+
                          |                  |
                  TTSUpdateSettings   TTSSpeakFrame
                          |                  |
                          v                  v
                   +------------+     +-------------+
                   | Cartesia TTS | --> | Transport   |
                   | (voice swap) |     | (WebRTC)    |
                   +------------+     +-------------+
                                            |
                                         Listener
```

The pipeline itself is minimal (`TTS -> transport.output()`). The
orchestration loop calls the LLM API directly (via the `openai` SDK) and
queues `TTSSpeakFrame` + `TTSUpdateSettingsFrame` pairs to alternate voices.

## References

- Chaturvedi, I., Llewellyn-Jones, R., & Schaffer, S. R. (2026). *PeerCoT:
  Structured Multi-Agent Chain-of-Thought Collaboration for Error
  Localization in LLM Reasoning.* ICLR 2026 Workshop on Logical Reasoning
  of Large Language Models.
- [Keenable API docs](https://keenable.ai/docs/api)
- [Grok x Keenable latency lab](https://grok.keenable.ai)
