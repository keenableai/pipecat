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

1. On client connect, Keenable searches the web for the chosen topic.
2. The Expert delivers an opening analysis grounded in search results.
3. The Curious Thinker responds — sometimes agreeing, sometimes pushing back.
4. They alternate for *N* turns (default 4), each seeing the other's
   previous responses.
5. The Expert wraps up with a closing remark.

All speech uses xAI TTS with voice switching (Rex/Eve) between speakers.

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
| `PEERCOT_TOPIC` | `the future of AI agents` | Discussion topic |
| `PEERCOT_MODEL` | `grok-3-fast` | LLM model override |
| `PEERCOT_TURNS` | `4` | Number of Expert/Student exchanges |

## Run

From the repo root:

```bash
uv sync --group dev --all-extras --no-extra gstreamer --no-extra local

export XAI_API_KEY=...
export KEENABLE_API_KEY=...

PEERCOT_TOPIC="quantum computing breakthroughs" \
    uv run python examples/peercot-voice/bot.py -t webrtc
```

Open **http://localhost:7860** in your browser and listen.

## Architecture

```
                          +------------------+
                          |  Keenable Search |
                          +--------+---------+
                                   |
                                   v
+-----------+    +-------------------------------------+    +-----------+
| Grok (Expert) <--->  Orchestration Loop  <---> Grok (Student) |
+-----------+    +-------------------------------------+    +-----------+
                          |                  |
                  TTSUpdateSettings   TTSSpeakFrame
                          |                  v
                   +------------+     +-------------+
                   | xAI TTS    | --> | Transport   |
                   | Rex / Eve  |     | (WebRTC)    |
                   +------------+     +-------------+
                                            |
                                         Listener
```

The pipeline itself is minimal (`TTS -> transport.output()`). The
orchestration loop calls the Grok API directly (via the `openai` SDK) and
queues `TTSSpeakFrame` + `TTSUpdateSettingsFrame` pairs to alternate voices.

## References

- Chaturvedi, I., Llewellyn-Jones, R., & Schaffer, S. R. (2026). *PeerCoT:
  Structured Multi-Agent Chain-of-Thought Collaboration for Error
  Localization in LLM Reasoning.* ICLR 2026 Workshop on Logical Reasoning
  of Large Language Models.
- [Keenable API docs](https://keenable.ai/docs/api)
- [Grok x Keenable latency lab](https://grok.keenable.ai)
