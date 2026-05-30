# Real-time voice fact-checking

Fact-check a live voice conversation **as it is spoken**. Built for the
conference / live-transcription use case: a speaker talks, the transcript
streams in, check-worthy claims are verified against the web with the
[Keenable](https://keenable.ai) search API, and each claim shows up in the
browser as a **green / red / amber card** with a one-line explanation and
source links — next to a live **cost & volume meter**.

```
transport.input ─▶ STT ─▶ FactCheckProcessor ─▶ user_aggregator
                                │                     │
                                │ (background)        ▼
                                │                    LLM ─▶ TTS ─▶ output
                                ▼
   transcript ─▶ [1] extract claims + queries  (LLM "trigger")
              ─▶ [2] Keenable web search        (capped)
              ─▶ [3] judge verdicts             (1 batched LLM call)
              ─▶ RTVIServerMessageFrame ─▶ UI claim cards + cost meter
```

Fact-checking runs entirely in the background, so it never blocks or delays the
spoken conversation.

## Checks stream in — no waiting for the final transcript

STT runs with `interim_results=True`. The processor evaluates **partial**
transcripts as soon as they gain enough new words (`interim_min_new_words`), so
claims surface mid-sentence. The utterance-final transcript catches anything the
partials missed, and de-duplication stops the same claim being checked twice.

## Setup

From the repo root:

```bash
cp env.example .env   # then add the keys below
uv run python examples/voice-fact-checking/bot.py
```

Open http://localhost:7860/ and click **Connect**.

Required environment variables:

| Variable          | Used for                                            |
| ----------------- | --------------------------------------------------- |
| `KEENABLE_API_KEY`| Web search ([console](https://keenable.ai/console)) |
| `OPENAI_API_KEY`  | Bot LLM **and** the extractor/judge calls           |
| `DEEPGRAM_API_KEY`| Streaming STT (interim + final transcripts)         |
| `CARTESIA_API_KEY`| TTS                                                 |

The UI is a dependency-free vanilla-WebRTC client (no build step) served from
`client/`.

## Controlling volume and cost

Most chit-chat is rejected by **cheap local gates before any paid API call**.
Tune everything in `FactCheckConfig` (`fact_checker.py`):

| Knob                      | Default | Effect                                              |
| ------------------------- | ------- | --------------------------------------------------- |
| `check_interim`           | `True`  | Check streaming partials, not just final transcripts |
| `interim_min_new_words`   | `8`     | New words a partial must gain before re-checking     |
| `min_words`               | `5`     | Ignore very short utterances                         |
| `cooldown_secs`           | `4.0`   | Minimum gap between triggered checks                 |
| `max_concurrent`          | `2`     | Drop (don't queue) checks beyond this                |
| `max_claims_per_turn`     | `3`     | Cap claims extracted per utterance                   |
| `max_searches_per_turn`   | `3`     | Cap Keenable searches per utterance                  |
| `max_checks_per_session`  | `50`    | Hard per-session ceiling on judged claims            |
| `dedupe`                  | `True`  | Never re-check an already-checked claim              |
| `extractor_model` / `judge_model` | `gpt-4o-mini` | Use a small/cheap model for both stages   |
| `openai_base_url`         | `None`  | Point the extractor/judge at an OSS/Groq endpoint    |

Per check-worthy utterance the worst case is **1 extractor call + ≤3 searches +
1 judge call**; non-factual lines usually cost just the single cheap extractor
call (or nothing, if a local gate fires first). The live meter in the UI shows
running checks, searches, LLM calls, gated-out count, and an estimated cost.

## Tuning the trigger (what gets fact-checked)

The trigger is **all prompt** — `DEFAULT_EXTRACTOR_PROMPT` in `prompts.py`
decides which lines are check-worthy and writes the search query. Three ways to
improve it, in increasing order of automation:

1. Edit `DEFAULT_EXTRACTOR_PROMPT` by hand.
2. Drop a tuned prompt into `optimized_extractor_prompt.txt` — it overrides the
   default automatically (`load_extractor_prompt()`).
3. Evolve it with **GEPA** (below).

## GEPA: optimize the trigger prompt

`gepa_optimize.py` runs [GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto
reflective prompt evolution) over the trigger prompt using the labeled
`dataset.json` (conference-style lines marked check-worthy or not). It runs the
prompt, reads natural-language feedback from its mistakes (false triggers,
missed claims, over-extraction), asks a reflection LLM to rewrite the prompt,
and keeps prompts on a Pareto frontier. The best prompt is written to
`optimized_extractor_prompt.txt`, which the bot then loads automatically.

```bash
# Real optimization (needs OPENAI_API_KEY); --budget caps cost
uv run python examples/voice-fact-checking/gepa_optimize.py \
    --budget 60 --task-model gpt-4o-mini --reflection-model gpt-4o

# Offline self-test of the GEPA loop (no API key, deterministic)
uv run python examples/voice-fact-checking/gepa_optimize.py --self-test
```

Add your own rows to `dataset.json` (e.g. lines your deployment over- or
under-triggered on) and re-run to specialize the trigger for your domain.

## Files

| File                          | Purpose                                              |
| ----------------------------- | ---------------------------------------------------- |
| `bot.py`                      | FastAPI + SmallWebRTC server, pipeline, custom UI     |
| `fact_checker.py`             | `FactCheckProcessor` + `FactCheckConfig` (cost knobs) |
| `prompts.py`                  | Trigger + judge prompts; optimized-prompt loader      |
| `gepa_optimize.py`            | GEPA trigger-prompt optimizer (+ offline self-test)   |
| `dataset.json`                | Labeled examples for GEPA                             |
| `client/`                     | Zero-dependency web UI (claim cards + cost meter)     |
