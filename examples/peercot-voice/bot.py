#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice PeerCoT: two AI agents discuss a topic, grounded by Keenable web search.

Implements a voice version of PeerCoT (Chaturvedi et al., ICLR 2026 Workshop on
Logical Reasoning of LLMs) where two personas -- an Expert and a Curious
Thinker -- hold a structured multi-turn discussion about a chosen topic.
Each turn is grounded by live web results from the Keenable API, and the
two speakers use distinct xAI TTS voices so the listener can follow
who is talking.

Only two API keys required:

- ``XAI_API_KEY``       -- https://console.x.ai  (used for both LLM and TTS)
- ``KEENABLE_API_KEY``  -- https://keenable.ai/console

Optional:

- ``PEERCOT_TOPIC``  -- discussion topic (default: "the future of AI agents")
- ``PEERCOT_MODEL``  -- LLM model name (default: grok-3-fast)
- ``PEERCOT_TURNS``  -- number of back-and-forth exchanges (default: 4)

Run with::

    PEERCOT_TOPIC="quantum computing breakthroughs" \\
        uv run python examples/peercot-voice/bot.py -t webrtc
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from pipecat.frames.frames import EndFrame, TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.keenable.search import KeenableSearchClient
from pipecat.services.xai.tts import XAIHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Persona prompts (inspired by PeerCoT: Chaturvedi et al., 2026)
# ---------------------------------------------------------------------------

LANGUAGE = os.environ.get("PEERCOT_LANGUAGE", "en")

EXPERT_SYSTEM_EN = """\
You are the Expert in a two-person voice discussion podcast. You are precise, \
analytical, and evidence-driven.

Rules:
- Your ONLY source of facts is the articles provided below. Do NOT make up \
facts or use your training data.
- Naturally reference article titles and news sources when citing facts, \
e.g. "as reported by Financial Express" or "the Republic World piece noted".
- NEVER say "search results", "web search", or "the research shows". Instead \
refer to specific articles or news outlets by name.
- Keep each turn to 3-4 sentences MAX. Be concise and punchy.
- No bullet points, no URLs, no markdown. Speak as if you are on a podcast.
- Acknowledge the other speaker briefly, then make ONE key point.
- Use contractions and a conversational tone."""

EXPERT_SYSTEM_HI = """\
Aap ek Expert ho ek do-logon ki voice discussion podcast mein. Aap precise, \
analytical, aur evidence-driven ho.

Rules:
- Aapka SIRF source hai neeche diye gaye articles. Apne se facts mat banao.
- Facts bolte waqt naturally article titles aur news sources ka naam lo, \
jaise "Financial Express ke mutabiq" ya "Republic World ki report mein".
- KABHI mat bolo "search results", "web search", ya "research shows". \
Specific articles ya news outlets ka naam lo.
- Har turn mein 3-4 sentences MAX. Concise aur punchy raho.
- Hindi mein bolo, natural conversational Hinglish style mein. Jaise podcast \
pe baat kar rahe ho.
- Dusre speaker ki baat briefly acknowledge karo, phir EK key point banao."""

STUDENT_SYSTEM_EN = """\
You are the Curious Thinker in a two-person voice discussion podcast. You are \
exploratory, creative, and love to ask probing questions.

Rules:
- Your ONLY source of facts is the articles provided below. Do NOT make up \
facts. Build on what the articles say.
- Naturally reference article titles and news sources when citing facts.
- NEVER say "search results", "web search", or "the research shows". Instead \
refer to specific articles or news outlets by name.
- Keep each turn to 3-4 sentences MAX. Be concise and punchy.
- No bullet points, no URLs, no markdown. Speak as if you are on a podcast.
- Sometimes agree, sometimes push back. Make ONE point per turn.
- Ask a sharp "what if" question or raise one counterpoint.
- Use contractions and a conversational tone."""

STUDENT_SYSTEM_HI = """\
Aap ek Curious Thinker ho ek do-logon ki voice discussion podcast mein. Aap \
exploratory, creative ho aur sharp sawaal poochte ho.

Rules:
- Aapka SIRF source hai neeche diye gaye articles. Apne se facts mat banao.
- Facts bolte waqt naturally article titles aur news sources ka naam lo.
- KABHI mat bolo "search results", "web search", ya "research shows". \
Specific articles ya news outlets ka naam lo.
- Har turn mein 3-4 sentences MAX. Concise aur punchy raho.
- Hindi mein bolo, natural conversational Hinglish style mein. Jaise podcast \
pe baat kar rahe ho.
- Kabhi agree karo, kabhi push back karo. Har turn mein EK point banao.
- Ek sharp "what if" sawaal ya counterpoint uthao."""

MODE = os.environ.get("PEERCOT_MODE", "discuss")  # "discuss" or "debate"

# ── Debate mode prompts (two experts take opposing sides) ────────────────

BULL_SYSTEM_EN = """\
You are an expert in a two-person voice debate podcast. You believe the \
answer to today's question is YES and argue that position with evidence.

Rules:
- Your ONLY source of facts is the articles provided below. Do NOT make up \
facts. Highlight evidence that supports your position.
- Naturally mention article titles and news outlets when citing facts, e.g. \
"as Reuters reported" or "the Bloomberg piece highlights". Let citations \
flow naturally in conversation.
- NEVER say "search results", "web search", "the research shows", or \
"according to the articles".
- NEVER describe yourself as "optimistic", "bullish", "in the yes camp", or \
reveal your assigned position. Just argue your case with facts. Let the \
listener figure out your stance from your arguments.
- Keep each turn to 3-4 sentences MAX. Be concise and punchy.
- No bullet points, no URLs, no markdown. Speak naturally like a podcast.
- Engage with the other expert's points, then make your case.
- Use contractions and a warm, conversational tone."""

BULL_SYSTEM_HI = """\
Aap ek expert ho ek do-logon ki voice debate podcast mein. Aap believe \
karte ho ki aaj ke sawaal ka jawaab YES hai aur evidence ke saath argue karo.

Rules:
- Aapka SIRF source hai neeche diye gaye articles. Apne se facts mat banao.
- Facts bolte waqt naturally article titles aur news outlets ka naam lo.
- KABHI mat bolo "search results", "web search", ya "research shows".
- KABHI apne aap ko "optimistic", "bullish", "yes camp mein" mat bolo. \
Sirf facts se argue karo. Listener khud samjhe.
- Har turn mein 3-4 sentences MAX. Concise aur punchy raho.
- Hindi mein bolo, natural conversational Hinglish style mein.
- Dusre expert ki baat se engage karo, phir apna case banao."""

BEAR_SYSTEM_EN = """\
You are an expert in a two-person voice debate podcast. You believe the \
answer to today's question is NO and argue that position with evidence.

Rules:
- Your ONLY source of facts is the articles provided below. Do NOT make up \
facts. Highlight evidence that supports your position.
- Naturally mention article titles and news outlets when citing facts, e.g. \
"but the Financial Times piece paints a different picture". Let citations \
flow naturally in conversation.
- NEVER say "search results", "web search", "the research shows", or \
"according to the articles".
- NEVER describe yourself as "skeptical", "bearish", "in the no camp", or \
reveal your assigned position. Just argue your case with facts. Let the \
listener figure out your stance from your arguments.
- Keep each turn to 3-4 sentences MAX. Be concise and punchy.
- No bullet points, no URLs, no markdown. Speak naturally like a podcast.
- Engage with the other expert's points, then poke holes or offer a \
counter-narrative.
- Use contractions and a conversational tone."""

BEAR_SYSTEM_HI = """\
Aap ek expert ho ek do-logon ki voice debate podcast mein. Aap believe \
karte ho ki aaj ke sawaal ka jawaab NO hai aur evidence ke saath argue karo.

Rules:
- Aapka SIRF source hai neeche diye gaye articles. Apne se facts mat banao.
- Facts bolte waqt naturally article titles aur news outlets ka naam lo.
- KABHI mat bolo "search results", "web search", ya "research shows".
- KABHI apne aap ko "skeptical", "bearish", "no camp mein" mat bolo. \
Sirf facts se argue karo. Listener khud samjhe.
- Har turn mein 3-4 sentences MAX. Concise aur punchy raho.
- Hindi mein bolo, natural conversational Hinglish style mein.
- Dusre expert ki baat se engage karo, phir holes nikalo ya counter do."""

def _resolve_personas():
    """Resolve personas from current env vars (called per session)."""
    mode = os.environ.get("PEERCOT_MODE", "discuss")
    lang = os.environ.get("PEERCOT_LANGUAGE", "en")
    if mode == "debate":
        a_sys = BULL_SYSTEM_HI if lang == "hi" else BULL_SYSTEM_EN
        b_sys = BEAR_SYSTEM_HI if lang == "hi" else BEAR_SYSTEM_EN
        a_label = "Yes Advocate"
        b_label = "No Advocate"
    else:
        a_sys = EXPERT_SYSTEM_HI if lang == "hi" else EXPERT_SYSTEM_EN
        b_sys = STUDENT_SYSTEM_HI if lang == "hi" else STUDENT_SYSTEM_EN
        a_label = "Expert"
        b_label = "Curious Thinker"
    return a_sys, b_sys, a_label, b_label, mode, lang

# ---------------------------------------------------------------------------
# xAI Grok TTS voices (Ara, Rex, Sal, Eve, Leo)
# ---------------------------------------------------------------------------

EXPERT_VOICE = "Rex"   # Confident, authoritative
STUDENT_VOICE = "Eve"  # Warm, conversational

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOPIC = "the future of AI agents"
DEFAULT_TURNS = 4

# ---------------------------------------------------------------------------
# Transport params
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transport params
# ---------------------------------------------------------------------------

def _daily_params():
    from pipecat.transports.daily.transport import DailyParams
    return DailyParams(audio_out_enabled=True)

def _twilio_params():
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
    return FastAPIWebsocketParams(audio_out_enabled=True)

transport_params = {
    "daily": _daily_params,
    "twilio": _twilio_params,
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------


def _make_llm_client() -> tuple[AsyncOpenAI, str]:
    """Return ``(client, model)`` using xAI Grok."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY is required — get one at https://console.x.ai")
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    model = os.environ.get("PEERCOT_MODEL", "grok-3-fast")
    logger.info(f"Using xAI Grok ({model})")
    return client, model


# ---------------------------------------------------------------------------
# Discussion orchestrator
# ---------------------------------------------------------------------------


async def run_discussion(
    topic: str,
    num_turns: int,
    search_client: KeenableSearchClient,
    llm_client: AsyncOpenAI,
    model: str,
    task: PipelineTask,
) -> None:
    """Search the web, then generate and speak a multi-turn PeerCoT discussion."""

    logger.info(f"Discussion topic: {topic}")

    # ── 1. Keenable web search + fetch full content ────────────────────
    try:
        import httpx

        headers = {
            "X-API-Key": os.environ["KEENABLE_API_KEY"],
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as http:
            # Search
            resp = await http.post(
                "https://api.keenable.ai/v1/search",
                json={"query": topic},
                headers=headers,
            )
            resp.raise_for_status()
            raw_results = resp.json().get("results", [])
            logger.info(f"Keenable search returned {len(raw_results)} result(s)")

            # Fetch full content for all results
            search_parts = []
            for r in raw_results:
                url = r.get("url", "")
                title = r.get("title", "")
                try:
                    fetch_resp = await http.get(
                        "https://api.keenable.ai/v1/fetch",
                        params={"url": url},
                        headers={"X-API-Key": os.environ["KEENABLE_API_KEY"]},
                    )
                    fetch_resp.raise_for_status()
                    content = fetch_resp.json().get("content", "")
                    # Truncate to ~2000 chars per article to fit context
                    content = content[:2000]
                    search_parts.append(
                        f"=== ARTICLE: {title} ===\n{content}\n"
                    )
                    logger.info(f"Fetched {len(content)} chars from {title}")
                except Exception as fetch_exc:
                    logger.warning(f"Fetch failed for {url}: {fetch_exc}")
                    snippet = r.get("snippet", r.get("description", ""))
                    search_parts.append(f"=== ARTICLE: {title} ===\n{snippet}\n")

            search_context = "\n".join(search_parts)
    except Exception as exc:
        logger.warning(f"Keenable search failed: {exc}")
        search_context = "(No articles available.)"

    base_context = f"TOPIC: {topic}\n\nNEWS ARTICLES:\n{search_context}"

    # ── 2. Resolve personas fresh from env vars ─────────────────────────
    AGENT_A_SYSTEM, AGENT_B_SYSTEM, AGENT_A_LABEL, AGENT_B_LABEL, mode, lang = _resolve_personas()
    logger.info(f"Mode: {mode}, Language: {lang}, A: {AGENT_A_LABEL}, B: {AGENT_B_LABEL}")

    if mode == "debate":
        opener = (
            f"Welcome to Voice PeerCoT. "
            f"Today's question: {topic}. "
            f"Two experts will weigh in with different perspectives. "
            f"Let's begin."
        )
        a_opener = "Make your opening argument FOR the proposition."
        b_prompt_first = lambda a_text: (
            f"{base_context}\n\n"
            f"The other expert opens with:\n\n{a_text}\n\n"
            "Make your opening argument AGAINST the proposition."
        )
        b_prompt_next = lambda a_text: (
            f"The other expert responds:\n\n{a_text}\n\n"
            "Counter their argument."
        )
        a_prompt_next = lambda b_text: (
            f"The other expert responds:\n\n{b_text}\n\n"
            "Counter their argument."
        )
    else:
        opener = (
            f"Welcome to Voice PeerCoT. "
            f"Today we're discussing: {topic}. "
            f"The Expert speaks first, followed by the Curious Thinker. "
            f"Let's begin."
        )
        a_opener = "Share your opening analysis of this topic."
        b_prompt_first = lambda a_text: (
            f"{base_context}\n\n"
            f"The {AGENT_A_LABEL} opens with:\n\n{a_text}\n\n"
            "Respond to their analysis."
        )
        b_prompt_next = lambda a_text: (
            f"The {AGENT_A_LABEL} responds:\n\n{a_text}\n\n"
            "Continue the discussion."
        )
        a_prompt_next = lambda b_text: (
            f"The {AGENT_B_LABEL} responds:\n\n{b_text}\n\n"
            "Continue the discussion."
        )

    await task.queue_frames([
        TTSUpdateSettingsFrame(
            delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
        ),
        TTSSpeakFrame(text=opener),
    ])

    # ── 3. Per-agent message histories ───────────────────────────────────
    agent_a_messages: list[dict] = [
        {"role": "system", "content": AGENT_A_SYSTEM},
        {
            "role": "user",
            "content": f"{base_context}\n\n{a_opener}",
        },
    ]
    agent_b_messages: list[dict] = [
        {"role": "system", "content": AGENT_B_SYSTEM},
        {"role": "user", "content": base_context},
        {"role": "assistant", "content": "Got it, I've reviewed the material. Ready."},
    ]

    # ── 4. Turn loop ─────────────────────────────────────────────────────
    for turn in range(num_turns):
        logger.info(f"Turn {turn + 1}/{num_turns}")

        # -- Agent A turn -------------------------------------------------
        try:
            a_resp = await llm_client.chat.completions.create(
                model=model,
                messages=agent_a_messages,
                temperature=0.3,
            )
            a_text = a_resp.choices[0].message.content
        except Exception as exc:
            logger.error(f"{AGENT_A_LABEL} LLM call failed: {exc}")
            await task.queue_frames([
                TTSSpeakFrame(
                    text="I'm sorry, something went wrong and I can't continue."
                ),
                EndFrame(),
            ])
            return

        agent_a_messages.append({"role": "assistant", "content": a_text})

        # Feed A's words into B's context
        if turn == 0:
            agent_b_messages.append({
                "role": "user",
                "content": b_prompt_first(a_text),
            })
        else:
            agent_b_messages.append({
                "role": "user",
                "content": b_prompt_next(a_text),
            })

        await task.queue_frames([
            TTSUpdateSettingsFrame(
                delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
            ),
            TTSSpeakFrame(text=a_text),
        ])

        # -- Agent B turn -------------------------------------------------
        try:
            b_resp = await llm_client.chat.completions.create(
                model=model,
                messages=agent_b_messages,
                temperature=0.9,
            )
            b_text = b_resp.choices[0].message.content
        except Exception as exc:
            logger.error(f"{AGENT_B_LABEL} LLM call failed: {exc}")
            await task.queue_frames([
                TTSSpeakFrame(
                    text="I'm sorry, something went wrong and I can't continue."
                ),
                EndFrame(),
            ])
            return

        agent_b_messages.append({"role": "assistant", "content": b_text})

        # Feed B's words back into A's context
        agent_a_messages.append({
            "role": "user",
            "content": a_prompt_next(b_text),
        })

        await task.queue_frames([
            TTSUpdateSettingsFrame(
                delta=XAIHttpTTSService.Settings(voice=STUDENT_VOICE),
            ),
            TTSSpeakFrame(text=b_text),
        ])

    # ── 5. Closing ───────────────────────────────────────────────────────
    await task.queue_frames([
        TTSUpdateSettingsFrame(
            delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
        ),
        TTSSpeakFrame(
            text=(
                f"That concludes our Voice PeerCoT discussion on {topic}. "
                "Thank you for listening."
            ),
        ),
        EndFrame(),
    ])

    logger.info("Discussion complete")


# ---------------------------------------------------------------------------
# Bot entry points
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Voice PeerCoT bot")

    topic = os.environ.get("PEERCOT_TOPIC", "")

    # If no topic set, pull trending questions from Polymarket
    if not topic:
        try:
            import httpx as _httpx

            async with _httpx.AsyncClient(timeout=15.0) as pm_http:
                pm_resp = await pm_http.get(
                    "https://gamma-api.polymarket.com/markets",
                    params={
                        "limit": 5,
                        "active": "true",
                        "closed": "false",
                        "order": "volume24hr",
                        "ascending": "false",
                    },
                )
                pm_resp.raise_for_status()
                markets = pm_resp.json()
                # Pick the top trending question as the topic
                if markets:
                    topic = markets[0].get("question", DEFAULT_TOPIC)
                    logger.info(f"Polymarket trending topic: {topic}")
                else:
                    topic = DEFAULT_TOPIC
        except Exception as exc:
            logger.warning(f"Polymarket fetch failed: {exc}")
            topic = DEFAULT_TOPIC

    if not topic:
        topic = DEFAULT_TOPIC
    num_turns = int(os.environ.get("PEERCOT_TURNS", str(DEFAULT_TURNS)))
    llm_client, model = _make_llm_client()
    search_client = KeenableSearchClient()

    from pipecat.transcriptions.language import Language

    tts_language = Language.HI if LANGUAGE == "hi" else Language.EN
    tts = XAIHttpTTSService(
        api_key=os.environ["XAI_API_KEY"],
        settings=XAIHttpTTSService.Settings(voice=EXPERT_VOICE, language=tts_language),
    )

    pipeline = Pipeline([tts, transport.output()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    discussion_handle: asyncio.Task | None = None

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal discussion_handle
        logger.info("Client connected — launching discussion")
        discussion_handle = asyncio.create_task(
            run_discussion(topic, num_turns, search_client, llm_client, model, task)
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if discussion_handle and not discussion_handle.done():
            discussion_handle.cancel()
        await search_client.aclose()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import app, main
    from fastapi.responses import HTMLResponse, JSONResponse

    # Shared mutable topic state
    _current_topic = {"value": os.environ.get("PEERCOT_TOPIC", "")}

    @app.get("/peercot", response_class=HTMLResponse)
    async def peercot_ui():
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice PeerCoT</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:#0c0c14;color:#f1f1f5;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center}
h1{font-size:1.6rem;margin-bottom:.3rem}
.sub{color:#9696a8;font-size:.85rem;margin-bottom:2rem}
.card{background:#14141f;border:1px solid #25253a;border-radius:12px;padding:2rem;width:90%;max-width:500px}
label{display:block;color:#9696a8;font-size:.8rem;margin-bottom:.4rem}
input[type=text]{width:100%;padding:.7rem 1rem;background:#1a1a28;border:1px solid #303048;border-radius:8px;color:#f1f1f5;font-size:1rem;outline:none;margin-bottom:1rem}
input[type=text]:focus{border-color:#4d8aff}
select{width:100%;padding:.6rem 1rem;background:#1a1a28;border:1px solid #303048;border-radius:8px;color:#f1f1f5;font-size:.9rem;margin-bottom:1rem;outline:none}
.row{display:flex;gap:.8rem;margin-bottom:1rem}
.row>*{flex:1}
button{width:100%;padding:.8rem;background:#005CFF;color:#fff;border:none;border-radius:8px;font-size:1rem;font-weight:600;cursor:pointer;margin-top:.5rem}
button:hover{background:#0046cc}
button:disabled{background:#303048;cursor:not-allowed}
.status{text-align:center;margin-top:1rem;color:#9696a8;font-size:.85rem;min-height:1.2rem}
.listening{color:#34d399}
.or{text-align:center;color:#5d5d72;font-size:.8rem;margin:.8rem 0}
</style>
</head>
<body>
<h1>Voice PeerCoT</h1>
<p class="sub">Two AI experts debate any topic, grounded in live web search</p>
<div class="card">
  <label>Topic</label>
  <input type="text" id="topic" placeholder="e.g. Will Bitcoin hit 150k by June 2026">
  <div class="or">or leave empty for trending Polymarket question</div>
  <div class="row">
    <div>
      <label>Mode</label>
      <select id="mode"><option value="debate">Debate</option><option value="discuss">Discussion</option></select>
    </div>
    <div>
      <label>Language</label>
      <select id="lang"><option value="en">English</option><option value="hi">Hindi</option></select>
    </div>
    <div>
      <label>Turns</label>
      <select id="turns"><option value="2">2</option><option value="3" selected>3</option><option value="4">4</option><option value="5">5</option></select>
    </div>
  </div>
  <button id="go" onclick="startDiscussion()">Start Discussion</button>
  <div class="status" id="status"></div>
</div>
<script>
async function startDiscussion() {
  const btn = document.getElementById('go');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Setting topic...';
  const topic = document.getElementById('topic').value;
  const mode = document.getElementById('mode').value;
  const lang = document.getElementById('lang').value;
  const turns = document.getElementById('turns').value;
  try {
    const r = await fetch('/api/topic', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({topic, mode, language: lang, turns: parseInt(turns)})
    });
    if (!r.ok) throw new Error('Failed');
    status.innerHTML = 'Connecting... <a href="/client/" style="color:#4d8aff">Open audio player</a>';
    setTimeout(() => { window.open('/client/', '_blank'); }, 500);
  } catch(e) {
    status.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
}
document.getElementById('topic').addEventListener('keydown', e => { if(e.key==='Enter') startDiscussion(); });
</script>
</body>
</html>"""

    @app.post("/api/topic")
    async def set_topic(request: dict):
        topic = request.get("topic", "")
        mode = request.get("mode", "debate")
        lang = request.get("language", "en")
        turns = request.get("turns", 3)
        # Update env vars so the next bot() call picks them up
        if topic:
            os.environ["PEERCOT_TOPIC"] = topic
        elif "PEERCOT_TOPIC" in os.environ:
            del os.environ["PEERCOT_TOPIC"]
        os.environ["PEERCOT_MODE"] = mode
        os.environ["PEERCOT_LANGUAGE"] = lang
        os.environ["PEERCOT_TURNS"] = str(turns)
        logger.info(f"Topic set: {topic or '(Polymarket)'} mode={mode} lang={lang} turns={turns}")
        return JSONResponse({"ok": True, "topic": topic or "(Polymarket trending)"})

    @app.get("/", include_in_schema=False)
    async def redirect_to_peercot():
        from fastapi.responses import RedirectResponse
        return RedirectResponse("/peercot")

    main()
