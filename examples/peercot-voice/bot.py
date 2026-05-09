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
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Persona prompts (inspired by PeerCoT: Chaturvedi et al., 2026)
# ---------------------------------------------------------------------------

EXPERT_SYSTEM = """\
You are the Expert in a two-person voice discussion podcast. You are precise, \
analytical, and evidence-driven.

Rules:
- Your ONLY source of facts is the WEB SEARCH RESULTS provided. Do NOT make \
up facts or use your training data. If the search results don't cover \
something, say so.
- Cite specific findings from the search results: mention titles, names, \
numbers, and dates you see in the results.
- Keep each turn to 3-4 sentences MAX. Be concise and punchy.
- No bullet points, no URLs, no markdown. Speak as if you are on a podcast.
- Acknowledge the other speaker briefly, then make ONE key point grounded \
in the search results.
- Use contractions and a conversational tone."""

STUDENT_SYSTEM = """\
You are the Curious Thinker in a two-person voice discussion podcast. You are \
exploratory, creative, and love to ask probing questions.

Rules:
- Your ONLY source of facts is the WEB SEARCH RESULTS provided. Do NOT make \
up facts or use your training data. Build on what the search results say.
- When you reference a fact, tie it back to a specific search result.
- Keep each turn to 3-4 sentences MAX. Be concise and punchy.
- No bullet points, no URLs, no markdown. Speak as if you are on a podcast.
- Sometimes agree, sometimes push back. Make ONE point per turn.
- Ask a sharp "what if" question or raise one counterpoint based on the \
search results.
- Use contractions and a conversational tone."""

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

transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
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
        search_context = "(No web search results available.)"

    base_context = f"TOPIC: {topic}\n\nFULL ARTICLE CONTENT FROM WEB:\n{search_context}"

    # ── 2. Opening narration ─────────────────────────────────────────────
    await task.queue_frames([
        TTSUpdateSettingsFrame(
            delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
        ),
        TTSSpeakFrame(
            text=(
                f"Welcome to Voice PeerCoT. "
                f"Today we're discussing: {topic}. "
                f"The Expert speaks first, followed by the Curious Thinker. "
                f"Let's begin."
            ),
        ),
    ])

    # ── 3. Per-agent message histories ───────────────────────────────────
    expert_messages: list[dict] = [
        {"role": "system", "content": EXPERT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{base_context}\n\n"
                "Share your opening analysis of this topic."
            ),
        },
    ]
    student_messages: list[dict] = [
        {"role": "system", "content": STUDENT_SYSTEM},
    ]

    # ── 4. Turn loop ─────────────────────────────────────────────────────
    for turn in range(num_turns):
        logger.info(f"Turn {turn + 1}/{num_turns}")

        # -- Expert turn --------------------------------------------------
        try:
            expert_resp = await llm_client.chat.completions.create(
                model=model,
                messages=expert_messages,
                temperature=0.3,
            )
            expert_text = expert_resp.choices[0].message.content
        except Exception as exc:
            logger.error(f"Expert LLM call failed: {exc}")
            await task.queue_frames([
                TTSSpeakFrame(
                    text="I'm sorry, something went wrong and I can't continue."
                ),
                EndFrame(),
            ])
            return

        expert_messages.append({"role": "assistant", "content": expert_text})

        # Feed Expert's words into the Student's context
        if turn == 0:
            student_messages.append({
                "role": "user",
                "content": (
                    f"{base_context}\n\n"
                    f"The Expert opens with:\n\n{expert_text}\n\n"
                    "Respond to their analysis."
                ),
            })
        else:
            student_messages.append({
                "role": "user",
                "content": (
                    f"The Expert responds:\n\n{expert_text}\n\n"
                    "Continue the discussion."
                ),
            })

        await task.queue_frames([
            TTSUpdateSettingsFrame(
                delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
            ),
            TTSSpeakFrame(text=expert_text),
        ])

        # -- Student turn -------------------------------------------------
        try:
            student_resp = await llm_client.chat.completions.create(
                model=model,
                messages=student_messages,
                temperature=0.9,
            )
            student_text = student_resp.choices[0].message.content
        except Exception as exc:
            logger.error(f"Student LLM call failed: {exc}")
            await task.queue_frames([
                TTSSpeakFrame(
                    text="I'm sorry, something went wrong and I can't continue."
                ),
                EndFrame(),
            ])
            return

        student_messages.append({"role": "assistant", "content": student_text})

        # Feed Student's words back into the Expert's context
        expert_messages.append({
            "role": "user",
            "content": (
                f"The Curious Thinker responds:\n\n{student_text}\n\n"
                "Continue the discussion."
            ),
        })

        await task.queue_frames([
            TTSUpdateSettingsFrame(
                delta=XAIHttpTTSService.Settings(voice=STUDENT_VOICE),
            ),
            TTSSpeakFrame(text=student_text),
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

    topic = os.environ.get("PEERCOT_TOPIC", DEFAULT_TOPIC)
    num_turns = int(os.environ.get("PEERCOT_TURNS", str(DEFAULT_TURNS)))
    llm_client, model = _make_llm_client()
    search_client = KeenableSearchClient()

    tts = XAIHttpTTSService(
        api_key=os.environ["XAI_API_KEY"],
        settings=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
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
    from pipecat.runner.run import main

    main()
