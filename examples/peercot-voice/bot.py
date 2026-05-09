#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice PeerCoT: two AI agents discuss a topic with a human, grounded by Keenable web search.

Implements a voice version of PeerCoT (Chaturvedi et al., ICLR 2026 Workshop on
Logical Reasoning of LLMs) where two personas -- an Expert and a Curious
Thinker -- hold a structured multi-turn discussion about a chosen topic.
A human listener can jump in at any time to steer the conversation.
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

from pipecat.frames.frames import (
    EndFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.keenable.search import KeenableSearchClient
from pipecat.services.xai.stt import XAISTTService
from pipecat.services.xai.tts import XAIHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Persona prompts (inspired by PeerCoT: Chaturvedi et al., 2026)
# ---------------------------------------------------------------------------

EXPERT_SYSTEM = """\
You are the Expert in a voice discussion podcast with another AI (the Curious \
Thinker) and a human listener who can jump in. You are precise, analytical, \
and evidence-driven. Ground your reasoning in the web search results provided.

Rules:
- Keep each turn to 2-3 short paragraphs of natural spoken language.
- No bullet points, no URLs, no markdown. Speak as if you are on a podcast.
- Weave facts from the search results naturally into your speech.
- Acknowledge the other speakers' points before extending or countering them.
- When the human listener speaks, address their input directly.
- Use contractions and a conversational tone."""

STUDENT_SYSTEM = """\
You are the Curious Thinker in a voice discussion podcast with another AI \
(the Expert) and a human listener who can jump in. You are exploratory, \
creative, and love to ask probing questions. You challenge assumptions and \
consider alternative angles.

Rules:
- Keep each turn to 2-3 short paragraphs of natural spoken language.
- No bullet points, no URLs, no markdown. Speak as if you are on a podcast.
- Sometimes agree and extend the Expert's reasoning; sometimes push back.
- Ask "what if" questions and explore unconventional ideas.
- Reference the search results when they support your counterpoints.
- When the human listener speaks, engage with their point enthusiastically.
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
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# User speech collector — captures TranscriptionFrames from STT
# ---------------------------------------------------------------------------


class UserSpeechCollector(FrameProcessor):
    """Captures user transcriptions and signals the orchestrator."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pending: list[str] = []
        self._event = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._pending.append(frame.text.strip())
            self._event.set()
        await self.push_frame(frame, direction)

    def take_input(self) -> str | None:
        """Return accumulated user speech and clear buffer, or None."""
        if not self._pending:
            return None
        text = " ".join(self._pending)
        self._pending.clear()
        self._event.clear()
        return text

    async def wait_for_input(self, timeout: float) -> str | None:
        """Wait up to ``timeout`` seconds for user speech."""
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.take_input()


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
    user_collector: UserSpeechCollector,
) -> None:
    """Search the web, then run a multi-turn PeerCoT discussion with human input."""

    logger.info(f"Discussion topic: {topic}")

    # ── 1. Keenable web search ───────────────────────────────────────────
    try:
        results = await search_client.search(topic)
        search_context = "\n".join(
            f"- {r.title}: {r.description}" for r in results
        )
        logger.info(f"Keenable returned {len(results)} result(s)")
    except Exception as exc:
        logger.warning(f"Keenable search failed: {exc}")
        search_context = "(No web search results available.)"

    base_context = f"TOPIC: {topic}\n\nWEB SEARCH RESULTS:\n{search_context}"

    # ── 2. Opening narration ─────────────────────────────────────────────
    await task.queue_frames([
        TTSUpdateSettingsFrame(
            delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
        ),
        TTSSpeakFrame(
            text=(
                f"Welcome to Voice PeerCoT. "
                f"Today we're discussing: {topic}. "
                f"I'm the Expert, and my co-host is the Curious Thinker. "
                f"You can jump in anytime with your thoughts or questions. "
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

        # -- Check for human input before Expert speaks -------------------
        user_input = user_collector.take_input()
        if user_input:
            logger.info(f"Human says: {user_input}")
            human_msg = f"The human listener says: \"{user_input}\"\n\nAddress their point, then continue the discussion."
            expert_messages.append({"role": "user", "content": human_msg})

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
            student_msg = (
                f"{base_context}\n\n"
                f"The Expert opens with:\n\n{expert_text}\n\n"
            )
            if user_input:
                student_msg += f"The human listener also said: \"{user_input}\"\n\n"
            student_msg += "Respond to their analysis."
            student_messages.append({"role": "user", "content": student_msg})
        else:
            student_msg = f"The Expert responds:\n\n{expert_text}\n\n"
            if user_input:
                student_msg += f"The human listener also said: \"{user_input}\"\n\n"
            student_msg += "Continue the discussion."
            student_messages.append({"role": "user", "content": student_msg})

        await task.queue_frames([
            TTSUpdateSettingsFrame(
                delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
            ),
            TTSSpeakFrame(text=expert_text),
        ])

        # -- Brief pause for human to jump in after Expert ----------------
        user_input = await user_collector.wait_for_input(timeout=3.0)
        if user_input:
            logger.info(f"Human says: {user_input}")
            human_msg = f"The human listener says: \"{user_input}\"\n\nAddress their point, then continue the discussion."
            student_messages.append({"role": "user", "content": human_msg})

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

        # -- Brief pause for human to jump in after Student ---------------
        if turn < num_turns - 1:
            user_input = await user_collector.wait_for_input(timeout=3.0)
            if user_input:
                logger.info(f"Human says: {user_input}")
                human_msg = f"The human listener says: \"{user_input}\"\n\nAddress their point in your next response."
                expert_messages.append({"role": "user", "content": human_msg})

    # ── 5. Closing ───────────────────────────────────────────────────────
    await task.queue_frames([
        TTSUpdateSettingsFrame(
            delta=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
        ),
        TTSSpeakFrame(
            text=(
                f"That concludes our Voice PeerCoT discussion on {topic}. "
                "Thanks for listening and joining in."
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

    stt = XAISTTService(api_key=os.environ["XAI_API_KEY"])

    tts = XAIHttpTTSService(
        api_key=os.environ["XAI_API_KEY"],
        settings=XAIHttpTTSService.Settings(voice=EXPERT_VOICE),
    )

    user_collector = UserSpeechCollector()

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_collector,
        tts,
        transport.output(),
    ])

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
            run_discussion(
                topic, num_turns, search_client,
                llm_client, model, task, user_collector,
            )
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
