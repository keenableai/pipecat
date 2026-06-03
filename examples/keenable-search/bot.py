#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice bot grounded by the Keenable web search API.

The bot listens to the user, calls the ``keenable_search`` tool whenever it
needs current web information (and ``keenable_fetch`` to read a result in full),
and speaks back a grounded answer.

This example runs **fully on free/local services by default**: local Whisper STT
and local Kokoro TTS (both download a small model on first run, no API key), and
Keenable's keyless public endpoints. The only key you need is for the LLM.

Required environment variables:

- ``OPENAI_API_KEY`` (or set ``GROQ_API_KEY`` and use the commented Groq block
  for a free LLM option).

Optional:

- ``KEENABLE_API_KEY`` - https://keenable.ai/console. Without it the bot uses
  Keenable's free, keyless public endpoints.

Run with::

    # from the repo root, install the local STT/TTS engines + an LLM provider:
    uv sync --extra openai --extra whisper --extra kokoro --extra silero
    uv run python examples/keenable-search/bot.py

To use premium cloud STT/TTS instead (Deepgram + Cartesia), see the commented
block in ``run_bot`` and the README in this directory. The LLM can likewise be
swapped to ``GroqLLMService`` (Llama 3.3 70B) — also shown below.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.keenable.search import (
    KEENABLE_FETCH_FUNCTION_SCHEMA,
    KEENABLE_SEARCH_FUNCTION_SCHEMA,
    KeenableSearchClient,
    keenable_fetch,
    keenable_search,
)
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)


SYSTEM_INSTRUCTION = """\
You are a helpful voice assistant. When the user asks about anything that
might be time-sensitive, factual, or beyond your training data, call the
`keenable_search` tool to look it up on the public web before answering. If a
single result needs reading in full, call `keenable_fetch` with its URL.

Each search result has a title, URL, and short snippet. Use those to ground
your reply in real sources, but answer in plain spoken language: no URLs,
no bullet points, no markdown. Keep responses to two or three sentences."""


# Defer transport parameter creation until the transport type is selected at runtime.
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Keenable-grounded bot")

    # Free/local by default: Whisper (STT) and Kokoro (TTS) run on-device and
    # download a small model on first use — no API keys. Kokoro requires a voice.
    stt = WhisperSTTService()
    tts = KokoroTTSService(settings=KokoroTTSService.Settings(voice="af_heart"))
    # Premium cloud alternative (uncomment + set DEEPGRAM_API_KEY / CARTESIA_API_KEY):
    # from pipecat.services.deepgram.stt import DeepgramSTTService
    # from pipecat.services.cartesia.tts import CartesiaTTSService
    # stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    # tts = CartesiaTTSService(
    #     api_key=os.environ["CARTESIA_API_KEY"],
    #     settings=CartesiaTTSService.Settings(
    #         voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    #     ),
    # )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        system_instruction=SYSTEM_INSTRUCTION,
    )
    # OSS-friendly alternative (uncomment + swap api_key env var):
    # from pipecat.services.groq.llm import GroqLLMService
    # llm = GroqLLMService(
    #     api_key=os.environ["GROQ_API_KEY"],
    #     settings=GroqLLMService.Settings(model="llama-3.3-70b-versatile"),
    #     system_instruction=SYSTEM_INSTRUCTION,
    # )

    llm.register_function("keenable_search", keenable_search)
    llm.register_function("keenable_fetch", keenable_fetch)
    tools = ToolsSchema(
        standard_tools=[KEENABLE_SEARCH_FUNCTION_SCHEMA, KEENABLE_FETCH_FUNCTION_SCHEMA]
    )

    context = LLMContext(tools=tools)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    # Share one Keenable client across all tool invocations in this session.
    search_client = KeenableSearchClient()

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        app_resources=search_client,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Briefly introduce yourself to the user."}
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
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
