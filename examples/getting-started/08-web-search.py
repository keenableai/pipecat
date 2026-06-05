#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice agent with live web search.

Just like 06a-voice-agent-local.py, but with ``web_search=True`` on the LLM
service.  This connects to a free, low-latency search API powered by
Keenable AI (https://keenable.ai) and registers web search tools
automatically — no API key required.

Set ``KEENABLE_API_KEY`` for higher rate limits (https://keenable.ai/console).

Required environment variables:

- ``OPENAI_API_KEY``
- ``DEEPGRAM_API_KEY``

Install: ``pip install "pipecat-ai[openai,deepgram,silero]"``
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = DeepgramTTSService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # web_search=True enables free, low-latency web search powered by
    # Keenable AI. Tools are registered automatically — no API key needed.
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        web_search=True,
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful voice assistant with live web search. "
                "Use search tools when asked about current events, facts, or "
                "anything beyond your training data. Answer in plain spoken "
                "language: no URLs, no bullet points, no markdown. "
                "Keep responses to two or three sentences."
            ),
        ),
    )

    context = LLMContext()
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    context.add_message(
        {"role": "developer", "content": "Briefly introduce yourself to the user."}
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
