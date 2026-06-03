#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice bot that talks to the Keenable search API via MCP (stdio transport).

Spawns the local ``keenable_mcp_server.py`` as an MCP stdio server,
registers its tools (``keenable_search``, ``keenable_fetch``) on an
OpenAI LLM, and lets the user grill the assistant with web-grounded
questions.

Required environment variables:

- ``OPENAI_API_KEY``
- ``DEEPGRAM_API_KEY``
- ``CARTESIA_API_KEY``

Optional:

- ``KEENABLE_API_KEY`` - https://keenable.ai/console. Without it the MCP server
  uses Keenable's free, keyless public endpoints.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from mcp import StdioServerParameters

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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

load_dotenv(override=True)

SERVER_SCRIPT = str(Path(__file__).parent / "keenable_mcp_server.py")

SYSTEM_INSTRUCTION = """\
You are a helpful voice assistant with access to MCP tools that query the
Keenable web search API. Use `keenable_search` whenever the user asks
about anything time-sensitive, factual, or beyond your training data, and
use `keenable_fetch` if you need the full content of a specific result.

If the search results do not clearly match the user's question, say so
plainly instead of stretching them. Answer in plain spoken language: no
URLs, no bullet points, no markdown. Keep responses to two or three
sentences."""


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
    logger.info("Starting Keenable MCP bot")

    stt = DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            keyterm=["Keenable"],
        ),
    )

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        system_instruction=SYSTEM_INSTRUCTION,
    )

    # Forward the Keenable key to the MCP server subprocess only if one is set;
    # otherwise the server uses Keenable's free, keyless public endpoints.
    server_env = {}
    if os.environ.get("KEENABLE_API_KEY"):
        server_env["KEENABLE_API_KEY"] = os.environ["KEENABLE_API_KEY"]

    async with MCPClient(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT],
            env=server_env,
        ),
    ) as mcp:
        tools = await mcp.register_tools(llm)

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

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
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
