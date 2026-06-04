#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Voice bot grounded by Keenable via its **public remote MCP server**.

Unlike ``mcp-keenable.py`` (which spawns a *local* stdio MCP server from
``keenable_mcp_server.py``), this example needs **no Keenable code at all**:
Keenable hosts a remote MCP server at ``https://api.keenable.ai/mcp`` speaking
the MCP Streamable HTTP transport, so pipecat's built-in
:class:`~pipecat.services.mcp_service.MCPClient` connects to it directly and
registers its tools on the LLM.

The remote server exposes three tools:

- ``search_web_pages``      - web search (supports ``site`` and
  ``published_after``/``published_before``/``acquired_after``/``acquired_before``
  date filters, plus ``mode`` = ``"pro"`` | ``"realtime"``).
- ``fetch_page_content``    - fetch a single URL and return it as markdown.
- ``submit_search_feedback`` - report per-result relevance back to Keenable.

The server works **keyless** (free tier) — no ``KEENABLE_API_KEY`` required.

Required environment variables (for the voice stack, not for Keenable):

- ``OPENAI_API_KEY``
- ``DEEPGRAM_API_KEY``
- ``CARTESIA_API_KEY``

Optional:

- ``KEENABLE_API_KEY`` - https://keenable.ai/console. When set, it is forwarded
  to the remote MCP server as the ``X-API-Key`` header to use the authenticated
  tier; otherwise the free, keyless tier is used.
- ``KEENABLE_MCP_URL`` - override the remote MCP URL (defaults to
  ``https://api.keenable.ai/mcp``).
"""

import os
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv
from loguru import logger
from mcp.client.session_group import StreamableHttpParameters

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

# Keenable's hosted MCP server (Streamable HTTP transport).
KEENABLE_MCP_URL = os.getenv("KEENABLE_MCP_URL", "https://api.keenable.ai/mcp")


def _keenable_version() -> str:
    """Best-effort pipecat version for the attribution User-Agent."""
    try:
        return version("pipecat-ai")
    except PackageNotFoundError:
        return "unknown"


def _keenable_mcp_headers() -> dict[str, str]:
    """Headers for the remote Keenable MCP connection.

    Always sends a repo-tagged ``User-Agent`` so Keenable can attribute traffic;
    forwards ``X-API-Key`` only when a key is configured (otherwise the keyless
    free tier is used).
    """
    headers = {"User-Agent": f"keenable-pipecat-mcp/{_keenable_version()}"}
    api_key = (os.getenv("KEENABLE_API_KEY") or "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


SYSTEM_INSTRUCTION = """\
You are a helpful voice assistant with access to MCP tools served by Keenable's
web search API. Use `search_web_pages` whenever the user asks about anything
time-sensitive, factual, or beyond your training data, and use
`fetch_page_content` if you need the full content of a specific result.

If the search results do not clearly match the user's question, say so plainly
instead of stretching them. Answer in plain spoken language: no URLs, no bullet
points, no markdown. Keep responses to two or three sentences."""


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
    logger.info("Starting Keenable remote-MCP bot")

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

    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"])

    # Connect straight to Keenable's hosted MCP server over Streamable HTTP —
    # no local MCP server process, no Keenable client code. Keyless by default.
    async with MCPClient(
        server_params=StreamableHttpParameters(
            url=KEENABLE_MCP_URL,
            headers=_keenable_mcp_headers(),
        ),
    ) as mcp:
        tools = await mcp.register_tools(llm)

        # OpenAI takes its system prompt as a context message (unlike Gemini's
        # `system_instruction`), so put it in the LLMContext, not on the service.
        context = LLMContext(
            messages=[{"role": "system", "content": SYSTEM_INSTRUCTION}],
            tools=tools,
        )
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
