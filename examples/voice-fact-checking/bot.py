#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Real-time fact-checking voice agent with a live claims UI.

Built for the live-conference use case: a speaker talks, the transcript streams
in, and check-worthy claims are verified against the web via the Keenable
search API *as they are spoken* (not only at utterance end). Each claim shows
up in the browser as a card with a green/red/amber verdict, a one-line
explanation, and source links, alongside a live cost/volume meter.

Pipeline::

    transport.input -> STT -> FactCheckProcessor -> user_aggregator
                    -> LLM -> TTS -> transport.output -> assistant_aggregator

The FactCheckProcessor runs all fact-checking in the background, so the spoken
conversation is never blocked. It also speaks normally as an assistant; fact
checks are an out-of-band overlay surfaced over RTVI server messages.

Required environment variables:

- ``KEENABLE_API_KEY``  - https://keenable.ai/console
- ``OPENAI_API_KEY``    - used for the bot LLM and the extractor/judge calls
- ``DEEPGRAM_API_KEY``  - streaming STT (interim + final transcripts)
- ``CARTESIA_API_KEY``  - TTS

Run with::

    uv run python examples/voice-fact-checking/bot.py

Then open http://localhost:7860/ and click Connect.
"""

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fact_checker import FactCheckConfig, FactCheckProcessor
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
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
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.keenable.search import KeenableSearchClient
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

load_dotenv(override=True)

CLIENT_DIR = Path(__file__).parent / "client"

SYSTEM_INSTRUCTION = """\
You are a concise voice assistant moderating a live conversation. Respond to
what the speaker says in one or two spoken sentences. Do not read out fact
checks or sources; a separate system verifies claims and shows them on screen.
Speak naturally, with no markdown, bullet points, or URLs."""


app = FastAPI()
pcs_map: dict[str, SmallWebRTCConnection] = {}
ice_servers = [IceServer(urls="stun:stun.l.google.com:19302")]


async def run_bot(webrtc_connection: SmallWebRTCConnection):
    logger.info("Starting fact-checking bot")

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    # Deepgram with interim results so we can fact-check mid-utterance.
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"], interim_results=True)

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

    # One Keenable client shared by every fact-check in this session.
    search_client = KeenableSearchClient()

    fact_checker = FactCheckProcessor(
        search_client=search_client,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        config=FactCheckConfig(),
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
            fact_checker,  # observes interim + final transcripts, never blocks
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {
                "role": "developer",
                "content": (
                    "Briefly greet the speaker and explain you'll fact-check claims live on screen."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await search_client.aclose()
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        connection = SmallWebRTCConnection(ice_servers)
        await connection.initialize(sdp=request["sdp"], type=request["type"])

        @connection.event_handler("closed")
        async def handle_disconnected(conn: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {conn.pc_id}")
            pcs_map.pop(conn.pc_id, None)

        background_tasks.add_task(run_bot, connection)

    answer = connection.get_answer()
    pcs_map[answer["pc_id"]] = connection
    return answer


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/client/")


# Custom UI with the live fact-check block (served instead of the prebuilt UI).
app.mount("/client", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await asyncio.gather(*(pc.disconnect() for pc in pcs_map.values()))
    pcs_map.clear()


app.router.lifespan_context = lifespan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice fact-checking bot")
    parser.add_argument("--host", default="localhost", help="HTTP host (default: localhost)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
