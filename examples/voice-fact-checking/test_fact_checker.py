#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the voice fact-checking example.

Run with::

    uv run pytest examples/voice-fact-checking/test_fact_checker.py
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fact_checker import FactCheckConfig, FactCheckProcessor

from pipecat.frames.frames import TranscriptionFrame
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.keenable.search import KeenableSearchResult
from pipecat.tests.utils import SleepFrame, run_test


def _llm_response(payload: dict, prompt_tokens=10, completion_tokens=10):
    message = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_processor(extractor_payload, judge_payload, search_results):
    search_client = AsyncMock()
    search_client.search = AsyncMock(return_value=search_results)

    proc = FactCheckProcessor(
        search_client=search_client,
        openai_api_key="test-key",
        config=FactCheckConfig(check_interim=False, min_words=3, cooldown_secs=0.0),
    )
    # Extractor call returns first payload, judge call returns the second.
    proc._llm = AsyncMock()
    proc._llm.chat.completions.create = AsyncMock(
        side_effect=[_llm_response(extractor_payload), _llm_response(judge_payload)]
    )
    return proc


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="user", timestamp="2026-01-01T00:00:00Z")


@pytest.mark.asyncio
async def test_factual_claim_emits_verdict():
    """A check-worthy transcript produces a verdict server message."""
    proc = _make_processor(
        extractor_payload={
            "claims": [{"claim": "Python was released in 1991", "query": "Python release year"}]
        },
        judge_payload={
            "verdicts": [
                {
                    "claim": "Python was released in 1991",
                    "verdict": "supported",
                    "explanation": "Python's first release was in 1991.",
                    "sources": ["https://en.wikipedia.org/wiki/Python"],
                }
            ]
        },
        search_results=[
            KeenableSearchResult(
                title="History of Python",
                url="https://en.wikipedia.org/wiki/Python",
                description="1991",
            )
        ],
    )

    down, _ = await run_test(
        proc,
        frames_to_send=[
            _transcription("Python was first released in 1991 by Guido."),
            SleepFrame(sleep=0.4),
        ],
        # One verdict message + one cost/volume stats message.
        expected_down_frames=[
            TranscriptionFrame,
            RTVIServerMessageFrame,
            RTVIServerMessageFrame,
        ],
    )

    server_msgs = [f for f in down if isinstance(f, RTVIServerMessageFrame)]
    fact_checks = [m for m in server_msgs if m.data.get("type") == "fact-check"]
    assert len(fact_checks) == 1
    data = fact_checks[0].data
    assert data["verdict"] == "supported"
    assert data["color"] == "green"
    assert data["sources"] == ["https://en.wikipedia.org/wiki/Python"]


@pytest.mark.asyncio
async def test_non_factual_transcript_does_not_search():
    """When the trigger extracts nothing, no search or verdict happens (cheap exit)."""
    proc = _make_processor(
        extractor_payload={"claims": []},
        judge_payload={"verdicts": []},
        search_results=[],
    )

    down, _ = await run_test(
        proc,
        frames_to_send=[
            _transcription("I really think this is the best conference ever."),
            SleepFrame(sleep=0.3),
        ],
    )

    assert not [f for f in down if isinstance(f, RTVIServerMessageFrame)]
    proc._search.search.assert_not_called()
    # Only the extractor (trigger) call was made; no judge call.
    assert proc._llm.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_short_utterance_is_gated_out():
    """Utterances below min_words never reach the extractor."""
    proc = _make_processor({"claims": []}, {"verdicts": []}, [])

    await run_test(
        proc,
        frames_to_send=[_transcription("Hello there."), SleepFrame(sleep=0.2)],
    )

    proc._llm.chat.completions.create.assert_not_called()
    proc._search.search.assert_not_called()
