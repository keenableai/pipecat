#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Real-time fact-checking processor for voice/conference transcripts.

:class:`FactCheckProcessor` is a pass-through :class:`FrameProcessor` that
watches the user's final transcripts and, for check-worthy utterances, runs a
small three-stage pipeline entirely in the background (so it never blocks the
conversation audio):

    transcript --> [1] extract claims + queries (1 LLM call, the "trigger")
               --> [2] Keenable web search (capped)
               --> [3] judge verdicts (1 batched LLM call)
               --> RTVIServerMessageFrame -> UI fact-check block

Cost and volume are controlled at every stage by :class:`FactCheckConfig`:
cheap local gates (min words, cooldown, per-session budget, de-duplication,
concurrency) short-circuit most chit-chat before any paid API call, and the
extractor prompt itself is the main "trigger" knob (see ``prompts.py``).

The processor emits two kinds of RTVI server messages, both consumed by the
example web client:

- ``{"type": "fact-check", ...}``  one per judged claim (claim, verdict,
  color, explanation, sources).
- ``{"type": "fact-check-stats", ...}``  a running tally of checks, searches,
  LLM calls and an estimated cost so volume/cost is visible live in the UI.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from pipecat.services.keenable.search import KeenableSearchClient

# Rough USD pricing used only for the live cost meter in the UI. These are
# deliberate over-estimates; override via FactCheckConfig if you care about
# precision. (model -> (input_per_1k, output_per_1k))
_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4.1-mini": (0.0004, 0.0016),
}
_DEFAULT_PRICE = (0.0005, 0.0015)

VERDICT_COLORS = {"supported": "green", "refuted": "red", "unverified": "amber"}


class FactCheckConfig(BaseModel):
    """Configuration controlling fact-checking quality, volume, and cost.

    Parameters:
        enabled: Master switch. When False the processor is a pure pass-through.
        check_interim: Fact-check streaming (interim) transcripts as they grow
            instead of waiting for the utterance-final transcript. Ideal for
            live conference transcription where claims should surface fast.
        interim_min_new_words: When ``check_interim`` is on, only re-evaluate a
            growing partial once it has gained at least this many words since
            the last trigger. Bounds how often mid-utterance checks fire.
        min_words: Skip utterances shorter than this (cheap local gate).
        cooldown_secs: Minimum seconds between two triggered checks. Smooths
            bursts of transcripts during fast speech.
        max_concurrent: Max fact-checks running at once. Extra transcripts are
            dropped rather than queued, bounding worst-case spend.
        max_claims_per_turn: Cap on claims extracted from a single utterance.
        max_searches_per_turn: Cap on Keenable searches per utterance.
        max_checks_per_session: Hard ceiling on judged claims per session. Once
            reached, fact-checking stops for the session.
        dedupe: Skip claims already checked in this session.
        extractor_model: Model for the claim/query extraction (trigger) call.
        judge_model: Model for the verdict call.
        extractor_max_tokens: Output token cap for the extractor call.
        judge_max_tokens: Output token cap for the judge call.
        search_cost_usd: Estimated cost per Keenable search, for the cost meter.
        openai_base_url: Optional OpenAI-compatible base URL (e.g. Groq) so the
            extractor/judge can run on an OSS model.
    """

    enabled: bool = True
    check_interim: bool = True
    interim_min_new_words: int = 8
    min_words: int = 5
    cooldown_secs: float = 4.0
    max_concurrent: int = 2
    max_claims_per_turn: int = 3
    max_searches_per_turn: int = 3
    max_checks_per_session: int = 50
    dedupe: bool = True
    extractor_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    extractor_max_tokens: int = 400
    judge_max_tokens: int = 500
    search_cost_usd: float = 0.005
    openai_base_url: str | None = None


class _Stats(BaseModel):
    """Running cost/volume tally surfaced to the UI."""

    checks: int = 0
    searches: int = 0
    llm_calls: int = 0
    skipped: int = 0
    est_cost_usd: float = 0.0


class FactCheckProcessor(FrameProcessor):
    """Fact-check the user's transcripts in real time and report to the UI.

    Place this right after your STT service and before the user context
    aggregator. It passes every frame through untouched; check-worthy final
    transcripts spawn a background fact-check.

    Example::

        fact_checker = FactCheckProcessor(
            search_client=KeenableSearchClient(),
            openai_api_key=os.environ["OPENAI_API_KEY"],
            config=FactCheckConfig(cooldown_secs=6.0),
        )
        pipeline = Pipeline([transport.input(), stt, fact_checker, user_agg, ...])
    """

    def __init__(
        self,
        *,
        search_client: KeenableSearchClient,
        openai_api_key: str,
        config: FactCheckConfig | None = None,
        **kwargs,
    ):
        """Initialize the fact-check processor.

        Args:
            search_client: Shared Keenable search client used for lookups.
            openai_api_key: API key for the extractor/judge model calls.
            config: Quality/volume/cost configuration. Uses defaults if omitted.
            **kwargs: Additional arguments passed to ``FrameProcessor``.
        """
        super().__init__(**kwargs)
        self._search = search_client
        self._config = config or FactCheckConfig()
        self._stats = _Stats()
        self._seen_claims: set[str] = set()
        self._last_check_ts: float = 0.0
        self._inflight: int = 0
        # Word count of the current partial at the last interim trigger; reset
        # when an utterance finalizes.
        self._last_trigger_words: int = 0

        # Imported lazily so the example only requires `openai` when used.
        from openai import AsyncOpenAI

        self._llm = AsyncOpenAI(api_key=openai_api_key, base_url=self._config.openai_base_url)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Pass frames through and trigger background checks on user transcripts.

        Fact-checks stream in as the transcript grows: interim (partial)
        transcripts are checked as soon as they gain enough new words, and the
        utterance-final transcript catches anything the partials missed. We
        never wait for the final transcript to start checking.
        """
        await super().process_frame(frame, direction)

        if self._config.enabled and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, InterimTranscriptionFrame) and self._config.check_interim:
                self._on_interim(frame.text)
            elif isinstance(frame, TranscriptionFrame):
                self._on_final(frame.text)

        await self.push_frame(frame, direction)

    # -- transcript handling -----------------------------------------------

    def _on_interim(self, text: str):
        """Check a growing partial transcript once it has gained enough words."""
        words = len((text or "").split())
        if words - self._last_trigger_words >= self._config.interim_min_new_words:
            if self._consider(text):
                self._last_trigger_words = words

    def _on_final(self, text: str):
        """Check the finalized utterance, then reset partial tracking."""
        self._consider(text)
        self._last_trigger_words = 0

    # -- gating ------------------------------------------------------------

    def _consider(self, text: str) -> bool:
        """Apply cheap local gates and spawn a background check if they pass.

        Returns:
            True if a background check was started, False if gated out.
        """
        text = (text or "").strip()
        if len(text.split()) < self._config.min_words:
            return False
        now = time.time()
        if now - self._last_check_ts < self._config.cooldown_secs:
            self._stats.skipped += 1
            return False
        if self._stats.checks >= self._config.max_checks_per_session:
            return False
        if self._inflight >= self._config.max_concurrent:
            self._stats.skipped += 1
            return False

        self._last_check_ts = now
        self._inflight += 1
        self.create_task(self._run_check(text), name="fact-check")
        return True

    # -- pipeline ----------------------------------------------------------

    async def _run_check(self, text: str):
        try:
            claims = await self._extract_claims(text)
            if not claims:
                return  # Cheap exit: the trigger said nothing is check-worthy.

            claims = claims[: self._config.max_claims_per_turn]
            if self._config.dedupe:
                claims = [c for c in claims if self._mark_seen(c["claim"])]
                if not claims:
                    return

            searched = claims[: self._config.max_searches_per_turn]
            for claim in searched:
                claim["results"] = await self._search_claim(claim["query"])

            verdicts = await self._judge(searched)
            for verdict in verdicts:
                await self._emit_verdict(verdict)
                self._stats.checks += 1

            await self._emit_stats()
        except Exception as exc:  # never let fact-checking crash the call
            logger.warning(f"Fact-check failed for {text!r}: {exc}")
        finally:
            self._inflight -= 1

    def _mark_seen(self, claim: str) -> bool:
        key = " ".join(claim.lower().split())
        if key in self._seen_claims:
            return False
        self._seen_claims.add(key)
        return True

    async def _extract_claims(self, text: str) -> list[dict[str, str]]:
        """Stage 1: the trigger. Returns ``[{"claim", "query"}, ...]``."""
        # Imported here so a custom/optimized prompt picks up on every reload.
        from prompts import load_extractor_prompt

        resp = await self._llm.chat.completions.create(
            model=self._config.extractor_model,
            max_tokens=self._config.extractor_max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": load_extractor_prompt()},
                {"role": "user", "content": text},
            ],
        )
        self._account(resp, self._config.extractor_model)
        claims = self._parse_json(resp).get("claims", [])
        out: list[dict[str, str]] = []
        for c in claims:
            claim = str(c.get("claim", "")).strip()
            query = str(c.get("query", "")).strip() or claim
            if claim:
                out.append({"claim": claim, "query": query})
        return out

    async def _search_claim(self, query: str) -> list[dict[str, str]]:
        try:
            results = await self._search.search(query)
            self._stats.searches += 1
            self._stats.est_cost_usd += self._config.search_cost_usd
            return [r.model_dump() for r in results[:5]]
        except Exception as exc:
            logger.warning(f"Keenable search failed for {query!r}: {exc}")
            return []

    async def _judge(self, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stage 3: one batched call returning a verdict per claim."""
        from prompts import JUDGE_PROMPT

        payload = [{"claim": c["claim"], "results": c.get("results", [])} for c in claims]
        resp = await self._llm.chat.completions.create(
            model=self._config.judge_model,
            max_tokens=self._config.judge_max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": json.dumps({"claims": payload})},
            ],
        )
        self._account(resp, self._config.judge_model)
        return self._parse_json(resp).get("verdicts", [])

    # -- output ------------------------------------------------------------

    async def _emit_verdict(self, verdict: dict[str, Any]):
        v = str(verdict.get("verdict", "unverified")).lower()
        if v not in VERDICT_COLORS:
            v = "unverified"
        sources = [s for s in verdict.get("sources", []) if isinstance(s, str)]
        data = {
            "type": "fact-check",
            "id": uuid.uuid4().hex,
            "claim": str(verdict.get("claim", "")),
            "verdict": v,
            "color": VERDICT_COLORS[v],
            "explanation": str(verdict.get("explanation", "")),
            "sources": sources,
            "timestamp": time.time(),
        }
        logger.info(f"Fact-check [{v}] {data['claim']!r}")
        await self.push_frame(RTVIServerMessageFrame(data=data))

    async def _emit_stats(self):
        await self.push_frame(
            RTVIServerMessageFrame(data={"type": "fact-check-stats", **self._stats.model_dump()})
        )

    # -- helpers -----------------------------------------------------------

    def _account(self, resp: Any, model: str):
        self._stats.llm_calls += 1
        usage = getattr(resp, "usage", None)
        if not usage:
            return
        in_price, out_price = _MODEL_PRICES.get(model, _DEFAULT_PRICE)
        self._stats.est_cost_usd += (usage.prompt_tokens or 0) / 1000 * in_price + (
            usage.completion_tokens or 0
        ) / 1000 * out_price

    @staticmethod
    def _parse_json(resp: Any) -> dict[str, Any]:
        try:
            return json.loads(resp.choices[0].message.content)
        except Exception:
            return {}
