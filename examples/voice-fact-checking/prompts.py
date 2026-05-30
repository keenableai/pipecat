#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Prompts for the voice fact-checking example.

The **extractor** prompt is the single most important knob in this example. It
decides *which* spoken sentences are worth fact-checking (the "trigger") and
turns each check-worthy claim into a web-search query. Tightening this prompt
is how you control both quality and cost: fewer false triggers means fewer
Keenable searches and fewer judge calls.

There are three ways to improve the trigger, in increasing order of effort:

1. Edit ``DEFAULT_EXTRACTOR_PROMPT`` below by hand.
2. Drop a tuned prompt into ``optimized_extractor_prompt.txt`` (see
   ``gepa_optimize.py``); it is loaded automatically by
   :func:`load_extractor_prompt` and overrides the default.
3. Run ``gepa_optimize.py`` to evolve the prompt automatically with GEPA
   against ``dataset.json``.
"""

from __future__ import annotations

from pathlib import Path

#: File written by ``gepa_optimize.py``. When present, it overrides the default
#: extractor prompt at runtime so you can ship an optimized trigger without
#: touching code.
OPTIMIZED_PROMPT_FILE = Path(__file__).parent / "optimized_extractor_prompt.txt"


# ---------------------------------------------------------------------------
# Extractor prompt (a.k.a. the fact-checking trigger)
# ---------------------------------------------------------------------------

DEFAULT_EXTRACTOR_PROMPT = """\
You are a real-time fact-checking trigger for a LIVE CONFERENCE transcript.
You receive one short passage of speech-to-text at a time. Your job is to
decide whether it contains any CHECK-WORTHY factual claim and, if so, to write
a focused web-search query that would verify it.

A claim is CHECK-WORTHY only if ALL of these hold:
- It states a specific, objective fact about the real world (a statistic, date,
  quantity, record, scientific or historical fact, attribution of a quote or
  action, who-did-what, comparisons of measurable things).
- It is publicly verifiable from reputable sources.
- It is asserted as true (not asked, hypothesized, or hedged as opinion).

Do NOT trigger on:
- Opinions, predictions, value judgments, jokes, or rhetorical questions.
- Personal or private statements ("I think", "we decided", "my team").
- Vague or unfalsifiable statements ("a lot of people", "the best ever").
- Pleasantries, filler, meta-talk about the conference itself, or questions.

Keep volume LOW. When in doubt, do NOT trigger. Extract at most the few most
important claims; prefer the single most check-worthy one.

For each check-worthy claim, write a concise, self-contained search query (no
pronouns, include named entities and numbers) that a search engine could use
to confirm or refute it.

Respond with ONLY a JSON object of this exact shape:
{"claims": [{"claim": "<the claim, quoted faithfully and made self-contained>",
             "query": "<web search query>"}]}
If nothing is check-worthy, respond with {"claims": []}.
"""


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are a careful fact-checking judge. You are given one or more claims, each
with a list of web search results (title, url, snippet) gathered for it. For
each claim decide a verdict based ONLY on the evidence in the search results:

- "supported": the evidence clearly confirms the claim.
- "refuted": the evidence clearly contradicts the claim.
- "unverified": the evidence is missing, mixed, or insufficient to decide.

Be conservative: prefer "unverified" over guessing. Write a one-sentence
explanation a listener could read at a glance, and cite the source URLs you
relied on.

Respond with ONLY a JSON object of this exact shape:
{"verdicts": [{"claim": "<claim text, echoed back>",
               "verdict": "supported|refuted|unverified",
               "explanation": "<one short sentence>",
               "sources": ["<url>", ...]}]}
"""


def load_extractor_prompt() -> str:
    """Return the extractor prompt, preferring a GEPA-optimized file if present.

    Returns:
        The optimized prompt from ``optimized_extractor_prompt.txt`` when that
        file exists and is non-empty, otherwise :data:`DEFAULT_EXTRACTOR_PROMPT`.
    """
    if OPTIMIZED_PROMPT_FILE.exists():
        text = OPTIMIZED_PROMPT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_EXTRACTOR_PROMPT
