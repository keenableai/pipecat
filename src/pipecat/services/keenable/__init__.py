#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keenable services for grounding LLM responses with live web search.

See :mod:`pipecat.services.keenable.search` for the client, tool handlers, and
function schemas.
"""

from pipecat.services.keenable.search import (
    KEENABLE_FETCH_FUNCTION_SCHEMA,
    KEENABLE_SEARCH_FUNCTION_SCHEMA,
    KeenableAuthError,
    KeenableError,
    KeenableInsufficientCreditsError,
    KeenableRateLimitError,
    KeenableSearchClient,
    keenable_fetch,
    keenable_search,
)

__all__ = [
    "KEENABLE_FETCH_FUNCTION_SCHEMA",
    "KEENABLE_SEARCH_FUNCTION_SCHEMA",
    "KeenableAuthError",
    "KeenableError",
    "KeenableInsufficientCreditsError",
    "KeenableRateLimitError",
    "KeenableSearchClient",
    "keenable_fetch",
    "keenable_search",
]
