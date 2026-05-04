#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keenable web search client and LLM tool helper.

Wraps the Keenable REST API (https://keenable.ai/docs/api) so pipecat bots
can ground answers with live web information. Provides:

- :class:`KeenableSearchClient`: an async HTTP wrapper around ``/v1/search``
  and ``/v1/fetch``.
- :data:`KEENABLE_SEARCH_FUNCTION_SCHEMA`: a ready-made
  :class:`~pipecat.adapters.schemas.function_schema.FunctionSchema` that any
  LLM service can advertise as a callable tool.
- :func:`keenable_search`: the matching ``FunctionCallHandler`` you register
  with ``llm.register_function("keenable_search", keenable_search)``.

Example::

    from pipecat.services.keenable.search import (
        KEENABLE_SEARCH_FUNCTION_SCHEMA,
        keenable_search,
    )
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    llm.register_function("keenable_search", keenable_search)
    tools = ToolsSchema(standard_tools=[KEENABLE_SEARCH_FUNCTION_SCHEMA])
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import Any

import httpx
from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams

DEFAULT_BASE_URL = "https://api.keenable.ai"
SEARCH_PATH = "/v1/search"
FETCH_PATH = "/v1/fetch"
DEFAULT_TIMEOUT_SECS = 30.0
API_KEY_ENV_VAR = "KEENABLE_API_KEY"


class KeenableSearchResult(BaseModel):
    """A single search result returned by the Keenable ``/v1/search`` endpoint."""

    title: str
    url: str
    description: str


class KeenableFetchResult(BaseModel):
    """A page-content result returned by the Keenable ``/v1/fetch`` endpoint."""

    url: str
    title: str | None = None
    content: str


class KeenableSearchClient:
    """Async HTTP client for the Keenable search API.

    The client can be used as an async context manager so its underlying
    ``httpx.AsyncClient`` is closed on exit. Alternatively pass a pre-built
    ``httpx.AsyncClient`` to share one across the application.

    Example::

        async with KeenableSearchClient() as client:
            results = await client.search("typescript best practices")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Keenable search client.

        Args:
            api_key: Keenable API key. If ``None``, falls back to the
                ``KEENABLE_API_KEY`` environment variable.
            base_url: Base URL for the Keenable API.
            timeout: HTTP timeout (seconds) for owned clients. Ignored when a
                pre-built ``client`` is passed.
            client: An existing ``httpx.AsyncClient`` to reuse. When provided,
                the client is borrowed and not closed by this instance.

        Raises:
            ValueError: If no ``api_key`` is given and ``KEENABLE_API_KEY`` is
                not set in the environment.
        """
        resolved = api_key or os.environ.get(API_KEY_ENV_VAR)
        if not resolved:
            raise ValueError(
                f"Keenable API key required: pass api_key=... or set {API_KEY_ENV_VAR}."
            )
        self._api_key: str = resolved
        self._base_url: str = base_url.rstrip("/")
        self._timeout: float = timeout
        self._client: httpx.AsyncClient | None = client
        self._owns_client: bool = client is None

    async def __aenter__(self) -> KeenableSearchClient:
        """Open the underlying ``httpx.AsyncClient`` if owned."""
        self._ensure_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying ``httpx.AsyncClient`` if owned."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying ``httpx.AsyncClient`` if this instance owns it."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def search(self, query: str) -> list[KeenableSearchResult]:
        """Search the public web via Keenable.

        Args:
            query: Search query text.

        Returns:
            A list of :class:`KeenableSearchResult` objects in Keenable's
            ranked order.

        Raises:
            httpx.HTTPStatusError: If the API responds with a non-2xx status.
        """
        client = self._ensure_client()
        response = await client.post(
            f"{self._base_url}{SEARCH_PATH}",
            json={"query": query},
            headers=self._json_headers(),
        )
        response.raise_for_status()
        payload = response.json()
        return [KeenableSearchResult(**item) for item in payload.get("results", [])]

    async def fetch(self, url: str) -> KeenableFetchResult:
        """Fetch the markdown content of an indexed page via Keenable.

        Args:
            url: A page URL from the Keenable index.

        Returns:
            A :class:`KeenableFetchResult` with the URL, optional title, and
            page content as markdown.

        Raises:
            httpx.HTTPStatusError: If the API responds with a non-2xx status.
        """
        client = self._ensure_client()
        response = await client.get(
            f"{self._base_url}{FETCH_PATH}",
            params={"url": url},
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        return KeenableFetchResult(**response.json())

    def _auth_headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key}

    def _json_headers(self) -> dict[str, str]:
        return {**self._auth_headers(), "Content-Type": "application/json"}

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client


KEENABLE_SEARCH_FUNCTION_SCHEMA: FunctionSchema = FunctionSchema(
    name="keenable_search",
    description=(
        "Search the public web for up-to-date information. "
        "Returns a list of results, each with a title, URL, and short snippet. "
        "Use this when the user asks about news, current events, or anything "
        "that may have changed recently."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "The natural-language search query to send to Keenable.",
        },
    },
    required=["query"],
)


def _resolve_client(app_resources: Any) -> tuple[KeenableSearchClient, bool]:
    """Resolve a :class:`KeenableSearchClient` from app resources.

    Returns:
        A tuple ``(client, owned)``. ``owned`` is ``True`` when the caller
        must close the client after use.
    """
    if isinstance(app_resources, KeenableSearchClient):
        return app_resources, False
    candidate = getattr(app_resources, "keenable_search_client", None)
    if isinstance(candidate, KeenableSearchClient):
        return candidate, False
    return KeenableSearchClient(), True


async def keenable_search(params: FunctionCallParams) -> None:
    """LLM tool handler that performs a Keenable web search.

    Looks for a :class:`KeenableSearchClient` on ``params.app_resources``
    (either as the resource itself or as a ``keenable_search_client``
    attribute) and reuses it. Otherwise it builds a per-call client from the
    ``KEENABLE_API_KEY`` environment variable and closes it after use.

    The result delivered to ``result_callback`` is a JSON-friendly dict so it
    serializes cleanly across all LLM providers::

        {"results": [{"title": ..., "url": ..., "description": ...}, ...]}

    On failure, the handler logs the error and returns
    ``{"error": "...", "results": []}`` so the LLM can recover gracefully
    instead of crashing the pipeline.
    """
    query = str(params.arguments.get("query", "")).strip()
    if not query:
        await params.result_callback({"error": "Missing 'query' argument.", "results": []})
        return

    client, owned = _resolve_client(params.app_resources)
    try:
        results = await client.search(query)
        await params.result_callback({"results": [r.model_dump() for r in results]})
    except Exception as exc:
        logger.warning(f"keenable_search failed for query={query!r}: {exc}")
        await params.result_callback({"error": str(exc), "results": []})
    finally:
        if owned:
            await client.aclose()
