#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keenable web search + page fetch for pipecat.

Wraps the `Keenable <https://keenable.ai>`_ REST API so pipecat bots can ground
their answers with live web information. Keenable is a web search engine built
for AI agents; this module turns it into LLM tools any pipecat ``LLMService``
can call.

Provides:

- :class:`KeenableSearchClient`: an async ``httpx`` wrapper around
  ``/v1/search`` (keyless ``/v1/search/public``) and ``/v1/fetch`` (keyless
  ``/v1/fetch/public``).
- :data:`KEENABLE_SEARCH_FUNCTION_SCHEMA` / :data:`KEENABLE_FETCH_FUNCTION_SCHEMA`:
  ready-made :class:`~pipecat.adapters.schemas.function_schema.FunctionSchema`
  objects an LLM service can advertise as callable tools.
- :func:`keenable_search` / :func:`keenable_fetch`: the matching
  ``FunctionCallHandler`` coroutines you register with
  ``llm.register_function(...)``.

No API key is required: the client defaults to Keenable's **free, keyless**
public endpoints and transparently upgrades to the authenticated endpoints when
``KEENABLE_API_KEY`` is set (or an ``api_key`` is passed). The API base URL is
read from the ``KEENABLE_API_URL`` environment variable (default
``https://api.keenable.ai``, HTTPS-enforced) and is intentionally never a
public/LLM-settable argument.

Example::

    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.services.keenable.search import (
        KEENABLE_FETCH_FUNCTION_SCHEMA,
        KEENABLE_SEARCH_FUNCTION_SCHEMA,
        KeenableSearchClient,
        keenable_fetch,
        keenable_search,
    )

    llm.register_function("keenable_search", keenable_search)
    llm.register_function("keenable_fetch", keenable_fetch)
    tools = ToolsSchema(
        standard_tools=[KEENABLE_SEARCH_FUNCTION_SCHEMA, KEENABLE_FETCH_FUNCTION_SCHEMA]
    )

    # Share one client across the session via PipelineTask(app_resources=...):
    task = PipelineTask(pipeline, app_resources=KeenableSearchClient())
"""

from __future__ import annotations

import os
from importlib import metadata
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import httpx
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema

if TYPE_CHECKING:
    # Imported only for type checking: pulling in ``llm_service`` at runtime would
    # drag the whole pipecat core in, and these clients (and the red-team / e2e
    # harness) are useful without it. The handlers accept a duck-typed object with
    # ``arguments``, ``result_callback`` and ``app_resources``.
    from pipecat.services.llm_service import FunctionCallParams

try:
    _VERSION = metadata.version("pipecat-ai")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout / path-load
    _VERSION = "unknown"

# Tagged User-Agent so Keenable can attribute traffic from this integration.
# Keeps the ``keenable-<framework>/<version>`` shape used across the integrations.
USER_AGENT = f"keenable-pipecat/{_VERSION}"

DEFAULT_BASE_URL = "https://api.keenable.ai"
API_KEY_ENV_VAR = "KEENABLE_API_KEY"
BASE_URL_ENV_VAR = "KEENABLE_API_URL"
DEFAULT_TIMEOUT_SECS = 30.0

# Keyed vs. keyless endpoint paths.
SEARCH_PATH = "/v1/search"
SEARCH_PUBLIC_PATH = "/v1/search/public"
FETCH_PATH = "/v1/fetch"
FETCH_PUBLIC_PATH = "/v1/fetch/public"

# Hosts we refuse to fetch client-side. The backend enforces full SSRF protection
# server-side; this is a cheap first line of defence before sending.
_BLOCKED_FETCH_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"})

SearchMode = Literal["pro", "realtime"]


# --- errors ----------------------------------------------------------------
#
# Distinct error types so callers (and the tool handlers) can react to *why* a
# request failed -- auth vs. credits vs. rate limit vs. server -- instead of an
# opaque error. Each message carries the backend's own explanation (which often
# includes upgrade/auth instructions) without ever echoing the request, so the
# API key cannot leak through an error string.


class KeenableError(RuntimeError):
    """Base class for all Keenable API errors."""


class KeenableAuthError(KeenableError):
    """The API key was missing, invalid, or rejected (HTTP 401)."""


class KeenableInsufficientCreditsError(KeenableError):
    """The account is out of credits (HTTP 402)."""


class KeenableRateLimitError(KeenableError):
    """The request was rate limited (HTTP 429).

    The keyless public endpoints are throttled (~2 RPS plus an hourly cap);
    keyed organizations have configurable limits. Back off and retry.
    """


def _extract_error_message(response: httpx.Response) -> str:
    """Pull the backend's human-readable message out of an error response.

    Tries the JSON body first (``detail``/``message``/``error``), falling back
    to the raw text. Never includes request data, so the key cannot leak.
    """
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:500] if text else "(no response body)"
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(body)[:500]


def _raise_for_status(response: httpx.Response) -> None:
    """Map an error status code to a specific :class:`KeenableError`."""
    status = response.status_code
    if status < 400:
        return
    message = _extract_error_message(response)
    if status == 401:
        raise KeenableAuthError(f"Keenable authentication failed (401): {message}")
    if status == 402:
        raise KeenableInsufficientCreditsError(
            f"Keenable request rejected for insufficient credits (402): {message}"
        )
    if status == 429:
        raise KeenableRateLimitError(f"Keenable rate limit hit (429): {message}")
    if status >= 500:
        raise KeenableError(f"Keenable server error ({status}): {message}")
    raise KeenableError(f"Keenable request failed ({status}): {message}")


def _resolve_base_url() -> str:
    """Resolve and validate the API base URL from ``KEENABLE_API_URL``.

    Defaults to ``https://api.keenable.ai`` and enforces HTTPS, except for
    loopback hosts used in local development. The base URL is intentionally
    *not* a public constructor argument -- it is an SSRF foothold and no
    competitor exposes it.
    """
    base = (os.environ.get(BASE_URL_ENV_VAR) or DEFAULT_BASE_URL).rstrip("/")
    parts = urlsplit(base)
    host = (parts.hostname or "").lower()
    is_loopback = host in {"localhost", "127.0.0.1", "::1"}
    if parts.scheme == "https" or (parts.scheme == "http" and is_loopback):
        return base
    raise ValueError(
        f"{BASE_URL_ENV_VAR} must use https (got {base!r}); "
        "http is only allowed for loopback during local development"
    )


class KeenableSearchClient:
    """Async client for the Keenable web search + page fetch API.

    Defaults to the **free, keyless** public endpoints
    (``/v1/search/public``, ``/v1/fetch/public``). When an API key is configured
    -- passed explicitly or read from the ``KEENABLE_API_KEY`` environment
    variable -- it uses the authenticated endpoints (``/v1/search``,
    ``/v1/fetch``) and sends the key in the ``X-API-Key`` header.

    Search *filters* (``site`` and the date ranges) are arguments to
    :meth:`search`, not constructor config, so the same client can be reused
    across queries with different filters -- the whole point of giving a model a
    search tool. ``mode`` may be set as a per-client default and overridden per
    call.

    The client can be used as an async context manager so its underlying
    ``httpx.AsyncClient`` is closed on exit, or share one ``httpx.AsyncClient``
    across the application by passing ``client=``.

    Example::

        async with KeenableSearchClient() as client:  # keyless / free tier
            results = await client.search("typescript best practices", site="github.com")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        mode: SearchMode = "pro",
        timeout: float = DEFAULT_TIMEOUT_SECS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Keenable client.

        Args:
            api_key: Keenable API key. If ``None`` (the default), falls back to
                the ``KEENABLE_API_KEY`` environment variable; if that is unset
                or blank, the keyless public endpoints are used (free tier).
            mode: Default search mode. ``"pro"`` (default) performs deeper
                retrieval; ``"realtime"`` is lower latency but is **not
                available on the keyless free tier**. Overridable per
                :meth:`search` call.
            timeout: HTTP timeout (seconds) for owned clients. Ignored when a
                pre-built ``client`` is passed.
            client: An existing ``httpx.AsyncClient`` to reuse. When provided,
                the client is borrowed and not closed by this instance.
        """
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV_VAR)
        # Treat a blank/whitespace key as "no key" so we cleanly fall back to the
        # free tier instead of sending an empty X-API-Key header.
        if api_key is not None and not api_key.strip():
            api_key = None

        # Held in a closure so the key never appears in ``__dict__``/``vars``/
        # ``repr`` or any naive serialization -- only ``_api_key_getter`` (a
        # function) is an instance attribute.
        self._api_key_getter = (lambda key: lambda: key)(api_key)
        self.base_url = _resolve_base_url()
        self.mode: SearchMode = mode
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    @property
    def _api_key(self) -> str | None:
        return self._api_key_getter()

    @property
    def has_api_key(self) -> bool:
        """Whether an API key is configured (authenticated endpoints in use)."""
        return self._api_key is not None

    def __repr__(self) -> str:
        # Deliberately omit the API key.
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, "
            f"has_api_key={self._api_key is not None})"
        )

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

    async def search(
        self,
        query: str,
        *,
        site: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        acquired_after: str | None = None,
        acquired_before: str | None = None,
        mode: SearchMode | None = None,
    ) -> list[dict[str, Any]]:
        """Run a web search and return the API's list of result dicts.

        Each result is a dict with keys such as ``title``, ``url``,
        ``description``, ``published_at`` and ``acquired_at``. The full result
        set the API returns is passed through unchanged -- there is no
        client-side cap (the API has no ``max_results`` parameter).

        Args:
            query: The search query.
            site: Restrict results to a single domain, e.g. ``"github.com"``.
            published_after: Only pages published on/after this ``YYYY-MM-DD``.
            published_before: Only pages published on/before this ``YYYY-MM-DD``.
            acquired_after: Only pages indexed on/after this ``YYYY-MM-DD``.
            acquired_before: Only pages indexed on/before this ``YYYY-MM-DD``.
            mode: Override the client's default search mode for this call.

        Returns:
            The list of result dictionaries returned by the API.

        Raises:
            ValueError: If ``query`` is empty or the API response is malformed.
            KeenableError: If the Keenable API returns an error (see the
                :class:`KeenableAuthError` / :class:`KeenableRateLimitError` /
                etc. subclasses) or the request fails.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        url = self._endpoint(SEARCH_PATH, SEARCH_PUBLIC_PATH)
        payload: dict[str, Any] = {"query": query, "mode": mode or self.mode}
        for field, value in (
            ("site", site),
            ("published_after", published_after),
            ("published_before", published_before),
            ("acquired_after", acquired_after),
            ("acquired_before", acquired_before),
        ):
            if value:
                payload[field] = value

        headers = {**self._headers(), "Content-Type": "application/json"}

        # Everything that can fail -- transport, HTTP status, JSON parsing,
        # shape validation -- is handled here so the caller gets a clean, typed
        # error instead of a raw ``httpx`` / ``JSONDecodeError``.
        client = self._ensure_client()
        try:
            response = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as e:
            # Never echo the request (which carries the key) into the message.
            raise KeenableError(f"Keenable search request failed: {e!r}") from e

        _raise_for_status(response)

        try:
            data = response.json()
        except ValueError as e:
            raise KeenableError("Keenable search returned a non-JSON response") from e

        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ValueError(f"Unexpected response from the Keenable search API: {data!r}")
        return results

    async def fetch(self, url: str) -> dict[str, Any]:
        """Fetch a web page via Keenable and return it as a dict.

        Args:
            url: A public ``http://`` or ``https://`` page URL.

        Returns:
            A dict with at least ``url``, ``title`` and ``content`` (markdown);
            extra fields such as ``description``, ``author`` and
            ``published_at`` appear when the page exposes them.

        Raises:
            ValueError: If ``url`` is missing, uses a non-web scheme, or targets
                an obviously private host.
            KeenableError: If the Keenable API returns an error or the request
                fails.
        """
        self._validate_fetch_url(url)

        endpoint = self._endpoint(FETCH_PATH, FETCH_PUBLIC_PATH)
        # ``/v1/fetch`` is a GET with a ``?url=`` query param -- a POST returns 404.
        client = self._ensure_client()
        try:
            response = await client.get(endpoint, params={"url": url}, headers=self._headers())
        except httpx.HTTPError as e:
            raise KeenableError(f"Keenable fetch request failed: {e!r}") from e

        _raise_for_status(response)

        try:
            data = response.json()
        except ValueError as e:
            raise KeenableError("Keenable fetch returned a non-JSON response") from e

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response from the Keenable fetch API: {data!r}")
        return data

    @staticmethod
    def _validate_fetch_url(url: str) -> None:
        """Client-side guard before sending a URL to the fetch endpoint."""
        if not url or not url.strip():
            raise ValueError("url must be a non-empty string")
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"only http(s) page URLs can be fetched, got scheme {parts.scheme!r}")
        host = (parts.hostname or "").lower()
        if host in _BLOCKED_FETCH_HOSTS:
            raise ValueError(f"refusing to fetch a private/internal host: {host!r}")

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        if self._api_key is not None:
            headers["X-API-Key"] = self._api_key
        return headers

    def _endpoint(self, keyed: str, public: str) -> str:
        path = keyed if self._api_key is not None else public
        return f"{self.base_url}{path}"

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True
        return self._client

    @staticmethod
    def format_results(results: list[dict[str, Any]]) -> str:
        """Render search results into a compact, model-friendly context block.

        Args:
            results: The list returned by :meth:`search`.

        Returns:
            A numbered, plain-text block suitable for injecting into a prompt.
        """
        if not results:
            return "No web search results were found."

        blocks = []
        for i, item in enumerate(results, start=1):
            title = item.get("title") or "(untitled)"
            url = item.get("url") or ""
            body = item.get("snippet") or item.get("description") or ""
            published = item.get("published_at")
            header = f"[{i}] {title}"
            if published:
                header += f" ({published})"
            block = header
            if url:
                block += f"\n    URL: {url}"
            if body:
                block += f"\n    {body.strip()}"
            blocks.append(block)
        return "\n\n".join(blocks)

    @staticmethod
    def format_page(page: dict[str, Any]) -> str:
        """Render a fetched page into a compact, model-friendly block.

        Args:
            page: The dict returned by :meth:`fetch`.

        Returns:
            A plain-text block (title, URL, then markdown content).
        """
        title = page.get("title") or "(untitled)"
        url = page.get("url") or ""
        content = (page.get("content") or "").strip()
        header = title if not url else f"{title}\nURL: {url}"
        return f"{header}\n\n{content}" if content else header


# --- LLM tool schemas -------------------------------------------------------
#
# All search filters are per-invocation properties so the model can vary them
# per query -- the whole point of giving an LLM a search tool. ``api_key`` and
# the base URL are NEVER exposed here (a secret / SSRF foothold respectively).

KEENABLE_SEARCH_FUNCTION_SCHEMA: FunctionSchema = FunctionSchema(
    name="keenable_search",
    description=(
        "Search the public web for up-to-date information via Keenable. "
        "Returns a list of results, each with a title, URL, and short snippet. "
        "Use this when the user asks about news, current events, or anything "
        "that may have changed recently or is beyond your training data."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "The natural-language search query to send to Keenable.",
        },
        "site": {
            "type": "string",
            "description": "Optional: restrict results to a single domain, e.g. 'github.com'.",
        },
        "published_after": {
            "type": "string",
            "description": "Optional: only pages published on/after this date (YYYY-MM-DD).",
        },
        "published_before": {
            "type": "string",
            "description": "Optional: only pages published on/before this date (YYYY-MM-DD).",
        },
        "acquired_after": {
            "type": "string",
            "description": "Optional: only pages indexed on/after this date (YYYY-MM-DD).",
        },
        "acquired_before": {
            "type": "string",
            "description": "Optional: only pages indexed on/before this date (YYYY-MM-DD).",
        },
    },
    required=["query"],
)

KEENABLE_FETCH_FUNCTION_SCHEMA: FunctionSchema = FunctionSchema(
    name="keenable_fetch",
    description=(
        "Fetch the full markdown content of a single web page via Keenable. "
        "Use this after `keenable_search` to read a specific result in full. "
        "Input is a single public http(s) URL."
    ),
    properties={
        "url": {
            "type": "string",
            "description": "The public http(s) URL of the page to fetch and read.",
        },
    },
    required=["url"],
)


def _resolve_client(app_resources: Any) -> tuple[KeenableSearchClient, bool]:
    """Resolve a :class:`KeenableSearchClient` from ``params.app_resources``.

    Returns:
        A tuple ``(client, owned)``. ``owned`` is ``True`` when the caller must
        close the client after use (i.e. it was built per-call because none was
        shared via ``PipelineTask(app_resources=...)``).
    """
    if isinstance(app_resources, KeenableSearchClient):
        return app_resources, False
    candidate = getattr(app_resources, "keenable_search_client", None)
    if isinstance(candidate, KeenableSearchClient):
        return candidate, False
    return KeenableSearchClient(), True


async def keenable_search(params: FunctionCallParams) -> None:
    """LLM tool handler that performs a Keenable web search.

    Looks for a shared :class:`KeenableSearchClient` on ``params.app_resources``
    (either the resource itself or a ``keenable_search_client`` attribute) and
    reuses it; otherwise builds a per-call keyless client and closes it after
    use. All per-invocation filters (``site`` and the date ranges) are read from
    ``params.arguments`` and forwarded to the search.

    The result delivered to ``result_callback`` is a JSON-friendly dict so it
    serializes cleanly across all LLM providers::

        {"results": [{"title": ..., "url": ..., "description": ...}, ...]}

    On failure, the handler returns ``{"error": "...", "results": []}`` instead
    of raising, so a rate limit or auth error lets the LLM recover gracefully
    rather than crashing the pipeline.
    """
    args = params.arguments
    query = str(args.get("query", "")).strip()
    if not query:
        await params.result_callback({"error": "Missing 'query' argument.", "results": []})
        return

    client, owned = _resolve_client(params.app_resources)
    try:
        results = await client.search(
            query,
            site=args.get("site"),
            published_after=args.get("published_after"),
            published_before=args.get("published_before"),
            acquired_after=args.get("acquired_after"),
            acquired_before=args.get("acquired_before"),
        )
        await params.result_callback({"results": results})
    except Exception as exc:
        logger.warning(f"keenable_search failed for query={query!r}: {exc}")
        await params.result_callback({"error": str(exc), "results": []})
    finally:
        if owned:
            await client.aclose()


async def keenable_fetch(params: FunctionCallParams) -> None:
    """LLM tool handler that fetches a page's markdown content via Keenable.

    Resolves a shared client the same way as :func:`keenable_search`. The result
    delivered to ``result_callback`` is the page dict (``{url, title, content,
    ...}``); on failure it returns ``{"error": "..."}`` so the LLM can recover.
    """
    url = str(params.arguments.get("url", "")).strip()
    if not url:
        await params.result_callback({"error": "Missing 'url' argument."})
        return

    client, owned = _resolve_client(params.app_resources)
    try:
        page = await client.fetch(url)
        await params.result_callback(page)
    except Exception as exc:
        logger.warning(f"keenable_fetch failed for url={url!r}: {exc}")
        await params.result_callback({"error": str(exc)})
    finally:
        if owned:
            await client.aclose()
