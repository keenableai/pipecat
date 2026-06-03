#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat-native tests for :mod:`pipecat.services.keenable.search`.

These run inside the pipecat dev environment (real imports) and focus on the
framework integration surface: the two ``FunctionSchema`` objects and the
``keenable_search`` / ``keenable_fetch`` tool handlers driven through a
``FunctionCallParams``-shaped object. The exhaustive transport / error-path
matrix and live e2e live in the keenable-integrations repo's ``tests/``.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from pipecat.services.keenable.search import (
    KEENABLE_FETCH_FUNCTION_SCHEMA,
    KEENABLE_SEARCH_FUNCTION_SCHEMA,
    KeenableRateLimitError,
    KeenableSearchClient,
    keenable_fetch,
    keenable_search,
)

_FAKE_RESULTS = [
    {
        "title": "Pipecat releases",
        "url": "https://example.com/pipecat",
        "description": "Latest pipecat releases.",
        "published_at": "2026-05-01T00:00:00Z",
    }
]
_FAKE_PAGE = {
    "url": "https://example.com/pipecat",
    "title": "Pipecat releases",
    "content": "# Pipecat\n\nLatest releases.",
}


def _make_response(json_payload, status_code: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json = MagicMock(return_value=json_payload)
    return response


def _params(arguments, app_resources=None) -> MagicMock:
    """A FunctionCallParams-shaped stand-in (arguments/result_callback/app_resources)."""
    params = MagicMock()
    params.arguments = arguments
    params.result_callback = AsyncMock()
    params.app_resources = app_resources
    return params


# --- function schemas -------------------------------------------------------


def test_search_schema_exposes_per_invocation_filters():
    """All search filters must be per-invocation properties (the #1 mistake)."""
    schema = KEENABLE_SEARCH_FUNCTION_SCHEMA
    assert schema.name == "keenable_search"
    assert schema.required == ["query"]
    for prop in (
        "query",
        "site",
        "published_after",
        "published_before",
        "acquired_after",
        "acquired_before",
    ):
        assert prop in schema.properties
    # The API has no max_results; the schema must not invent one.
    assert "max_results" not in schema.properties


def test_fetch_schema_shape():
    schema = KEENABLE_FETCH_FUNCTION_SCHEMA
    assert schema.name == "keenable_fetch"
    assert schema.required == ["url"]
    assert schema.properties["url"]["type"] == "string"


# --- keenable_search handler ------------------------------------------------


@pytest.mark.asyncio
async def test_search_handler_delivers_results_dict():
    client = MagicMock(spec=KeenableSearchClient)
    client.search = AsyncMock(return_value=_FAKE_RESULTS)
    params = _params({"query": "pipecat releases"}, app_resources=client)

    await keenable_search(params)

    client.search.assert_awaited_once()
    assert client.search.call_args.args == ("pipecat releases",)
    params.result_callback.assert_awaited_once_with({"results": _FAKE_RESULTS})


@pytest.mark.asyncio
async def test_search_handler_forwards_per_call_filters():
    client = MagicMock(spec=KeenableSearchClient)
    client.search = AsyncMock(return_value=[])
    params = _params(
        {"query": "local llm", "site": "github.com", "published_after": "2026-01-01"},
        app_resources=client,
    )

    await keenable_search(params)

    kwargs = client.search.call_args.kwargs
    assert kwargs["site"] == "github.com"
    assert kwargs["published_after"] == "2026-01-01"


@pytest.mark.asyncio
async def test_search_handler_resolves_client_from_attribute():
    client = MagicMock(spec=KeenableSearchClient)
    client.search = AsyncMock(return_value=[])
    container = MagicMock()
    container.keenable_search_client = client
    params = _params({"query": "x"}, app_resources=container)

    await keenable_search(params)

    client.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_handler_returns_error_string_does_not_raise():
    client = MagicMock(spec=KeenableSearchClient)
    client.search = AsyncMock(side_effect=KeenableRateLimitError("slow down (429)"))
    params = _params({"query": "x"}, app_resources=client)

    await keenable_search(params)  # must not raise

    payload = params.result_callback.call_args.args[0]
    assert payload["results"] == []
    assert "429" in payload["error"]


@pytest.mark.asyncio
async def test_search_handler_rejects_empty_query():
    params = _params({"query": "   "}, app_resources=None)
    await keenable_search(params)
    payload = params.result_callback.call_args.args[0]
    assert payload["results"] == []
    assert "query" in payload["error"].lower()


# --- keenable_fetch handler -------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_handler_delivers_page():
    client = MagicMock(spec=KeenableSearchClient)
    client.fetch = AsyncMock(return_value=_FAKE_PAGE)
    params = _params({"url": "https://example.com/pipecat"}, app_resources=client)

    await keenable_fetch(params)

    client.fetch.assert_awaited_once_with("https://example.com/pipecat")
    params.result_callback.assert_awaited_once_with(_FAKE_PAGE)


@pytest.mark.asyncio
async def test_fetch_handler_returns_error_string_does_not_raise():
    client = MagicMock(spec=KeenableSearchClient)
    client.fetch = AsyncMock(side_effect=ValueError("refusing to fetch a private/internal host"))
    params = _params({"url": "http://127.0.0.1/admin"}, app_resources=client)

    await keenable_fetch(params)  # must not raise

    payload = params.result_callback.call_args.args[0]
    assert "private/internal" in payload["error"]


@pytest.mark.asyncio
async def test_fetch_handler_rejects_empty_url():
    params = _params({}, app_resources=None)
    await keenable_fetch(params)
    payload = params.result_callback.call_args.args[0]
    assert "url" in payload["error"].lower()


# --- client basics (full matrix lives in the keenable-integrations repo) -----


@pytest.mark.asyncio
async def test_client_keyless_uses_public_endpoint(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=_make_response({"results": _FAKE_RESULTS}))

    client = KeenableSearchClient(client=fake_http)
    assert client.has_api_key is False
    results = await client.search("anything")

    assert results == _FAKE_RESULTS
    url = fake_http.post.call_args.args[0]
    assert url.endswith("/v1/search/public")
    assert "X-API-Key" not in fake_http.post.call_args.kwargs["headers"]
    assert fake_http.post.call_args.kwargs["headers"]["User-Agent"].startswith("keenable-pipecat/")


@pytest.mark.asyncio
async def test_client_keyed_uses_authenticated_endpoint():
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=_make_response({"results": []}))

    client = KeenableSearchClient(api_key="secret-key", client=fake_http)
    await client.search("anything")

    url = fake_http.post.call_args.args[0]
    assert url.endswith("/v1/search")
    assert fake_http.post.call_args.kwargs["headers"]["X-API-Key"] == "secret-key"


def test_client_does_not_leak_key_in_repr_or_vars():
    client = KeenableSearchClient(api_key="super-secret")
    assert "super-secret" not in repr(client)
    assert "super-secret" not in str(vars(client))


def test_client_has_no_base_url_param():
    import inspect

    assert "base_url" not in inspect.signature(KeenableSearchClient).parameters
