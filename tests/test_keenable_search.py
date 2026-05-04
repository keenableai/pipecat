#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for :mod:`pipecat.services.keenable.search`."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pipecat.services.keenable.search import (
    DEFAULT_BASE_URL,
    FETCH_PATH,
    KEENABLE_SEARCH_FUNCTION_SCHEMA,
    SEARCH_PATH,
    KeenableFetchResult,
    KeenableSearchClient,
    KeenableSearchResult,
    keenable_search,
)


def _make_response(json_payload: dict, status_code: int = 200) -> MagicMock:
    """Build a mock response object compatible with httpx behavior."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json = MagicMock(return_value=json_payload)
    if status_code >= 400:
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("error", request=MagicMock(), response=response)
        )
    else:
        response.raise_for_status = MagicMock(return_value=None)
    return response


def test_init_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "env-key")
    client = KeenableSearchClient()
    assert client._api_key == "env-key"


def test_init_raises_when_no_key_available(monkeypatch):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="KEENABLE_API_KEY"):
        KeenableSearchClient()


def test_init_strips_trailing_slash_from_base_url():
    client = KeenableSearchClient(api_key="k", base_url="https://api.keenable.ai/")
    assert client._base_url == "https://api.keenable.ai"


@pytest.mark.asyncio
async def test_search_posts_correct_request():
    """search() must POST to /v1/search with the query body and X-API-Key header."""
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=_make_response({"results": []}))

    client = KeenableSearchClient(api_key="secret-key", client=fake_http)
    await client.search("typescript best practices")

    fake_http.post.assert_awaited_once()
    args, kwargs = fake_http.post.call_args
    assert args[0] == f"{DEFAULT_BASE_URL}{SEARCH_PATH}"
    assert kwargs["json"] == {"query": "typescript best practices"}
    assert kwargs["headers"]["X-API-Key"] == "secret-key"
    assert kwargs["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_search_parses_results():
    payload = {
        "results": [
            {
                "title": "TypeScript Best Practices",
                "url": "https://example.com/ts",
                "description": "A guide.",
            },
            {
                "title": "Another",
                "url": "https://example.com/other",
                "description": "More info.",
            },
        ]
    }
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=_make_response(payload))

    client = KeenableSearchClient(api_key="k", client=fake_http)
    results = await client.search("anything")

    assert len(results) == 2
    assert all(isinstance(r, KeenableSearchResult) for r in results)
    assert results[0].title == "TypeScript Best Practices"
    assert results[0].url == "https://example.com/ts"
    assert results[1].description == "More info."


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_results_key():
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=_make_response({}))

    client = KeenableSearchClient(api_key="k", client=fake_http)
    assert await client.search("anything") == []


@pytest.mark.asyncio
async def test_search_raises_for_http_error():
    fake_http = MagicMock()
    fake_http.post = AsyncMock(return_value=_make_response({}, status_code=401))

    client = KeenableSearchClient(api_key="bad", client=fake_http)
    with pytest.raises(httpx.HTTPStatusError):
        await client.search("anything")


@pytest.mark.asyncio
async def test_fetch_uses_get_with_url_param():
    payload = {
        "url": "https://example.com/ts",
        "title": "Title",
        "content": "# Markdown",
    }
    fake_http = MagicMock()
    fake_http.get = AsyncMock(return_value=_make_response(payload))

    client = KeenableSearchClient(api_key="secret", client=fake_http)
    result = await client.fetch("https://example.com/ts")

    fake_http.get.assert_awaited_once()
    args, kwargs = fake_http.get.call_args
    assert args[0] == f"{DEFAULT_BASE_URL}{FETCH_PATH}"
    assert kwargs["params"] == {"url": "https://example.com/ts"}
    assert kwargs["headers"] == {"X-API-Key": "secret"}
    assert isinstance(result, KeenableFetchResult)
    assert result.title == "Title"
    assert result.content == "# Markdown"


@pytest.mark.asyncio
async def test_aclose_closes_owned_client(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "k")
    client = KeenableSearchClient()
    inner = client._ensure_client()
    inner.aclose = AsyncMock()
    await client.aclose()
    inner.aclose.assert_awaited_once()
    assert client._client is None


@pytest.mark.asyncio
async def test_aclose_does_not_close_borrowed_client():
    borrowed = MagicMock()
    borrowed.aclose = AsyncMock()
    client = KeenableSearchClient(api_key="k", client=borrowed)
    await client.aclose()
    borrowed.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_context_manager(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "k")
    fake_aclose = AsyncMock()
    with patch("httpx.AsyncClient") as mock_cls:
        mock_cls.return_value.aclose = fake_aclose
        async with KeenableSearchClient() as client:
            assert client._client is mock_cls.return_value
        fake_aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_keenable_search_tool_calls_result_callback():
    callback = AsyncMock()
    fake_client = MagicMock(spec=KeenableSearchClient)
    fake_client.search = AsyncMock(
        return_value=[KeenableSearchResult(title="t", url="https://x", description="d")]
    )

    params = MagicMock()
    params.arguments = {"query": "hello"}
    params.result_callback = callback
    params.app_resources = fake_client

    await keenable_search(params)

    fake_client.search.assert_awaited_once_with("hello")
    callback.assert_awaited_once()
    payload = callback.call_args.args[0]
    assert payload == {"results": [{"title": "t", "url": "https://x", "description": "d"}]}


@pytest.mark.asyncio
async def test_keenable_search_tool_resolves_client_from_attribute():
    callback = AsyncMock()
    fake_client = MagicMock(spec=KeenableSearchClient)
    fake_client.search = AsyncMock(return_value=[])

    container = MagicMock()
    container.keenable_search_client = fake_client

    params = MagicMock()
    params.arguments = {"query": "x"}
    params.result_callback = callback
    params.app_resources = container

    await keenable_search(params)

    fake_client.search.assert_awaited_once_with("x")


@pytest.mark.asyncio
async def test_keenable_search_tool_handles_error_gracefully():
    callback = AsyncMock()
    fake_client = MagicMock(spec=KeenableSearchClient)
    fake_client.search = AsyncMock(side_effect=httpx.HTTPError("boom"))

    params = MagicMock()
    params.arguments = {"query": "hello"}
    params.result_callback = callback
    params.app_resources = fake_client

    # Should not raise.
    await keenable_search(params)

    payload = callback.call_args.args[0]
    assert payload["results"] == []
    assert "boom" in payload["error"]


@pytest.mark.asyncio
async def test_keenable_search_tool_rejects_empty_query():
    callback = AsyncMock()
    params = MagicMock()
    params.arguments = {"query": "   "}
    params.result_callback = callback
    params.app_resources = None

    await keenable_search(params)

    payload = callback.call_args.args[0]
    assert payload["results"] == []
    assert "query" in payload["error"].lower()


@pytest.mark.asyncio
async def test_keenable_search_tool_builds_per_call_client(monkeypatch):
    """When no client is provided, the tool builds and closes its own."""
    monkeypatch.setenv("KEENABLE_API_KEY", "env-key")
    callback = AsyncMock()

    built_clients: list[KeenableSearchClient] = []
    real_init = KeenableSearchClient.__init__

    def tracking_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built_clients.append(self)

    with patch.object(KeenableSearchClient, "__init__", tracking_init):
        with patch.object(
            KeenableSearchClient, "search", AsyncMock(return_value=[])
        ) as search_mock:
            with patch.object(KeenableSearchClient, "aclose", AsyncMock()) as aclose_mock:
                params = MagicMock()
                params.arguments = {"query": "hi"}
                params.result_callback = callback
                params.app_resources = None

                await keenable_search(params)

                search_mock.assert_awaited_once_with("hi")
                aclose_mock.assert_awaited_once()

    assert len(built_clients) == 1


def test_function_schema_shape():
    """The exported FunctionSchema declares a single required `query` string."""
    schema = KEENABLE_SEARCH_FUNCTION_SCHEMA
    assert schema.name == "keenable_search"
    assert schema.required == ["query"]
    assert schema.properties["query"]["type"] == "string"
