#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Keenable web search integration."""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, NotGiven


class TestBuiltinToolsWithNotGiven(unittest.TestCase):
    """Test that builtin_tools are injected when user specifies no tools."""

    def setUp(self):
        self.adapter = OpenAILLMAdapter()
        self.search_schema = FunctionSchema(
            name="search_web_pages",
            description="Search the web",
            properties={"query": {"type": "string", "description": "Search query"}},
            required=["query"],
        )

    def test_builtin_tools_injected_when_tools_not_given(self):
        """When tools=NOT_GIVEN and builtin_tools exist, they should be sent."""
        self.adapter.builtin_tools["search_web_pages"] = self.search_schema
        result = self.adapter.from_standard_tools(NOT_GIVEN)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search_web_pages"

    def test_builtin_tools_injected_when_tools_none(self):
        """When tools=None and builtin_tools exist, they should be sent."""
        self.adapter.builtin_tools["search_web_pages"] = self.search_schema
        result = self.adapter.from_standard_tools(None)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_builtin_tools_merged_with_existing_tools(self):
        """When tools=ToolsSchema and builtin_tools exist, all should be merged."""
        user_tool = FunctionSchema(
            name="get_weather",
            description="Get weather",
            properties={"city": {"type": "string"}},
            required=["city"],
        )
        user_tools = ToolsSchema(standard_tools=[user_tool])

        self.adapter.builtin_tools["search_web_pages"] = self.search_schema
        result = self.adapter.from_standard_tools(user_tools)
        assert isinstance(result, list)
        assert len(result) == 2
        names = {t["function"]["name"] for t in result}
        assert names == {"get_weather", "search_web_pages"}

    def test_no_builtin_tools_not_given_returns_not_given(self):
        """When no builtin_tools and tools=NOT_GIVEN, should return NOT_GIVEN."""
        result = self.adapter.from_standard_tools(NOT_GIVEN)
        assert isinstance(result, NotGiven)


class TestKeenableWebSearchInit(unittest.TestCase):
    """Test KeenableWebSearch initialization and configuration."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_no_api_key(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch()
        assert search._api_key is None

    @patch.dict(os.environ, {"KEENABLE_API_KEY": "test-key"})
    def test_api_key_from_env(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch()
        assert search._api_key == "test-key"

    def test_explicit_api_key(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch(api_key="explicit-key")
        assert search._api_key == "explicit-key"

    @patch.dict(os.environ, {"KEENABLE_API_KEY": "  "})
    def test_blank_api_key_treated_as_none(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch()
        assert search._api_key is None

    def test_headers_without_api_key(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch()
        headers = search._build_headers()
        assert "User-Agent" in headers
        assert headers["User-Agent"].startswith("pipecat/")
        assert "X-API-Key" not in headers

    def test_headers_with_api_key(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch(api_key="my-key")
        headers = search._build_headers()
        assert headers["X-API-Key"] == "my-key"

    def test_not_connected_raises(self):
        from pipecat.services.keenable.search import KeenableWebSearch

        search = KeenableWebSearch()
        with self.assertRaises(RuntimeError):
            search._ensure_connected()


if __name__ == "__main__":
    unittest.main()
