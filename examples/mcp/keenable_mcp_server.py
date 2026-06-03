#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Minimal stdio MCP server that exposes the Keenable web search API.

Wraps :class:`pipecat.services.keenable.search.KeenableSearchClient` and
exposes two tools:

- ``keenable_search`` - search the public web.
- ``keenable_fetch``  - fetch the markdown content of an indexed URL.

Speaks MCP over stdio so any MCP client (including
``pipecat.services.mcp_service.MCPClient`` with ``StdioServerParameters``) can
connect. ``KEENABLE_API_KEY`` is optional -- without it the keyless public
endpoints (free tier) are used.

Run directly for debugging::

    python examples/mcp/keenable_mcp_server.py
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from pipecat.services.keenable.search import KeenableSearchClient

mcp = FastMCP("keenable-search")

_client: KeenableSearchClient | None = None


def _get_client() -> KeenableSearchClient:
    global _client
    if _client is None:
        _client = KeenableSearchClient()
    return _client


@mcp.tool(
    description=(
        "Search the public web via Keenable. Returns up to ten ranked results, "
        "each with a title, URL, and short snippet. Use for news, current "
        "events, or anything that may have changed recently."
    ),
)
async def keenable_search(query: str) -> dict[str, Any]:
    """Search the public web for ``query`` and return ranked results."""
    return {"results": await _get_client().search(query)}


@mcp.tool(
    description=(
        "Fetch the markdown content of a single indexed URL via Keenable. "
        "Use after `keenable_search` to read a specific result in full."
    ),
)
async def keenable_fetch(url: str) -> dict[str, Any]:
    """Fetch and return the markdown content of ``url`` from the Keenable index."""
    return await _get_client().fetch(url)


if __name__ == "__main__":
    mcp.run("stdio")
