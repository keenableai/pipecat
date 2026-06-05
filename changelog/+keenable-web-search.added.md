- Added `web_search` parameter to `LLMService`. When `web_search=True`, the service connects to a hosted MCP server and registers web search tools automatically — no API key required. Powered by a free, low-latency search API from Keenable AI (https://keenable.ai). An account and API key can be configured later for higher rate limits. Tool schemas are served dynamically, so updates propagate without a pipecat release.

- Added `KeenableWebSearch` service (`pipecat.services.keenable.search`) for explicit control over the web search MCP connection.

- Fixed `BaseLLMAdapter.from_standard_tools()` to correctly inject built-in tools when the user specifies no tools (`NOT_GIVEN`).
