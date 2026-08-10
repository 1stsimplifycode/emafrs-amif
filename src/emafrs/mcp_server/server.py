"""Helper MCP server for building/debugging EMAFRS.

Not part of the research system or any production deployment -- a local dev aid exposing
(a) the DAL as callable tools, and (b) project-inspection tools (cache stats, docs list,
git status, test runner). Run with:

    python -m emafrs.mcp_server

and point an MCP client (e.g. Claude Desktop / Claude Code) at this command over stdio.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from emafrs.mcp_server import tools_data, tools_inspect

server = MCPServer(
    name="emafrs-dev-tools",
    version="0.1.0",
    instructions=(
        "Development helper for the EMAFRS/AMIF reconstruction project. Provides direct "
        "access to the project's Data Acquisition Layer (yfinance, Alpha Vantage, Finnhub, "
        "NewsAPI, YouTube, FRED, SEC EDGAR) and project-inspection tools (cache stats, "
        "docs listing, git status, test runner). Every data tool returns {'ok': bool, ...}; "
        "'ok': False with an 'error' message means the source is unavailable or unconfigured "
        "-- never treat a missing/failed tool call as zero or empty data."
    ),
)

_DATA_TOOLS = [
    tools_data.get_ohlcv,
    tools_data.get_alphavantage_daily,
    tools_data.get_quote,
    tools_data.get_company_news,
    tools_data.search_news,
    tools_data.search_youtube_videos,
    tools_data.get_youtube_transcript,
    tools_data.get_fred_series,
    tools_data.sec_ticker_to_cik,
    tools_data.sec_full_text_search,
    tools_data.check_data_source_health,
]

_INSPECT_TOOLS = [
    tools_inspect.list_configured_sources,
    tools_inspect.cache_stats,
    tools_inspect.list_docs,
    tools_inspect.repo_status,
    tools_inspect.run_tests,
]

for fn in _DATA_TOOLS + _INSPECT_TOOLS:
    server.add_tool(fn)


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
