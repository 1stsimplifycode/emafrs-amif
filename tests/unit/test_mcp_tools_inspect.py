"""Unit tests for the helper MCP server's project-inspection tools."""
from emafrs.mcp_server import tools_inspect


def test_list_configured_sources_reports_all_seven():
    result = tools_inspect.list_configured_sources()
    assert result["ok"] is True
    assert set(result["sources"]) == {
        "yfinance", "alphavantage", "finnhub", "newsapi", "youtube", "fred", "sec_edgar"
    }


def test_cache_stats_returns_ok_even_when_empty():
    result = tools_inspect.cache_stats()
    assert result["ok"] is True
    assert "providers" in result


def test_list_docs_finds_traceability_doc():
    result = tools_inspect.list_docs()
    assert result["ok"] is True
    names = [d["file"] for d in result["docs"]]
    assert "PDF_TO_IMPLEMENTATION_TRACEABILITY.md" in names


def test_repo_status_returns_branch_and_commits():
    result = tools_inspect.repo_status()
    assert result["ok"] is True
    assert result["branch"]
    assert isinstance(result["recent_commits"], list)


def test_run_tests_rejects_arbitrary_marker_expression():
    result = tools_inspect.run_tests(marker="'; rm -rf /'")
    assert result["ok"] is False
    assert "marker must be one of" in result["error"]
