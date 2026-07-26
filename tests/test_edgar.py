from unittest.mock import MagicMock
from unittest.mock import patch

import pipeline.retrieval as retrieval
from pipeline.agents import AnalystAgent
from pipeline.retrieval import fetch_10k_chunks
from pipeline.retrieval import _latest_10k
from pipeline.retrieval import _resolve_cik


def _resp(json_data=None, content=b""):
    response = MagicMock()
    response.json.return_value = json_data
    response.content = content
    response.raise_for_status.return_value = None
    return response


def _fake_get(url, **kwargs):
    if "company_tickers.json" in url:
        return _resp(json_data={"0": {"ticker": "AAPL", "cik_str": 320193}})
    if "submissions/CIK" in url:
        return _resp(json_data={"filings": {"recent": {
            "form": ["8-K", "10-K"],
            "accessionNumber": ["0000320193-24-000099", "0000320193-24-000123"],
            "primaryDocument": ["a8k.htm", "aapl-20240928.htm"],
            "filingDate": ["2024-10-01", "2024-11-01"],
        }}})
    return _resp(content=b"<html><body><h1>Risk Factors</h1><p>Competition is intense. Revenue was $383B.</p></body></html>")


def setup_function():
    retrieval._ticker_to_cik = {}


def test_resolve_cik():
    with patch("pipeline.retrieval.requests.get", side_effect=_fake_get):
        assert _resolve_cik("aapl") == "0000320193"
        assert _resolve_cik("ZZZZ") is None


def test_latest_10k_skips_other_forms():
    with patch("pipeline.retrieval.requests.get", side_effect=_fake_get):
        filing = _latest_10k("0000320193")
        assert filing["accession"] == "0000320193-24-000123"
        assert filing["date"] == "2024-11-01"


def test_fetch_10k_chunks_extracts_text():
    with patch("pipeline.retrieval.requests.get", side_effect=_fake_get):
        chunks, filed = fetch_10k_chunks("AAPL")
        assert filed == "2024-11-01"
        assert chunks
        assert any("Revenue was $383B" in c.page_content for c in chunks)


def test_fetch_filing_tool_indexes_into_agent():
    agent = AnalystAgent()
    assert agent.vector_db is None
    with patch("pipeline.retrieval.requests.get", side_effect=_fake_get):
        message = agent._fetch_filing("AAPL")
    assert "Loaded AAPL 10-K" in message
    assert agent.vector_db is not None
    assert "383B" in agent._search_filings("revenue") or "Revenue" in agent._search_filings("revenue")


def test_fetch_filing_unknown_ticker_is_graceful():
    agent = AnalystAgent()
    with patch("pipeline.retrieval.requests.get", side_effect=_fake_get):
        assert "not found" in agent._fetch_filing("ZZZZ").lower()


def test_search_filings_before_any_load():
    assert "fetch_filing" in AnalystAgent()._search_filings("anything")
