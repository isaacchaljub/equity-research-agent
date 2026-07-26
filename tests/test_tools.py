from unittest.mock import patch

from pipeline.tools import _calculator
from pipeline.tools import _get_market_data
from pipeline.tools import _web_scrape
from pipeline.tools import _web_search


def test_calculator_arithmetic():
    assert _calculator("2 + 3 * 4") == "2 + 3 * 4 = 14"
    assert _calculator("(120.5 - 98) / 98 * 100").startswith("(120.5 - 98) / 98 * 100 = 22.95")


def test_calculator_rejects_non_arithmetic():
    assert "Could not evaluate" in _calculator("__import__('os').system('ls')")
    assert "Could not evaluate" in _calculator("price * 2")


@patch("pipeline.tools.yf.Ticker")
def test_market_data_formats_fields(mock_ticker):
    mock_ticker.return_value.info = {
        "shortName": "Apple Inc.",
        "currentPrice": 190.0,
        "currency": "USD",
        "marketCap": 3_000_000_000_000,
        "trailingPE": 31.5,
    }
    out = _get_market_data("AAPL")
    assert "Apple Inc." in out
    assert "190.0" in out
    assert "Trailing P/E: 31.5" in out
    assert "Market cap: 3,000,000,000,000" in out


@patch("pipeline.tools.yf.Ticker")
def test_market_data_handles_unknown_ticker(mock_ticker):
    mock_ticker.return_value.info = {}
    assert "No market data" in _get_market_data("ZZZZ")


@patch("pipeline.tools.yf.Ticker")
def test_market_data_handles_lookup_error(mock_ticker):
    mock_ticker.side_effect = RuntimeError("network down")
    assert "Could not fetch market data" in _get_market_data("AAPL")


@patch("pipeline.tools.requests.post")
def test_web_search_renders_organic_results(mock_post):
    mock_post.return_value.raise_for_status.return_value = None
    mock_post.return_value.json.return_value = {"organic": [
        {"title": "Apple Q4 results", "link": "https://example.com/a", "snippet": "Revenue up 5%."},
        {"title": "AAPL analysis", "link": "https://example.com/b", "snippet": "Margins improved."},
    ]}
    out = _web_search("apple earnings")
    assert "Apple Q4 results" in out
    assert "https://example.com/a" in out
    assert "Revenue up 5%." in out


@patch("pipeline.tools.requests.get")
def test_web_scrape_extracts_visible_text(mock_get):
    mock_get.return_value.raise_for_status.return_value = None
    mock_get.return_value.content = (
        b"<html><body><script>tracking()</script><p>Net income was $50B.</p>"
        b"<footer>cookie banner</footer></body></html>"
    )
    out = _web_scrape("https://example.com/report")
    assert "Net income was $50B." in out
    assert "cookie banner" not in out
    assert "tracking" not in out
