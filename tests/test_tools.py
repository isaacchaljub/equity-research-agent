from unittest.mock import patch

from pipeline.analyst import _calculator
from pipeline.analyst import _get_market_data


def test_calculator_arithmetic():
    assert _calculator("2 + 3 * 4") == "2 + 3 * 4 = 14"
    assert _calculator("(120.5 - 98) / 98 * 100").startswith("(120.5 - 98) / 98 * 100 = 22.95")


def test_calculator_rejects_non_arithmetic():
    assert "Could not evaluate" in _calculator("__import__('os').system('ls')")
    assert "Could not evaluate" in _calculator("price * 2")


@patch("pipeline.analyst.yf.Ticker")
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


@patch("pipeline.analyst.yf.Ticker")
def test_market_data_handles_unknown_ticker(mock_ticker):
    mock_ticker.return_value.info = {}
    assert "No market data" in _get_market_data("ZZZZ")


@patch("pipeline.analyst.yf.Ticker")
def test_market_data_handles_lookup_error(mock_ticker):
    mock_ticker.side_effect = RuntimeError("network down")
    assert "Could not fetch market data" in _get_market_data("AAPL")
