"""Tool functions the agents call: web search/scrape, calculator, market data, and the two
terminal tools (submit_findings, submit_verdict). Pure functions + their argument schemas —
the agents wrap them into LangChain StructuredTools."""

import ast
import operator

import requests
import structlog
import yfinance as yf
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from pipeline.config import MAX_SCRAPE_CHARS
from pipeline.config import MAX_SEARCH_CHARS
from pipeline.config import SERPER_API_KEY

logger = structlog.get_logger(__name__)

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0 Safari/537.36 equity-research-agent"
)


class _WebSearchInput(BaseModel):
    search_query: str = Field(description="What to search the web for.")


class _ScrapeInput(BaseModel):
    website_url: str = Field(description="URL of a promising page to fetch and read.")


class _SubmitFindingsInput(BaseModel):
    summary: str = Field(description="A thorough, self-contained summary of what you found that answers the question.")


def _web_search(search_query: str) -> str:
    """Google search via the Serper API (POST) — returns the top organic results as text."""
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY or "", "Content-Type": "application/json"},
        json={"q": search_query},
        timeout=20,
    )
    response.raise_for_status()
    results = response.json().get("organic", [])
    rendered = [f"{r.get('title', '')} — {r.get('link', '')}\n{r.get('snippet', '')}" for r in results[:6]]
    return ("\n\n".join(rendered) or "No results found.")[:MAX_SEARCH_CHARS]


def _web_scrape(website_url: str) -> str:
    """Fetch a page and extract its visible text (scripts/nav/boilerplate stripped)."""
    response = requests.get(website_url, headers={"User-Agent": _BROWSER_USER_AGENT}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)[:MAX_SCRAPE_CHARS]


def _submit_findings(summary: str) -> str:
    return summary


def build_web_tools() -> list[BaseTool]:
    """The web sub-agent's tools: Serper search, page scrape, and a terminal submit_findings.
    Search/scrape outputs are truncated so accumulated results stay under token-per-minute limits."""
    return [
        StructuredTool.from_function(
            func=_web_search,
            name="web_search",
            description="Search the web (Google via Serper) and return result snippets with links.",
            args_schema=_WebSearchInput,
        ),
        StructuredTool.from_function(
            func=_web_scrape,
            name="web_scrape",
            description="Fetch a URL and return its extracted text. Use on the most relevant search result.",
            args_schema=_ScrapeInput,
        ),
        StructuredTool.from_function(
            func=_submit_findings,
            name="submit_findings",
            description="Return your final researched summary. Call this exactly once, when you have enough info.",
            args_schema=_SubmitFindingsInput,
        ),
    ]


_ARITHMETIC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class _CalculatorInput(BaseModel):
    expression: str = Field(description="A pure arithmetic expression, e.g. '(120.5 - 98) / 98 * 100'.")


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a pure-arithmetic AST node. No names, calls, or attributes — numbers and the
    operators in `_ARITHMETIC_OPS` only, so there is no code-execution surface (unlike eval)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPS:
        return _ARITHMETIC_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_OPS:
        return _ARITHMETIC_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


def _calculator(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"


class _MarketDataInput(BaseModel):
    ticker: str = Field(description="A stock ticker symbol, e.g. AAPL, MSFT, NVDA.")


def _format_market_value(key: str, value: object) -> str:
    if isinstance(value, float) and key in {"Dividend yield"}:
        return f"{value:.2%}"
    if isinstance(value, (int, float)) and key == "Market cap":
        return f"{value:,.0f}"
    return str(value)


def _get_market_data(ticker: str) -> str:
    """Live quote and headline fundamentals for a ticker via yfinance."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        logger.warning("Market data lookup failed for %s: %s", ticker, e)
        return f"Could not fetch market data for '{ticker}': {e}"

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not info or price is None:
        return f"No market data found for ticker '{ticker}'. Check the symbol (e.g. AAPL, MSFT)."

    fields = {
        "Name": info.get("shortName") or info.get("longName"),
        "Price": price,
        "Currency": info.get("currency"),
        "Market cap": info.get("marketCap"),
        "Trailing P/E": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "Price/Book": info.get("priceToBook"),
        "52-week high": info.get("fiftyTwoWeekHigh"),
        "52-week low": info.get("fiftyTwoWeekLow"),
        "Dividend yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
    }
    lines = [f"Market data for {ticker.upper()}:"]
    lines += [f"- {k}: {_format_market_value(k, v)}" for k, v in fields.items() if v is not None]
    return "\n".join(lines)


class _VerdictInput(BaseModel):
    verdict: str = Field(
        description="One of 'supported', 'partially supported', or 'unsupported', plus a one-line justification."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Specific claims in the draft answer NOT backed by the evidence (empty if all are supported).",
    )


def _submit_verdict(verdict: str, unsupported_claims: list[str] | None = None) -> str:
    lines = [f"Verification: {verdict}"]
    if unsupported_claims:
        lines.append("Flagged (not supported by the sources):")
        lines.extend(f"  - {claim}" for claim in unsupported_claims)
    return "\n".join(lines)
