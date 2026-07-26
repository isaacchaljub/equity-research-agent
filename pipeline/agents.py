"""The three agents, all subclassing the engine's BaseAgent:

- WebResearchAgent — search -> scrape -> submit_findings (a sub-agent, delegated to as a tool).
- VerifierAgent    — fact-checks a draft answer against the evidence via submit_verdict.
- AnalystAgent     — the orchestrator: fetch/search filings, market data, calculator, research_web.
"""

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from pipeline.config import MAX_FILINGS_CHARS
from pipeline.config import global_embeddings
from pipeline.config import orchestrator_llm
from pipeline.config import web_backup_llm
from pipeline.config import web_llm
from pipeline.engine import BaseAgent
from pipeline.retrieval import fetch_10k_chunks
from pipeline.tools import _calculator
from pipeline.tools import _CalculatorInput
from pipeline.tools import _get_market_data
from pipeline.tools import _MarketDataInput
from pipeline.tools import _submit_verdict
from pipeline.tools import _VerdictInput
from pipeline.tools import build_web_tools

logger = structlog.get_logger(__name__)


_WEB_RESEARCH_SYSTEM_PROMPT = """You are a financial-news research assistant. For the user's question:
1. Call web_search to find relevant, recent pages (news, press releases, analyst notes).
2. Call web_scrape on the most promising result(s) to read the actual content.
3. When you have enough information, call submit_findings with a thorough, self-contained summary
   that directly answers the question, citing the key facts and their sources.
Always finish by calling submit_findings — do not answer in plain text."""


class WebResearchAgent(BaseAgent):
    """Bounded search -> scrape -> summarize loop behind the submit_findings terminal tool.

    Tries Gemini first, waits 20s then 60s on failure, then falls back to Groq's gpt-oss-120b."""

    def __init__(self) -> None:
        super().__init__(
            llm=web_llm,
            tools=build_web_tools(),
            system_prompt=_WEB_RESEARCH_SYSTEM_PROMPT,
            final_tool_names=["submit_findings"],
            max_iter=6,
            backup_llm=web_backup_llm,
            retry_waits=(20, 60),
        )

    def _process_output(self) -> str:
        if self._final_payload is not None:
            return self._final_payload
        last_ai = next((m for m in reversed(self.messages) if isinstance(m, AIMessage)), None)
        return str(last_ai.content) if last_ai else "No web findings."


_VERIFIER_SYSTEM_PROMPT = """You are a meticulous fact-checker for an equity-research assistant.

You are given a QUESTION, a DRAFT ANSWER, and the EVIDENCE (the raw tool outputs the answer was
built from: filing excerpts, market data, calculations, web findings). Check every factual and
numeric claim in the draft answer against the evidence:
- A claim is supported only if the evidence directly backs it. Do not use outside knowledge.
- Numbers must match the evidence (small rounding is fine).
Then call submit_verdict with an overall verdict and a list of any specific claims that are not
supported by the evidence. Always finish by calling submit_verdict."""


class VerifierAgent(BaseAgent):
    """Second agent: checks a draft answer's claims against the evidence and returns a verdict via
    the submit_verdict terminal tool. It calls no research tools — it only judges."""

    def __init__(self) -> None:
        super().__init__(
            llm=orchestrator_llm,
            tools=[
                StructuredTool.from_function(
                    func=_submit_verdict,
                    name="submit_verdict",
                    description="Report your fact-check: an overall verdict and any unsupported claims.",
                    args_schema=_VerdictInput,
                )
            ],
            system_prompt=_VERIFIER_SYSTEM_PROMPT,
            final_tool_names=["submit_verdict"],
            max_iter=3,
        )

    def _process_output(self) -> str:
        if self._final_payload is not None:
            return self._final_payload
        return "Verification: inconclusive (verifier did not return a verdict)."


class _SearchFilingsInput(BaseModel):
    query: str = Field(description="A focused search query for the loaded filings knowledge base.")


class _FetchFilingInput(BaseModel):
    ticker: str = Field(description="Ticker whose latest annual report (10-K) to pull from SEC EDGAR, e.g. AAPL.")


class _ResearchWebInput(BaseModel):
    query: str = Field(description="The news/context question to research on the web.")


_ANALYST_SYSTEM_PROMPT = """You are an equity-research analyst assistant having a conversation with the user.

You have five tools:
- fetch_filing: download a company's latest annual report (10-K) from SEC EDGAR by ticker and index it for search.
- search_filings: semantic search over the filings you have loaded (fundamentals, risks, strategy, segments).
- get_market_data: live price and headline fundamentals (P/E, market cap, 52-week range, ...) for a ticker.
- calculator: evaluate an arithmetic expression (use it for any ratio or growth-rate math — do not do arithmetic in your head).
- research_web: delegate to a web-research agent for recent news or anything the filings and market data do not cover.

How to work:
1. If the question is about a company whose 10-K you have not loaded yet, call fetch_filing(ticker) first, then use
   search_filings to read the relevant parts. Use get_market_data for current numbers, calculator for any computation,
   and research_web only for recent news or gaps.
2. Ground every factual or numeric claim in a tool result — a downstream verifier will check your answer against the
   exact evidence the tools returned, so do not state figures the tools did not give you.
3. When you have enough information, write your final answer as a normal message with no tool call.

This is a conversation: earlier turns stay in context, so handle follow-ups naturally and reuse what you already
gathered. Be accurate and concise, note that this is educational and not financial advice when you give any
investment-related conclusion, and end each turn by briefly asking whether the user wants more detail or has a
follow-up, then stop and wait for their reply."""


class AnalystAgent(BaseAgent):
    """Top-level orchestrator: decides among filings search, market data, calculation, and web
    research to answer investing questions. Ends when it stops calling tools and answers in text
    (so `final_tool_names` is empty)."""

    def __init__(self, vector_db: FAISS | None = None) -> None:
        self.vector_db = vector_db
        super().__init__(
            llm=orchestrator_llm,
            tools=self._build_tools(),
            system_prompt=_ANALYST_SYSTEM_PROMPT,
            final_tool_names=[],
            max_iter=10,
        )

    def _process_output(self) -> str:
        for message in reversed(self.messages):
            if isinstance(message, AIMessage) and str(message.content).strip():
                return str(message.content)
        return "I could not find enough information to answer that question."

    def _build_tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                func=self._fetch_filing,
                name="fetch_filing",
                description="Download a company's latest 10-K from SEC EDGAR by ticker and index it so search_filings can read it.",
                args_schema=_FetchFilingInput,
            ),
            StructuredTool.from_function(
                func=self._search_filings,
                name="search_filings",
                description="Search the loaded filings and return the most relevant passages. Call fetch_filing first if the company isn't loaded.",
                args_schema=_SearchFilingsInput,
            ),
            StructuredTool.from_function(
                func=_get_market_data,
                name="get_market_data",
                description="Live price and headline fundamentals (P/E, market cap, 52-week range, ...) for a ticker.",
                args_schema=_MarketDataInput,
            ),
            StructuredTool.from_function(
                func=_calculator,
                name="calculator",
                description="Evaluate a pure arithmetic expression. Use for every ratio, percentage, or growth math.",
                args_schema=_CalculatorInput,
            ),
            StructuredTool.from_function(
                func=self._research_web,
                name="research_web",
                description="Delegate to a web-research agent for recent news or anything the filings/market data lack.",
                args_schema=_ResearchWebInput,
            ),
        ]

    def _search_filings(self, query: str) -> str:
        if self.vector_db is None:
            return "No filings are loaded yet. Call fetch_filing(ticker) to download a company's 10-K first."
        docs = self.vector_db.similarity_search(query, k=5)
        if not docs:
            return "No relevant passages found in the loaded filings."
        return "\n\n".join(doc.page_content for doc in docs)[:MAX_FILINGS_CHARS]

    def _fetch_filing(self, ticker: str) -> str:
        try:
            chunks, filed = fetch_10k_chunks(ticker)
        except ValueError as e:
            return str(e)
        except Exception as e:
            logger.warning("EDGAR fetch failed for %s: %s", ticker, e)
            return f"Could not fetch the 10-K for '{ticker}': {e}"
        if self.vector_db is None:
            self.vector_db = FAISS.from_documents(chunks, global_embeddings)
        else:
            self.vector_db.add_documents(chunks)
        return (
            f"Loaded {ticker.upper()} 10-K (filed {filed}) — {len(chunks)} sections indexed. "
            "Now call search_filings to read specific parts (risk factors, MD&A, segments, ...)."
        )

    def _research_web(self, query: str) -> str:
        return WebResearchAgent().run(f"Research and answer this question: {query}")
