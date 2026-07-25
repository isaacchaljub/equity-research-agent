"""analyst.py — an agentic equity-research assistant (RAG over filings + live data + web + a verifier).

A top-level orchestrator agent (`AnalystAgent`) answers investing questions by deciding, at run
time, which of four tools to use:
- search_filings   : semantic search over local filing/annual-report PDFs (FAISS).
- get_market_data  : live quote & fundamentals for a ticker (yfinance).
- calculator       : safe arithmetic for ratios and back-of-envelope math.
- research_web     : delegate to a web-research sub-agent for recent news/context.

After the analyst drafts an answer, a second agent (`VerifierAgent`) fact-checks each claim
against the evidence the tools returned and flags anything unsupported. So the pipeline is
genuinely multi-agent: orchestrator -> (nested web agent) -> verifier.

Shared engine (`ResilientChat`, `BaseAgent`) is domain-agnostic: a plain `while` loop over a
tool-bound model, with retry/backoff/fallback and a token-bounded sliding window. No graph.

Model strategy (matters on free-tier keys):
- The web sub-agent tries gemini-3-flash-preview, waits 20s then 60s on failure, then falls back
  to Groq's gpt-oss-120b. The orchestrator and verifier run on gpt-oss-120b (off Gemini, so they
  don't share Gemini's request quota). Tool outputs are truncated to stay under token limits.

This is an educational tool, not financial advice.

Run: `python pipeline/analyst.py` (interactive) or import `process_query`.
"""

import ast
import logging
import operator
import os
import time
from abc import ABC
from abc import abstractmethod

import numpy as np
import yfinance as yf
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolCall
from langchain_core.messages import ToolMessage
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel
from pydantic import Field

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
global_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

web_llm = ChatLiteLLM(
    model="gemini/gemini-3-flash-preview",
    temperature=0.3,
    max_tokens=2000,
    timeout=None,
    max_retries=0,
    api_key=GEMINI_API_KEY,
)

web_backup_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    max_tokens=2000,
    timeout=None,
    max_retries=0,
    api_key=GROQ_API_KEY,
)

orchestrator_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1500,
    timeout=None,
    max_retries=0,
    api_key=GROQ_API_KEY,
)


class ResilientChat:
    """A primary chat model with a retry schedule and an optional backup it falls back to.

    On each failed primary `invoke`, waits the next interval in `retry_waits` and retries the
    primary; rate limits clear only after a wait, so an immediate retry is useless — the waits
    are the point. Once the schedule is spent, it switches to the backup and STAYS there for the
    rest of this instance's life (sticky), so a primary that is down for the whole run does not
    re-incur the full wait schedule every turn. The backup gets its own immediate retries to ride
    out stochastic failures (e.g. a model hallucinating a tool name the API rejects). Retry and
    fallback live here, so the agent loop that calls invoke() stays a plain while.
    """

    def __init__(
        self,
        primary: Runnable,
        backup: Runnable | None = None,
        retry_waits: tuple[int, ...] = (5, 15, 30),
        backup_retries: int = 2,
    ):
        self.primary = primary
        self.backup = backup
        self.retry_waits = retry_waits
        self.backup_retries = backup_retries
        self._use_backup = False

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last_error: Exception | None = None
        if not self._use_backup:
            for i in range(len(self.retry_waits) + 1):
                try:
                    return self.primary.invoke(messages)
                except Exception as e:
                    last_error = e
                    if i < len(self.retry_waits):
                        wait = self.retry_waits[i]
                        logger.warning(
                            "Primary model failed (%s); retry %d/%d in %ds",
                            e.__class__.__name__, i + 1, len(self.retry_waits), wait,
                        )
                        time.sleep(wait)
            if self.backup is None:
                raise last_error
            logger.warning("Primary model exhausted; switching to backup for the rest of this run")
            self._use_backup = True
        for _ in range(self.backup_retries + 1):
            try:
                return self.backup.invoke(messages)
            except Exception as e:
                last_error = e
        raise last_error


class BaseAgent(ABC):
    """A ReAct agent is a loop, not a graph.

    Invoke the model, run whatever tools it asked for, append the results, repeat — until the
    model calls a terminal tool (one named in `final_tool_names`) or the iteration budget runs
    out. State lives on `self`, so there is no state dict to thread between nodes and no edges to
    wire: the control flow is the `while` in `run`. Resilience lives in `self.llm` (a
    ResilientChat) and context is bounded by `_windowed_messages`, keeping the loop clean.
    """

    def __init__(
        self,
        llm: Runnable,
        tools: list[BaseTool],
        system_prompt: str,
        final_tool_names: list[str],
        max_iter: int = 6,
        backup_llm: Runnable | None = None,
        retry_waits: tuple[int, ...] = (5, 15, 30),
        max_history_tokens: int = 4000,
    ) -> None:
        primary = llm.bind_tools(tools)
        backup = backup_llm.bind_tools(tools) if backup_llm is not None else None
        self.llm = ResilientChat(primary, backup, retry_waits)
        self.tools = {tool.name: tool for tool in tools}
        self.final_tool_names = final_tool_names
        self.max_iter = max_iter
        self.max_history_tokens = max_history_tokens
        self.messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._final_payload: str | None = None

    @traceable(run_type="chain")
    def run(self, user_message: str) -> str:
        self.messages.append(HumanMessage(content=user_message))
        n_iter = 0
        done = False
        while not done and n_iter < self.max_iter:
            resp = self.llm.invoke(self._windowed_messages())
            self.messages.append(resp)
            done = self._run_tools(resp.tool_calls) if resp.tool_calls else True
            n_iter += 1
        return self._process_output()

    def _windowed_messages(self) -> list[BaseMessage]:
        """Sliding-window view of the history actually sent to the model: the system prompt plus
        the most recent messages that fit in `max_history_tokens`, dropping older turns.
        `self.messages` keeps the full record — only what is *sent* is bounded, so token cost and
        context stay flat as a conversation grows. `start_on='human'` keeps whole turns, so an
        AIMessage's tool calls are never split from their ToolMessage results (which a provider
        rejects as an orphaned pair).

        Guard: if even the latest turn is larger than the budget, trim would drop it and leave
        only the system prompt (an invalid request). In that case send the system prompt plus the
        whole current turn, uncut."""
        windowed = trim_messages(
            self.messages,
            max_tokens=self.max_history_tokens,
            strategy="last",
            token_counter=count_tokens_approximately,
            include_system=True,
            start_on="human",
            allow_partial=False,
        )
        if any(isinstance(message, HumanMessage) for message in windowed):
            return windowed
        last_human = max(i for i, message in enumerate(self.messages) if isinstance(message, HumanMessage))
        return [self.messages[0], *self.messages[last_human:]]

    def _run_tools(self, tool_calls: list[ToolCall]) -> bool:
        """Execute each requested tool, appending its ToolMessage. Return True once a terminal
        tool has fired so the loop can stop."""
        reached_final = False
        for tool_call in tool_calls:
            selected_tool = self.tools.get(tool_call["name"])
            if selected_tool is None:
                self.messages.append(
                    ToolMessage(
                        content=f'Unknown tool {tool_call["name"]}; pick one of {list(self.tools)}.',
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
                continue
            try:
                tool_msg = selected_tool.invoke(tool_call)
            except Exception as e:
                tool_msg = ToolMessage(
                    content=f'Error calling {tool_call["name"]}: {e}',
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    status="error",
                )
            self.messages.append(tool_msg)
            if tool_call["name"] in self.final_tool_names:
                self._final_payload = str(tool_msg.content)
                reached_final = True
        return reached_final

    @abstractmethod
    def _process_output(self) -> str:
        raise NotImplementedError


_MAX_SEARCH_CHARS = 2000
_MAX_SCRAPE_CHARS = 3000
_MAX_FILINGS_CHARS = 4000
_MAX_EVIDENCE_CHARS = 6000


class _WebSearchInput(BaseModel):
    search_query: str = Field(description="What to search the web for.")


class _ScrapeInput(BaseModel):
    website_url: str = Field(description="URL of a promising page to fetch and read.")


class _SubmitFindingsInput(BaseModel):
    summary: str = Field(description="A thorough, self-contained summary of what you found that answers the question.")


_serper = SerperDevTool()
_scraper = ScrapeWebsiteTool()


def _web_search(search_query: str) -> str:
    return str(_serper.run(search_query=search_query))[:_MAX_SEARCH_CHARS]


def _web_scrape(website_url: str) -> str:
    return str(_scraper.run(website_url=website_url))[:_MAX_SCRAPE_CHARS]


def _submit_findings(summary: str) -> str:
    return summary


def build_web_tools() -> list[BaseTool]:
    """The web sub-agent's tools: search, scrape, and a terminal submit_findings. Search/scrape
    outputs are truncated so accumulated results stay under free-tier token-per-minute ceilings."""
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


query_cache: list[tuple[list[float], str, str]] = []
max_cache_size = 20
similarity_threshold = 0.85


def cosine_similarity(query_embedding, cached_embedding):
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))


def check_cache(query: str) -> str | None:
    """Return a cached answer for a sufficiently similar past query, else None."""
    if not query_cache:
        return None
    embedded_query = global_embeddings.embed_query(query)
    scores = [cosine_similarity(embedded_query, item[0]) for item in query_cache]
    if np.max(scores) > similarity_threshold:
        return query_cache[int(np.argmax(scores))][2]
    return None


def update_cache(query: str, answer: str) -> None:
    query_cache.append((global_embeddings.embed_query(query), query, answer))
    if len(query_cache) > max_cache_size:
        query_cache.pop(0)


def load_all_documents(documents_directory="documents"):
    """Load every PDF and text filing in the directory into LangChain Documents."""
    all_documents = []
    for file in os.listdir(documents_directory):
        path = os.path.join(documents_directory, file)
        if file.endswith(".pdf"):
            all_documents.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            all_documents.extend(TextLoader(path).load())
    return all_documents


def create_vector_database(documents):
    """Chunk the documents and index them in FAISS for similarity search."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, global_embeddings)


def initialize_vectorstore(documents_directory="documents"):
    """Load all filings and build a single FAISS store."""
    all_documents = load_all_documents(documents_directory)
    if not all_documents:
        raise ValueError(
            f"No .pdf or .txt filings found in '{documents_directory}'. Add 10-K / annual-report files there."
        )
    return create_vector_database(all_documents)


class _SearchFilingsInput(BaseModel):
    query: str = Field(description="A focused search query for the local filing knowledge base.")


class _MarketDataInput(BaseModel):
    ticker: str = Field(description="A stock ticker symbol, e.g. AAPL, MSFT, NVDA.")


class _CalculatorInput(BaseModel):
    expression: str = Field(description="A pure arithmetic expression, e.g. '(120.5 - 98) / 98 * 100'.")


class _ResearchWebInput(BaseModel):
    query: str = Field(description="The news/context question to research on the web.")


_ANALYST_SYSTEM_PROMPT = """You are an equity-research analyst assistant having a conversation with the user.

You have four tools:
- search_filings: semantic search over the local filing/annual-report knowledge base (fundamentals, risks, strategy).
- get_market_data: live price and headline fundamentals (P/E, market cap, 52-week range, ...) for a ticker.
- calculator: evaluate an arithmetic expression (use it for any ratio or growth-rate math — do not do arithmetic in your head).
- research_web: delegate to a web-research agent for recent news or anything the filings and market data do not cover.

How to work:
1. Decide which tools the question needs. Use search_filings for qualitative/fundamental questions, get_market_data
   for current numbers, calculator for any computation, and research_web only for recent news or gaps.
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

    def __init__(self, vector_db: FAISS) -> None:
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
                func=self._search_filings,
                name="search_filings",
                description="Search the local filing knowledge base and return the most relevant passages.",
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
        docs = self.vector_db.similarity_search(query, k=5)
        if not docs:
            return "No relevant passages found in the local filings."
        return "\n\n".join(doc.page_content for doc in docs)[:_MAX_FILINGS_CHARS]

    def _research_web(self, query: str) -> str:
        return WebResearchAgent().run(f"Research and answer this question: {query}")


def _collect_evidence(messages: list[BaseMessage]) -> str:
    """Concatenate the tool outputs (the ground truth the answer must be built from) for the
    verifier, capped so the verifier request stays within token limits."""
    parts = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(parts)[:_MAX_EVIDENCE_CHARS]


def verify_answer(question: str, answer: str, analyst_messages: list[BaseMessage]) -> str:
    """Run the verifier agent over the analyst's answer and the evidence its tools returned."""
    evidence = _collect_evidence(analyst_messages)
    if not evidence:
        return "Verification: skipped (the answer used no tool evidence)."
    verifier_input = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT ANSWER:\n{answer}\n\n"
        f"EVIDENCE (tool outputs the answer must be grounded in):\n{evidence}"
    )
    return VerifierAgent().run(verifier_input)


def process_query(query: str, vector_db: FAISS | None = None) -> str:
    """Answer an investing question, then fact-check the answer against the sources.

    The cache is a deterministic fast-path around the agents; everything else — which tools to
    use, whether to escalate to the web, and the final wording — is the model's call."""
    logger.info("Processing query: %s", query)

    if (cached_answer := check_cache(query)) is not None:
        logger.info("Cache hit")
        return cached_answer

    if vector_db is None:
        vector_db = initialize_vectorstore()

    analyst = AnalystAgent(vector_db)
    answer = analyst.run(query)
    verdict = verify_answer(query, answer, analyst.messages)
    final = f"{answer}\n\n---\n{verdict}"
    update_cache(query, final)
    return final


def chat(documents_directory: str = "documents") -> None:
    """Interactive, event-driven analyst. One AnalystAgent persists for the session, so history
    carries across turns; each answer is fact-checked before it is shown."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    vector_db = initialize_vectorstore(documents_directory)
    analyst = AnalystAgent(vector_db)
    print("Equity-research analyst ready. Ask about a company or filing, or type 'exit'.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        answer = analyst.run(user_input)
        verdict = verify_answer(user_input, answer, analyst.messages)
        print(f"\nAnalyst: {answer}\n\n[{verdict}]")


def main():
    """Answer a single hardcoded question (non-interactive)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    vector_db = initialize_vectorstore("documents")
    query = "What are the main risk factors, and how does the current P/E compare to the 52-week range?"
    print(process_query(query, vector_db))


if __name__ == "__main__":
    chat()
