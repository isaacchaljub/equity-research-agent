# Code Walkthrough

A tour of how the agent works and how the package is laid out. (An earlier, single-file version of
this codebase — before the package refactor — is preserved in the project's git history.)

## The one-paragraph model

An equity-research assistant over company filings that also pulls **live market data**, does **math**,
and searches the **web** — then **fact-checks its own answer**. It's *agentic* because a language model
decides the control flow at run time: which tools to call, in what order, and when to answer. No graph
— just a `while` loop that hands the model tools. Three agents: `AnalystAgent` (orchestrator) →
`research_web` → `WebResearchAgent` (a sub-agent used as a tool) → then `VerifierAgent` checks the answer.

## Two ideas worth knowing

**Agentic = the model owns the control flow.** A pipeline runs steps you wrote; an agent loops and the
model chooses each step. `search_filings` / `get_market_data` / `calculator` are plain functions the
agent *chooses* to call — not agentic themselves.

**Two ways a loop ends.** Sub-agents end by calling a **terminal tool** (`submit_findings` /
`submit_verdict`) whose argument is the return value. The orchestrator ends by **stopping tool calls
and writing text** (`final_tool_names=[]`) — it's answering a human, not handing a string back to code.

## Package layout

```
pipeline/
  config.py         keys, model clients, tunable constants          (imports nothing local)
  observability.py  structlog logging + Sentry setup                (imports nothing local)
  engine.py         ResilientChat + BaseAgent (reusable framework)  (imports nothing local)
  tools.py          web search/scrape, calculator, market data      -> config
  retrieval.py      cache, FAISS store, SEC EDGAR fetch             -> config
  agents.py         WebResearchAgent, VerifierAgent, AnalystAgent   -> config, engine, retrieval, tools
  service.py        verify + pipeline + CLI entrypoints             -> config, agents, observability, retrieval
```

The layering keeps the **framework** (`engine.py`) separate from the **app** (the rest), and the
import arrows only point down — no cycles.

## The engine (`engine.py`)

**`ResilientChat`** wraps a primary model + optional backup with a retry schedule: on failure it waits
(rate limits only clear after a wait) and retries; when the schedule is spent it **sticky-switches to
the backup**. So the agent loop that calls `.invoke()` never sees a failure.

**`BaseAgent`** is the ReAct loop: invoke the model → run whatever tools it asked for → append results
→ repeat, until a terminal tool fires or `max_iter`. `ABC` + `@abstractmethod _process_output` make it
a template subclasses must complete. State lives on `self.messages`; a **sliding window**
(`_windowed_messages`) bounds what's *sent* to the model without splitting a tool call from its result.

> **Two `.invoke`s:** `self.llm.invoke(messages)` is **ResilientChat.invoke** (a model call →
> `AIMessage`); `selected_tool.invoke(tool_call)` is **BaseTool.invoke** (runs a tool → `ToolMessage`).
> Same name, different objects — they connect only when a tool *is* an agent (`research_web`).

## The domain (`tools.py`, `retrieval.py`, `agents.py`)

- **tools** — Serper search, page scrape, a safe AST-based calculator (no `eval`), yfinance market
  data, and the terminal tools. Outputs truncated to fit token limits.
- **retrieval** — the semantic answer cache, the FAISS filing store (starts empty; fetch on demand),
  and SEC EDGAR fetching (ticker → CIK → latest 10-K → text with inline-XBRL stripped → chunks).
- **agents** — the three `BaseAgent` subclasses. `AnalystAgent`'s tools include **bound methods**
  (`fetch_filing`/`search_filings`/`research_web`), which is why they're built with
  `StructuredTool.from_function` at instance time rather than the `@tool` decorator.

## The app (`service.py`)

`process_query` (a `@traceable` LangSmith root) runs: cache → `AnalystAgent` → `verify_answer` → cache.
`verify_answer` is best-effort — a verifier rate-limit returns "verification unavailable" instead of
sinking a good answer. `chat`/`main` are the CLI entrypoints and call `setup_logging()` + `setup_sentry()`.

## Observability

- **structlog** — structured application logs (`observability.setup_logging`).
- **Sentry** — error tracking; a `logger.exception(...)` auto-reports (set `SENTRY_DSN`).
- **LangSmith** — LLM tracing. Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`; each request is one
  nested trace (`process_query → analyst run → model/tools → verifier run`) because `process_query`,
  `BaseAgent.run`, and `ResilientChat.invoke` are all `@traceable`.

## Deployment

Multi-stage Dockerfile (CPU-only, non-root, embedding model baked in) + one-command
deploy/teardown scripts for AWS ECS Fargate and Azure Container Apps (see `deploy/` and the README).
