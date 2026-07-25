# `pipeline/analyst.py` — Complete Code Walkthrough

> A section-by-section explanation of the whole system, written so you can re-read it months
> later and immediately understand *what* every piece does and *why* it exists.

---

## 0. The one-paragraph mental model

This is an **equity-research assistant** over your filing PDFs that can also pull **live market
data**, do **math**, and go to the **web** — and then **fact-check its own answer**. What makes it
*agentic* (not a fixed pipeline) is that **a language model decides the control flow at run
time**: it chooses which tools to call, in what order, how many times, and when to answer. There
is **no graph** and **no hardcoded router** — just a `while` loop that hands the model tools and
does what it asks, until it stops asking.

There are **three agents**, all built on one engine (`BaseAgent`):

```
AnalystAgent (orchestrator)         ← decides which tools to use, writes the answer
  ├── tool: fetch_filing             → downloads a ticker's latest 10-K from SEC EDGAR + indexes it
  ├── tool: search_filings           → FAISS similarity search over loaded filings (not an agent)
  ├── tool: get_market_data          → live quote & fundamentals via yfinance (not an agent)
  ├── tool: calculator               → safe arithmetic (not an agent)
  └── tool: research_web             → runs WebResearchAgent (an agent, used as a tool)
        WebResearchAgent
          ├── web_search / web_scrape → Google (Serper) + page scraping
          └── submit_findings         → TERMINAL tool: ends the web loop with a summary

VerifierAgent                        ← runs AFTER the analyst; checks the answer vs the evidence
  └── submit_verdict                  → TERMINAL tool: returns the fact-check
```

---

## 1. Two ideas to hold in your head

### 1a. "Agentic" means the model owns the control flow

- **Non-agentic (a pipeline):** *you* wrote the order of steps. The model is called like a
  function — it returns text, and your code decides what happens next.
- **Agentic (a loop):** the model's output *is a decision that steers execution* — a tool call the
  runtime dispatches, whose result feeds back so the model decides again, looping an unknown
  number of times until it decides to stop.

`AnalystAgent` is agentic. `search_filings`, `get_market_data`, and `calculator` are *not* agentic
— they're plain functions the agent chooses to call.

### 1b. Two ways a loop can end (this trips everyone up)

All three agents run the *same* loop but **terminate differently**:

| Agent | `final_tool_names` | How its loop ends |
|-------|--------------------|-------------------|
| `WebResearchAgent` | `["submit_findings"]` | the model calls the **terminal tool** `submit_findings`; its argument becomes the return value |
| `VerifierAgent` | `["submit_verdict"]` | the model calls the **terminal tool** `submit_verdict` |
| `AnalystAgent` | `[]` (empty) | the model **stops calling tools and writes plain text**; that text is the answer |

**Sub-agents end with a terminal tool call; the orchestrator ends by answering in text.** Why?
A sub-agent hands a *structured string back to code* (a summary, a verdict), so we force it through
a named tool whose argument we capture. The orchestrator answers a *human*, so the natural end is
"it says the answer" — in the loop, "no tool calls this turn" is the terminal condition.

That's why the analyst prompt says *"write your final answer as a normal message with no tool
call"* while the sub-agent prompts say *"Always finish by calling submit_…"*.

---

## 2. Section-by-section walkthrough

### 2.1 Imports & config (top of file)

Standard library: `ast`/`operator` (the safe calculator), `logging`, `os`, `time` (retry backoff),
`abc` (abstract base class). Third-party: `numpy` (cosine similarity for the cache), `yfinance`
(market data), `crewai_tools` (Serper search + scraping), `dotenv`, `langchain_*` (models,
messages, tools, splitters, FAISS, embeddings), `langsmith` (`traceable`), `pydantic` (tool arg
schemas).

```python
load_dotenv()
logger = logging.getLogger(__name__)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # + GEMINI_API_KEY, SERPER_API_KEY
global_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
```
Load `.env`, get a module logger, read keys, and build one shared embedding model (used for both
FAISS indexing and the semantic cache).

**Three models, chosen deliberately:**
```python
web_llm          = ChatLiteLLM(model="gemini/gemini-3-flash-preview", ..., max_retries=0)  # web primary
web_backup_llm   = ChatGroq(model="openai/gpt-oss-120b", ..., max_retries=0)               # web fallback
orchestrator_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, ..., max_retries=0) # analyst + verifier
```
`gpt-oss-120b` (not `llama-3.3`) because llama on Groq emits malformed tool calls the API rejects;
gpt-oss tool-calls cleanly. The orchestrator/verifier run on Groq, **off Gemini**, so they don't
compete for Gemini's tight free-tier request quota. `max_retries=0` on all three makes
`ResilientChat` the single retry authority (no hidden double-retrying underneath).

### 2.2 `ResilientChat` — the resilience layer

Wraps a model so the agent loop never has to think about failures.

```python
def __init__(self, primary, backup=None, retry_waits=(5, 15, 30), backup_retries=2):
    ...
    self._use_backup = False
```
- `retry_waits` — the wait schedule **between primary attempts**. `(20, 60)` for the web agent =
  try, wait 20s, try, wait 60s, try, then give up on the primary.
- `_use_backup` — the **sticky flag**: once we fall back, we stay on the backup.

```python
def invoke(self, messages):
    if not self._use_backup:
        for i in range(len(self.retry_waits) + 1):
            try:
                return self.primary.invoke(messages)
            except Exception as e:
                last_error = e
                if i < len(self.retry_waits):
                    time.sleep(self.retry_waits[i])     # rate limits clear only after real time
        if self.backup is None:
            raise last_error
        self._use_backup = True                          # sticky: don't re-wait the primary next turn
    for _ in range(self.backup_retries + 1):
        try:
            return self.backup.invoke(messages)
        except Exception as e:
            last_error = e
    raise last_error
```
`time.sleep` matters because a rate limit only clears after time passes — an immediate retry is
useless. **Sticky fallback** matters because if Gemini is down for the whole run and we re-tried it
every turn, every turn would pay the full `20+60 = 80s` before falling back; sticky pays it once.
The backup gets its own quick retries to shrug off stochastic failures (e.g. a hallucinated tool
name). Because all of this lives in `invoke`, the agent loop stays a plain `while`.

### 2.3 `BaseAgent` — the agent engine

```python
class BaseAgent(ABC):
```
`ABC` = **Abstract Base Class**. You can't instantiate `BaseAgent` directly; combined with the
`@abstractmethod` on `_process_output`, Python raises `TypeError` if a subclass forgets to
implement it. It turns "please implement this" into an enforced contract.

```python
def __init__(self, llm, tools, system_prompt, final_tool_names,
             max_iter=6, backup_llm=None, retry_waits=(5, 15, 30), max_history_tokens=4000):
    primary = llm.bind_tools(tools)
    backup = backup_llm.bind_tools(tools) if backup_llm is not None else None
    self.llm = ResilientChat(primary, backup, retry_waits)
    self.tools = {tool.name: tool for tool in tools}
    ...
    self.messages = [SystemMessage(content=system_prompt)]
```
`llm.bind_tools(tools)` is the key line. `llm` is a LangChain chat model (`ChatGroq`/`ChatLiteLLM`,
both `BaseChatModel` where `bind_tools` is defined). It **does not call the model** — it returns a
new `Runnable` with the tools' JSON schemas attached, so a later `.invoke(messages)` lets the model
emit tool calls. We wrap the tool-bound primary (and backup) in a `ResilientChat`. From here on,
`self.llm.invoke(...)` means **`ResilientChat.invoke`** — our wrapper, not a raw model call.

```python
@traceable(run_type="chain")
def run(self, user_message):
    self.messages.append(HumanMessage(content=user_message))
    n_iter, done = 0, False
    while not done and n_iter < self.max_iter:
        resp = self.llm.invoke(self._windowed_messages())
        self.messages.append(resp)
        done = self._run_tools(resp.tool_calls) if resp.tool_calls else True
        n_iter += 1
    return self._process_output()
```
**This is the whole agent.** Append the user message, then loop: call the model (via
`ResilientChat`, on a **windowed** view of history — §2.4), append the response, and branch — if it
made tool calls, run them (`_run_tools` returns `True` only if a *terminal* tool fired); if not, the
model answered → `done`. `max_iter` caps a runaway. `@traceable` makes each agent run a named span
in LangSmith (nesting model + tool calls under it). Because `run` *appends* to `self.messages`,
calling `run` again on the **same instance** continues the conversation — that's how the CLI, the
`/chat` endpoint, and the Streamlit UI get multi-turn memory.

```python
def _run_tools(self, tool_calls):
    reached_final = False
    for tool_call in tool_calls:
        selected_tool = self.tools.get(tool_call["name"])
        if selected_tool is None:                      # model hallucinated a tool name
            self.messages.append(ToolMessage(content=f'Unknown tool {tool_call["name"]}; ...', ...))
            continue
        try:
            tool_msg = selected_tool.invoke(tool_call)  # <-- LangChain BaseTool.invoke, NOT ResilientChat
        except Exception as e:
            tool_msg = ToolMessage(content=f'Error calling ...: {e}', ..., status="error")
        self.messages.append(tool_msg)
        if tool_call["name"] in self.final_tool_names:
            self._final_payload = str(tool_msg.content)
            reached_final = True
    return reached_final
```

> **The two `.invoke`s — the single most confusing part:**
> - `self.llm.invoke(messages)` → **`ResilientChat.invoke`** → an `AIMessage` (a *model* call).
> - `selected_tool.invoke(tool_call)` → **`BaseTool.invoke`** (LangChain) → a `ToolMessage` (runs
>   the tool function).
>
> They share the name only because both implement LangChain's `Runnable` interface. They connect
> *transitively*: when the tool is `research_web`, its function spins up a `WebResearchAgent`, whose
> own loop calls *its* `ResilientChat.invoke`. So `tool.invoke` reaches a `ResilientChat.invoke`
> only when the tool is itself an agent. For `search_filings` it's just a FAISS lookup — no model.

Note the two safety nets: an **unknown tool** becomes an error `ToolMessage` (the model sees valid
options and can retry) rather than a crash, and a **raising tool** becomes an error `ToolMessage`
too. Each `tool_call_id` is echoed back, because every tool call needs exactly one matching result
or the provider rejects the next request.

### 2.4 `_windowed_messages` — the sliding window (context management)

Without this, `self.messages` grows forever and the whole history is re-sent every call — cost and
latency climb, and you eventually blow past the context window or a per-minute token limit. This
bounds what is *sent* while keeping the full record on `self`.

```python
def _windowed_messages(self):
    windowed = trim_messages(
        self.messages, max_tokens=self.max_history_tokens, strategy="last",
        token_counter=count_tokens_approximately, include_system=True,
        start_on="human", allow_partial=False,
    )
    if any(isinstance(m, HumanMessage) for m in windowed):
        return windowed
    last_human = max(i for i, m in enumerate(self.messages) if isinstance(m, HumanMessage))
    return [self.messages[0], *self.messages[last_human:]]
```
A **sliding window** is one trimming strategy: keep the system prompt + the most recent messages,
drop the oldest. The `trim_messages` params: `strategy="last"` (keep most recent), `include_system`
(never drop the system prompt), **`start_on="human"`** (the window must begin on a human message, so
a tool call is never split from its result — providers reject an orphaned pair), `allow_partial=
False` (whole messages only). **The guard:** if even the latest turn exceeds the budget,
`trim_messages` would return just the system message (an invalid request), so we fall back to
"system + the whole current turn." Because `start_on="human"` keeps whole turns, trimming only drops
**older, completed turns** — so it does nothing within a single sub-agent run and does its real work
across the many turns of a conversation.

### 2.5 Web tools + `WebResearchAgent`

Truncation caps (`_MAX_SEARCH_CHARS`, `_MAX_SCRAPE_CHARS`) keep tool outputs small — a full scraped
page can blow past a model's per-minute token limit. `_web_search`/`_web_scrape` wrap the CrewAI
tools and truncate; `_submit_findings` is the **terminal tool** whose returned `summary` becomes the
agent's output. `WebResearchAgent` configures the engine with the web tools, `final_tool_names=
["submit_findings"]`, Gemini primary + Groq backup, and the **`(20, 60)`** wait schedule. Its
`_process_output` returns the `submit_findings` payload (or the last text as a fallback). The tool is
named `web_scrape` (not `scrape_website`) so the model's `web_search`→`web_scrape` instinct is
correct, reducing tool-name hallucinations.

### 2.6 The calculator (safe arithmetic)

```python
def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
    if isinstance(node, ast.BinOp)  and type(node.op) in _ARITHMETIC_OPS: return _ARITHMETIC_OPS[...](...)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_OPS: return _ARITHMETIC_OPS[...](...)
    raise ValueError("only numbers and + - * / // % ** are allowed")
```
Parses the expression to an AST and walks it, allowing **only** numbers and arithmetic operators —
no names, calls, or attributes. So there's **no code-execution surface** (unlike `eval`, which would
run `__import__('os').system(...)`). `_calculator` wraps it and returns `"<expr> = <result>"`, or a
friendly error. The analyst prompt tells the model to use this for *all* math rather than doing
arithmetic in its head (LLMs are unreliable at arithmetic).

### 2.7 The market-data tool

```python
def _get_market_data(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return f"Could not fetch market data for '{ticker}': {e}"
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if not info or price is None:
        return f"No market data found for ticker '{ticker}'. ..."
    fields = {"Name": ..., "Price": price, "Trailing P/E": info.get("trailingPE"), ...}
    return "Market data for AAPL:\n- Name: ...\n- Price: ...\n..."
```
Fetches a live quote and headline fundamentals via `yfinance`, formats the fields that exist, and
degrades gracefully on a bad ticker or a network error (returning a string the model can react to,
rather than raising).

### 2.8 `VerifierAgent` — the fact-checker

```python
def _submit_verdict(verdict, unsupported_claims=None):
    lines = [f"Verification: {verdict}"]
    if unsupported_claims:
        lines.append("Flagged (not supported by the sources):")
        lines.extend(f"  - {c}" for c in unsupported_claims)
    return "\n".join(lines)

class VerifierAgent(BaseAgent):
    def __init__(self):
        super().__init__(llm=orchestrator_llm, tools=[submit_verdict_tool],
                         system_prompt=_VERIFIER_SYSTEM_PROMPT, final_tool_names=["submit_verdict"], max_iter=3)
```
A second agent that **calls no research tools** — it only judges. It's given the question, the draft
answer, and the **evidence** (the raw tool outputs), and is prompted to check every claim against the
evidence *without using outside knowledge*, then call `submit_verdict` with an overall verdict and a
list of unsupported claims. Making it a `BaseAgent` with a terminal tool keeps it consistent with the
rest of the architecture and gives structured output.

### 2.9 Cache & vector store

The **semantic cache** (`check_cache`/`update_cache`) stores `(embedding, query, answer)` triples and
returns a cached answer when a new query's embedding is >0.85 cosine-similar to a past one — an
*answer-level* shortcut, separate from context management. The **vector store**
(`load_all_documents` → `create_vector_database` → `initialize_vectorstore`) loads every `.pdf` and
`.txt` filing, chunks them (~1000 chars, 200 overlap), embeds, and indexes in FAISS.
`initialize_vectorstore` returns **None** when the folder is empty (instead of raising), so the app
can start with no local files and pull filings on demand.

**EDGAR fetch** (`fetch_filing` → `fetch_10k_chunks` → `_resolve_cik`/`_latest_10k`/
`_download_filing_text`): resolves a ticker to its SEC CIK (cached), finds the latest 10-K in the
submissions feed, downloads the primary document, and strips hidden/inline-XBRL noise before
extracting prose. The `_fetch_filing` tool then indexes those chunks — creating the FAISS store if it
was `None`, else adding to it — so `search_filings` can read them. This is what lets you research a
company from just a ticker; no PDF hunting.

### 2.10 `AnalystAgent` — the orchestrator

Configures the engine with **five tools** (`fetch_filing`, `search_filings`, `get_market_data`,
`calculator`, `research_web`), `final_tool_names=[]` (ends by answering in text), and `max_iter=10`
(room for fetch + multiple tools + web). `self.vector_db` (which may start `None`) is set **before**
`super().__init__` because `_build_tools` (called inside it) references it. `_search_filings`,
`_fetch_filing`, and `_research_web` are **bound methods** — which is exactly why we build tools with
`StructuredTool.from_function` at instance-construction time rather than the `@tool` decorator (see
§3). `_search_filings` returns a "call fetch_filing first" hint when nothing is loaded yet.
`_process_output` returns the most recent AI message with
non-empty text (so a `max_iter` stall mid-tool-call doesn't return `""`). Its prompt makes it decide
which tools to use, **ground every claim in a tool result** (so the verifier can check it), add a
"not financial advice" note, and end each turn by asking if you want more.

### 2.11 The verify flow & entry points

```python
def verify_answer(question, answer, analyst_messages):
    evidence = _collect_evidence(analyst_messages)     # concatenated ToolMessage contents
    if not evidence: return "Verification: skipped (the answer used no tool evidence)."
    return VerifierAgent().run(f"QUESTION:\n{question}\n\nDRAFT ANSWER:\n{answer}\n\nEVIDENCE:\n{evidence}")

def process_query(query, vector_db=None):
    if (cached := check_cache(query)) is not None: return cached
    if vector_db is None: vector_db = initialize_vectorstore()
    analyst = AnalystAgent(vector_db)
    answer = analyst.run(query)
    verdict = verify_answer(query, answer, analyst.messages)
    final = f"{answer}\n\n---\n{verdict}"
    update_cache(query, final)
    return final
```
`_collect_evidence` pulls the analyst's `ToolMessage` contents — the ground truth the answer must be
built from — and `verify_answer` runs the verifier over (question, answer, evidence). `process_query`
is the stateless one-shot (cache → analyst → verify → cache). `chat()` is the interactive multi-turn
REPL (one persistent `AnalystAgent`, verify each turn). `main()` is a hardcoded one-shot for quick
testing. Running the file launches `chat()`.

---

## 3. Why `StructuredTool.from_function` instead of the `@tool` decorator?

Both produce a `StructuredTool` (a `BaseTool` subclass); `@tool` is just a decorator shortcut. We use
`from_function` because:
1. **The orchestrator's tools are instance methods.** `@tool` runs at *class-definition* time, when
   there's no `self`. `StructuredTool.from_function(func=self._search_filings, ...)` runs inside
   `_build_tools()` at *instance-construction* time, capturing the **bound method** — so the tool can
   reach `self.vector_db` and spawn sub-agents.
2. **Explicit control** over `name`/`description`/`args_schema`, independent of the Python function
   name (e.g. registering `_web_scrape` as `web_scrape`).

`BaseTool` is never instantiated — it's only the **type hint** (`list[BaseTool]`), the common base of
every tool.

---

## 4. End-to-end trace (a question needing filings + market data, then verified)

```
process_query("How does AAPL's P/E compare to what the filing implies?")
  ├─ check_cache → miss
  ├─ AnalystAgent.run(...)
  │    turn 1: model calls search_filings("valuation / earnings")  → 5 FAISS chunks
  │    turn 2: model calls get_market_data("AAPL")                 → live price, trailing P/E, ...
  │    turn 3: model calls calculator("<price>/<eps>")             → a number
  │    turn 4: model writes the answer in plain text (no tool call) → done
  ├─ verify_answer(question, answer, analyst.messages)
  │    _collect_evidence → the filing chunks + market data + calc results
  │    VerifierAgent.run(...) → submit_verdict("supported", []) (or flags unsupported claims)
  └─ return  answer + "\n---\n" + verdict   (and cache it)
```

If the filings + market data don't cover the question, turn *N* instead calls `research_web(...)`,
which runs the nested `WebResearchAgent` (Gemini → 20s → 60s → Groq fallback; web_search → web_scrape
→ submit_findings), and its summary becomes evidence like any other tool result.

---

## 5. LangSmith, logging, and tests

- **LangSmith**: `@traceable` on `BaseAgent.run` makes each agent run a named span; LangChain
  auto-traces every model and tool `.invoke` under it. Set `LANGSMITH_TRACING=true` +
  `LANGSMITH_API_KEY` in `.env` and traces nest as *analyst → research_web → web agent → tools*, plus
  a sibling *verifier* span. With no key set, `@traceable` is a cheap passthrough.
- **Logging**: the library uses `logging.getLogger(__name__)`; retries/fallbacks/cache-hits are log
  lines (not prints). Entry points call `logging.basicConfig(...)`.
- **Tests** (`uv run pytest`, all mocked/offline): `ResilientChat` (retry schedule, sticky fallback,
  backup retries, raise-on-exhaust), the sliding window (bounds + guard + no orphaned tool pairs), the
  agent loop (`test_base_agent`: stops on text, runs tools, terminal tool ends it, unknown-tool
  handling, `max_iter`), the tools (calculator correctness + safety, market data formatting/errors),
  the cache, a mocked end-to-end analyst→verifier flow, and the FastAPI endpoints.

---

## 6. Known limitations (so future-you isn't surprised)

- **Sliding window, not summarization.** The window bounds cost but *forgets* old turns once history
  exceeds `max_history_tokens`. For long sessions where old context matters, summarization/compaction
  is the next step. `self.messages` also still holds the full history in memory.
- **Answer accuracy is not guaranteed.** The verifier catches claims unsupported *by the gathered
  evidence*, but it can't catch a wrong-but-plausible source, and market data can be stale/rate-limited.
- **Free-tier ceilings shape behavior.** Gemini's ~20 req/min and Groq's ~8000 tokens/min drove the
  fallback design and truncation caps; on paid tiers you can raise `max_iter`, loosen truncation, and
  shorten `retry_waits`.
- **Session state is in-memory, single-process.** The FastAPI `/chat` conversation store is a dict;
  a multi-worker deployment needs a shared store (e.g. Redis) keyed by conversation id.
