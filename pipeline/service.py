"""The pipeline and entrypoints: answer + verify, the semantic-cache wrapper, and the CLI."""

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langsmith import traceable

from pipeline.agents import AnalystAgent
from pipeline.agents import VerifierAgent
from pipeline.config import MAX_EVIDENCE_CHARS
from pipeline.observability import setup_logging
from pipeline.observability import setup_sentry
from pipeline.retrieval import check_cache
from pipeline.retrieval import initialize_vectorstore
from pipeline.retrieval import update_cache

logger = structlog.get_logger(__name__)


def _collect_evidence(messages: list[BaseMessage]) -> str:
    """Concatenate the tool outputs (the ground truth the answer must be built from) for the
    verifier, capped so the verifier request stays within token limits."""
    parts = [str(m.content) for m in messages if isinstance(m, ToolMessage)]
    return "\n\n".join(parts)[:MAX_EVIDENCE_CHARS]


def verify_answer(question: str, answer: str, analyst_messages: list[BaseMessage]) -> str:
    """Run the verifier agent over the analyst's answer and the evidence its tools returned.

    Best-effort: the answer is already produced, so a verifier failure (e.g. a rate limit) must not
    sink the whole response — we return a "verification unavailable" note instead of raising."""
    evidence = _collect_evidence(analyst_messages)
    if not evidence:
        return "Verification: skipped (the answer used no tool evidence)."
    verifier_input = (
        f"QUESTION:\n{question}\n\n"
        f"DRAFT ANSWER:\n{answer}\n\n"
        f"EVIDENCE (tool outputs the answer must be grounded in):\n{evidence}"
    )
    try:
        return VerifierAgent().run(verifier_input)
    except Exception as e:
        logger.warning("Verifier failed; returning the answer without a fact-check: %s", e)
        return f"Verification unavailable ({e.__class__.__name__})."


@traceable(run_type="chain", name="equity_research")
def process_query(query: str, vector_db: FAISS | None = None) -> str:
    """Answer an investing question, then fact-check the answer against the sources.

    The `@traceable` root makes one LangSmith trace per request, with the analyst run, its model
    and tool calls, and the verifier run all nested under it (rather than as loose siblings).

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
    setup_logging()
    setup_sentry()
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


def main() -> None:
    """Answer a single hardcoded question (non-interactive)."""
    setup_logging()
    setup_sentry()
    vector_db = initialize_vectorstore("documents")
    query = "What are the main risk factors, and how does the current P/E compare to the 52-week range?"
    print(process_query(query, vector_db))


if __name__ == "__main__":
    chat()
