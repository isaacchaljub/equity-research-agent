from unittest.mock import MagicMock
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage

from pipeline.agents import AnalystAgent
from pipeline.agents import VerifierAgent
from pipeline.retrieval import query_cache
from pipeline.service import _collect_evidence
from pipeline.service import process_query


def _tool_call(name, args, call_id="1"):
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def _fake_vector_db(passage):
    vdb = MagicMock()
    doc = MagicMock()
    doc.page_content = passage
    vdb.similarity_search.return_value = [doc]
    return vdb


def test_analyst_searches_filings_then_answers(scripted_chat):
    vdb = _fake_vector_db("Apple's trailing P/E is 31.5 according to the filing.")
    analyst = AnalystAgent(vdb)
    analyst.llm = scripted_chat([
        AIMessage(content="", tool_calls=[_tool_call("search_filings", {"query": "P/E"})]),
        AIMessage(content="Apple's trailing P/E is 31.5. Educational, not financial advice."),
    ])

    answer = analyst.run("What is Apple's P/E?")

    assert "31.5" in answer
    vdb.similarity_search.assert_called_once()
    assert "31.5" in _collect_evidence(analyst.messages)


def test_verifier_returns_verdict_from_terminal_tool(scripted_chat):
    verifier = VerifierAgent()
    verifier.llm = scripted_chat([
        AIMessage(content="", tool_calls=[_tool_call(
            "submit_verdict",
            {"verdict": "supported: the P/E matches the filing", "unsupported_claims": []},
        )]),
    ])
    verdict = verifier.run("QUESTION ... DRAFT ... EVIDENCE: trailing P/E 31.5")
    assert verdict.startswith("Verification: supported")


@patch("pipeline.service.VerifierAgent")
@patch("pipeline.service.AnalystAgent")
def test_process_query_glues_answer_and_verdict(mock_analyst_cls, mock_verifier_cls):
    query_cache.clear()
    analyst_instance = mock_analyst_cls.return_value
    analyst_instance.run.return_value = "Apple's P/E is 31.5."
    analyst_instance.messages = [
        ToolMessage(content="filing: trailing P/E 31.5", tool_call_id="1", name="search_filings"),
    ]
    mock_verifier_cls.return_value.run.return_value = "Verification: supported"

    result = process_query("What is Apple's P/E?", MagicMock())

    assert "Apple's P/E is 31.5." in result
    assert "Verification: supported" in result
    query_cache.clear()
