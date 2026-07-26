from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from pipeline.config import orchestrator_llm
from pipeline.engine import BaseAgent


class _EchoInput(BaseModel):
    x: str = Field(description="anything")


def _echo(x: str) -> str:
    return f"echo:{x}"


def _finish(x: str) -> str:
    return f"final:{x}"


_ECHO_TOOL = StructuredTool.from_function(func=_echo, name="echo", description="echo", args_schema=_EchoInput)
_FINISH_TOOL = StructuredTool.from_function(func=_finish, name="finish", description="finish", args_schema=_EchoInput)


class _Agent(BaseAgent):
    def _process_output(self):
        for message in reversed(self.messages):
            if isinstance(message, AIMessage) and str(message.content).strip():
                return str(message.content)
        return "(none)"


def _agent(final_tool_names=None, max_iter=5):
    return _Agent(
        llm=orchestrator_llm,
        tools=[_ECHO_TOOL, _FINISH_TOOL],
        system_prompt="system",
        final_tool_names=final_tool_names or [],
        max_iter=max_iter,
    )


def _tool_call(name, x, call_id="1"):
    return {"id": call_id, "name": name, "args": {"x": x}, "type": "tool_call"}


def test_run_stops_when_model_answers_in_text(scripted_chat):
    agent = _agent()
    agent.llm = scripted_chat([AIMessage(content="the answer")])
    assert agent.run("q") == "the answer"
    assert agent.llm.calls == 1


def test_run_executes_tool_then_answers(scripted_chat):
    agent = _agent()
    agent.llm = scripted_chat([
        AIMessage(content="", tool_calls=[_tool_call("echo", "hi")]),
        AIMessage(content="done"),
    ])
    assert agent.run("q") == "done"
    assert any(isinstance(m, ToolMessage) and m.content == "echo:hi" for m in agent.messages)


def test_terminal_tool_ends_loop(scripted_chat):
    agent = _agent(final_tool_names=["finish"])
    agent.llm = scripted_chat([
        AIMessage(content="", tool_calls=[_tool_call("finish", "payload")]),
        AIMessage(content="should never run"),
    ])
    agent.run("q")
    assert agent._final_payload == "final:payload"
    assert agent.llm.calls == 1


def test_unknown_tool_is_reported_not_fatal(scripted_chat):
    agent = _agent()
    agent.llm = scripted_chat([
        AIMessage(content="", tool_calls=[_tool_call("nope", "x")]),
        AIMessage(content="recovered"),
    ])
    assert agent.run("q") == "recovered"
    assert any(isinstance(m, ToolMessage) and "Unknown tool" in m.content for m in agent.messages)


def test_max_iter_caps_the_loop(scripted_chat):
    agent = _agent(max_iter=3)
    agent.llm = scripted_chat([AIMessage(content="", tool_calls=[_tool_call("echo", "a")])] * 10)
    agent.run("q")
    assert agent.llm.calls == 3


def test_windowed_messages_drops_old_turns_but_keeps_pairs():
    agent = _agent()
    agent.max_history_tokens = 200
    convo = list(agent.messages)
    for t in range(4):
        convo.append(HumanMessage(content=f"question {t} " * 15))
        convo.append(AIMessage(content="", tool_calls=[_tool_call("echo", "x", call_id=str(t))]))
        convo.append(ToolMessage(content=f"result {t} " * 60, tool_call_id=str(t), name="echo"))
        convo.append(AIMessage(content=f"answer {t}"))
    agent.messages = convo

    windowed = agent._windowed_messages()
    assert len(windowed) < len(agent.messages)
    assert any(isinstance(m, HumanMessage) for m in windowed)
    call_ids = {tc["id"] for m in windowed if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    tool_ids = {m.tool_call_id for m in windowed if isinstance(m, ToolMessage)}
    assert tool_ids <= call_ids


def test_windowed_messages_guard_keeps_current_turn_when_over_budget():
    agent = _agent()
    agent.max_history_tokens = 10
    agent.messages = [
        agent.messages[0],
        HumanMessage(content="a big question " * 50),
        AIMessage(content="", tool_calls=[_tool_call("echo", "x")]),
        ToolMessage(content="a big result " * 100, tool_call_id="1", name="echo"),
    ]
    windowed = agent._windowed_messages()
    assert any(isinstance(m, HumanMessage) for m in windowed)
    call_ids = {tc["id"] for m in windowed if isinstance(m, AIMessage) for tc in (m.tool_calls or [])}
    tool_ids = {m.tool_call_id for m in windowed if isinstance(m, ToolMessage)}
    assert tool_ids <= call_ids
