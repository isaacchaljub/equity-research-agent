"""The reusable, domain-agnostic agent engine: a resilient model wrapper and a ReAct loop.

Nothing here knows about filings, market data, or verifiers — it's a small framework you could
lift into any tool-using agent project."""

import time
from abc import ABC
from abc import abstractmethod

import structlog
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
from langsmith import traceable

logger = structlog.get_logger(__name__)


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

    @traceable(run_type="llm", name="model", process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != "self"})
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
