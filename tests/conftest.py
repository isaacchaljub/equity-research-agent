import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("SERPER_API_KEY", "test-serper-key")


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Make ResilientChat's backoff instant so retry/fallback tests don't actually wait."""
    monkeypatch.setattr("time.sleep", lambda *args, **kwargs: None)


@pytest.fixture
def scripted_chat():
    """Factory for a fake chat model: `scripted_chat([msg1, msg2, ...])` returns an object whose
    `.invoke` yields the scripted messages in order and counts calls. Drop it onto `agent.llm` to
    drive the loop with no network."""

    class ScriptedChat:
        def __init__(self, responses):
            self._responses = list(responses)
            self.calls = 0

        def invoke(self, messages):
            response = self._responses[self.calls]
            self.calls += 1
            return response

    return ScriptedChat
