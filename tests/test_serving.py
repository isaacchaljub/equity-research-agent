from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient

from serving_api.main import app


@patch("serving_api.main.initialize_vectorstore", return_value=MagicMock())
def test_health_root_and_schema(_mock_init):
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200
        schema = client.get("/openapi.json").json()
        assert "ChatRequest" in schema["components"]["schemas"]


@patch("serving_api.main.verify_answer", return_value="Verification: supported")
@patch("serving_api.main.AnalystAgent")
@patch("serving_api.main.initialize_vectorstore", return_value=MagicMock())
def test_chat_returns_answer_and_verification(_mock_init, mock_agent_cls, _mock_verify):
    mock_agent_cls.return_value.run.return_value = "Apple's P/E is 31.5."
    mock_agent_cls.return_value.messages = []
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "What is Apple's P/E?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Apple's P/E is 31.5."
        assert body["verification"] == "Verification: supported"
        assert body["conversation_id"]
        assert client.post("/chat", json={}).status_code == 422


@patch("serving_api.main.initialize_vectorstore", return_value=MagicMock())
def test_same_conversation_id_reuses_agent(_mock_init):
    with (
        patch("serving_api.main.AnalystAgent") as mock_agent_cls,
        patch("serving_api.main.verify_answer", return_value="Verification: supported"),
    ):
        mock_agent_cls.return_value.run.return_value = "answer"
        mock_agent_cls.return_value.messages = []
        with TestClient(app) as client:
            first = client.post("/chat", json={"message": "hi"}).json()
            cid = first["conversation_id"]
            client.post("/chat", json={"conversation_id": cid, "message": "again"})
            assert mock_agent_cls.call_count == 1
