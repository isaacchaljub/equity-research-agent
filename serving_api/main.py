"""main.py — FastAPI serving layer for the equity-research analyst.

Endpoints:
- GET  /            -> index of endpoints
- GET  /health      -> readiness
- GET  /docs        -> interactive Swagger UI
- POST /chat        -> {message, conversation_id?} -> {conversation_id, answer, verification}
- POST /query       -> {query} -> {answer}   (stateless one-shot; verification is appended inline)

Each conversation id maps to its own persistent AnalystAgent, so history carries across turns.
A per-conversation lock serializes turns within one conversation; different conversations run
concurrently. Conversation state is in-memory (single process) — use a shared store for multiple
workers.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from pydantic import BaseModel
from pydantic import Field

from pipeline.analyst import AnalystAgent
from pipeline.analyst import initialize_vectorstore
from pipeline.analyst import process_query
from pipeline.analyst import verify_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = getLogger(__name__)

_agents: dict[str, AnalystAgent] = {}
_locks: dict[str, asyncio.Lock] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The user's message for this turn.")
    conversation_id: str | None = Field(
        default=None, description="Omit to start a new conversation; pass the returned id to continue one."
    )


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="A one-shot investing question (no conversation history).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, ready
    try:
        vector_db = initialize_vectorstore(str(Path(__file__).parent.parent / "documents"))
        logger.info("Filing vector store loaded")
        ready = True
    except Exception as e:
        ready = False
        logger.error("Failed to initialize the vector store: %s", e)
        raise Exception(f"Failed to initialize the vector store: {e}")

    yield

    logger.info("Shutting down")
    _agents.clear()
    _locks.clear()
    vector_db = None


app = FastAPI(
    lifespan=lifespan,
    title="Equity Research Agent",
    description="An agentic equity-research assistant with a fact-checking verifier. Educational, not financial advice.",
    version="1.0.0",
)


def _answer_and_verify(agent: AnalystAgent, message: str) -> tuple[str, str]:
    answer = agent.run(message)
    verdict = verify_answer(message, answer, agent.messages)
    return answer, verdict


@app.get("/")
async def root():
    """Index so a browser GET / is not a bare 404. The chat/query endpoints are POST."""
    return {
        "service": "Equity Research Agent",
        "disclaimer": "Educational tool, not financial advice.",
        "endpoints": {
            "GET /health": "readiness check",
            "POST /chat": "{message, conversation_id?} -> {conversation_id, answer, verification}",
            "POST /query": "{query} -> {answer} (stateless; verification appended inline)",
            "GET /docs": "interactive Swagger UI",
        },
    }


@app.get("/health")
async def get_health():
    if ready:
        return Response(content="OK", status_code=200)
    return Response(content="NOT OK", status_code=500)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Continue (or start) a conversation. Omit conversation_id to start a new one; pass the
    returned id on later turns to keep context. The answer is fact-checked before returning."""
    try:
        conversation_id = req.conversation_id or str(uuid.uuid4())
        if conversation_id not in _agents:
            _agents[conversation_id] = AnalystAgent(vector_db)
        lock = _locks.setdefault(conversation_id, asyncio.Lock())

        async with lock:
            answer, verification = await asyncio.to_thread(_answer_and_verify, _agents[conversation_id], req.message)

        return {"conversation_id": conversation_id, "answer": answer, "verification": verification}

    except Exception as e:
        logger.error("Error in chat: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def answer_query(req: QueryRequest):
    """Stateless one-shot question. The verification note is appended to the answer string."""
    try:
        answer = await asyncio.to_thread(process_query, req.query, vector_db)
        return {"answer": answer}

    except Exception as e:
        logger.error("Error answering query: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with a consistent error format."""
    logger.error("HTTP exception: %s - %s", exc.status_code, exc.detail)
    return Response(status_code=exc.status_code, content=f"HTTP error: {exc.status_code} - {exc.detail}")
