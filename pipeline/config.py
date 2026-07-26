"""Central configuration: API keys, model clients, embeddings, and tunable constants.

Everything the rest of the package reads from one place, so swapping a model or a limit is a
one-line change here rather than a hunt through the codebase."""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_litellm import ChatLiteLLM

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "equity-research-agent admin@example.com")
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
global_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

MAX_SEARCH_CHARS = 2000
MAX_SCRAPE_CHARS = 3000
MAX_FILINGS_CHARS = 4000
MAX_EVIDENCE_CHARS = 6000

MAX_CACHE_SIZE = 20
SIMILARITY_THRESHOLD = 0.85

web_llm = ChatLiteLLM(
    model="gemini/gemini-flash-latest",
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
