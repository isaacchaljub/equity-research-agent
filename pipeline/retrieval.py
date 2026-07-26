"""Retrieval layer: the semantic answer cache, the FAISS filing store, and SEC EDGAR fetching."""

import os

import numpy as np
import requests
import structlog
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy import dot
from numpy.linalg import norm

from pipeline.config import MAX_CACHE_SIZE
from pipeline.config import SEC_HEADERS
from pipeline.config import SIMILARITY_THRESHOLD
from pipeline.config import global_embeddings

logger = structlog.get_logger(__name__)

_ticker_to_cik: dict[str, str] = {}
query_cache: list[tuple[list[float], str, str]] = []


def cosine_similarity(query_embedding, cached_embedding):
    return dot(query_embedding, cached_embedding) / (norm(query_embedding) * norm(cached_embedding))


def check_cache(query: str) -> str | None:
    """Return a cached answer for a sufficiently similar past query, else None."""
    if not query_cache:
        return None
    embedded_query = global_embeddings.embed_query(query)
    scores = [cosine_similarity(embedded_query, item[0]) for item in query_cache]
    if np.max(scores) > SIMILARITY_THRESHOLD:
        return query_cache[int(np.argmax(scores))][2]
    return None


def update_cache(query: str, answer: str) -> None:
    query_cache.append((global_embeddings.embed_query(query), query, answer))
    if len(query_cache) > MAX_CACHE_SIZE:
        query_cache.pop(0)


def load_all_documents(documents_directory="documents"):
    """Load every PDF and text filing in the directory into LangChain Documents."""
    all_documents = []
    for file in os.listdir(documents_directory):
        path = os.path.join(documents_directory, file)
        if file.endswith(".pdf"):
            all_documents.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            all_documents.extend(TextLoader(path).load())
    return all_documents


def create_vector_database(documents):
    """Chunk the documents and index them in FAISS for similarity search."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, global_embeddings)


def initialize_vectorstore(documents_directory="documents"):
    """Build a FAISS store from any local filings, or return None if the folder is empty.

    Returning None (instead of raising) lets the app start with no local PDFs and pull filings on
    demand via `fetch_filing`."""
    all_documents = load_all_documents(documents_directory)
    if not all_documents:
        logger.info("No local filings found; starting empty (use fetch_filing to pull 10-Ks from EDGAR).")
        return None
    return create_vector_database(all_documents)


def _resolve_cik(ticker: str) -> str | None:
    """Map a ticker to its zero-padded SEC CIK, caching the full ticker->CIK table after the first call."""
    global _ticker_to_cik
    if not _ticker_to_cik:
        response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=20)
        response.raise_for_status()
        _ticker_to_cik = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in response.json().values()}
    return _ticker_to_cik.get(ticker.upper())


def _latest_10k(cik: str) -> dict | None:
    """Return the most recent 10-K's accession/document/date from EDGAR's submissions feed, or None."""
    response = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=SEC_HEADERS, timeout=20)
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]
    for form, accession, document, date in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"], recent["filingDate"], strict=False
    ):
        if form == "10-K":
            return {"accession": accession, "document": document, "date": date}
    return None


def _download_filing_text(cik: str, accession: str, document: str) -> str:
    """Download a filing's primary document and extract readable text.

    Modern 10-Ks are inline-XBRL: they carry a lot of machine-readable tag metadata in hidden and
    script/style elements. We drop those before extracting so the index holds prose, not XBRL noise."""
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"
    response = requests.get(url, headers=SEC_HEADERS, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for hidden in soup.select('[style*="display:none"], [style*="display: none"]'):
        hidden.decompose()
    return soup.get_text(" ", strip=True)


def fetch_10k_chunks(ticker: str) -> tuple[list[Document], str]:
    """Fetch a ticker's latest 10-K from EDGAR and return (chunked Documents, filing date).

    Raises ValueError with a user-facing message if the ticker or filing can't be found."""
    cik = _resolve_cik(ticker)
    if cik is None:
        raise ValueError(f"Ticker '{ticker}' was not found in SEC EDGAR.")
    filing = _latest_10k(cik)
    if filing is None:
        raise ValueError(f"No 10-K filing found for {ticker.upper()} on SEC EDGAR.")
    text = _download_filing_text(cik, filing["accession"], filing["document"])
    document = Document(page_content=text, metadata={"ticker": ticker.upper(), "form": "10-K", "filed": filing["date"]})
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents([document])
    return chunks, filing["date"]
