# syntax=docker/dockerfile:1
# =============================================================================
# Equity Research Agent — FastAPI service, CPU-only.
#
# Multi-stage build:
#   Stage 1 "builder"  — install Python deps + pre-download the embedding model.
#   Stage 2 "runtime"  — copy ONLY what's needed to run; small, non-root image.
#
# Why multi-stage? The tools used to BUILD an image (uv, caches, compilers) are
# dead weight at RUNTIME. We do the heavy work in "builder", then copy just the
# finished virtualenv + model cache + app code into a clean final image.
# =============================================================================


# ---------- Stage 1: builder -------------------------------------------------
# Pin the Python version to match the project (.python-version = 3.12).
# "slim" = Debian minus docs/man/extras -> smaller. "bookworm" = Debian 12.
FROM python:3.12-slim-bookworm AS builder

# Bring in the uv package manager by copying its static binaries from the
# official uv image. Pinning the version keeps builds reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /bin/

# uv behavior knobs:
#   UV_COMPILE_BYTECODE=1     -> precompile .pyc at build time -> faster cold starts.
#   UV_LINK_MODE=copy         -> COPY packages into the venv (don't hardlink to uv's
#                                cache). Essential for multi-stage: the venv must be
#                                self-contained so it still works after we copy it.
#   UV_PYTHON_DOWNLOADS=never -> use the image's Python, don't fetch another one.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# faiss-cpu and torch link against OpenMP (libgomp) at import time — needed here
# because we import them below to pre-download the model.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Dependency layer (cached) ----------------------------------------------
# Copy ONLY the dependency manifests first. Docker caches layers by their
# inputs, so as long as pyproject.toml + uv.lock don't change, this expensive
# step is reused even when you edit application code. Big build-speed win.
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv from the locked versions.
#   --frozen             -> fail if uv.lock is out of date (reproducible builds).
#   --no-install-project -> install deps only, not our own code (we run from src).
#   --no-dev             -> skip dev/test dependency groups.
RUN uv sync --frozen --no-install-project --no-dev

# --- Pre-bake the embedding model -------------------------------------------
# The app embeds with sentence-transformers/all-MiniLM-L6-v2 (~90 MB). By
# default it downloads on first startup — slow, and needs network at runtime.
# We download it now, into a cache dir we ship in the final image, so the
# container starts fast and needs NO Hugging Face network access to run.
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c "\
from langchain_huggingface.embeddings import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2').embed_query('warmup')"


# ---------- Stage 2: runtime -------------------------------------------------
# Start clean from the same base (guarantees the venv's Python matches).
FROM python:3.12-slim-bookworm AS runtime

# Same OpenMP runtime lib — required by faiss/torch when the app runs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user. Never run app processes as root: if the app is
# compromised, the blast radius is a nobody-user, not root.
RUN groupadd --system app \
 && useradd --system --gid app --home-dir /app --create-home app

# Runtime environment:
#   PATH                    -> put the venv first so `python`/`fastapi` = venv's.
#   PYTHONUNBUFFERED        -> stream logs immediately -> visible in `docker logs`
#                              / CloudWatch / Azure log stream.
#   PYTHONDONTWRITEBYTECODE -> don't scatter .pyc at runtime (already compiled).
#   HF_HOME                 -> point at the model cache we baked in.
#   *_OFFLINE               -> force load-from-cache; never phone home to HF at runtime.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# Copy the finished virtualenv and the baked model cache from the builder.
# --chown sets ownership in the same step (avoids a second, size-doubling layer).
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/hf   /opt/hf

# Copy ONLY the code this API needs. `serving_api` and `pipeline` are top-level
# packages so the absolute import `from pipeline...` resolves. `documents/` may be
# empty — the agent fetches 10-Ks from SEC EDGAR on demand.
COPY --chown=app:app serving_api/ ./serving_api/
COPY --chown=app:app pipeline/    ./pipeline/
COPY --chown=app:app documents/   ./documents/

# Drop privileges for everything below.
USER app

# Document the port the app listens on (informational; publish with -p at run).
EXPOSE 8000

# Container-level health probe. Orchestrators (ECS, Container Apps) use this to
# know when the app is actually ready vs. just "process started".
#   start-period=120s -> grace window while it loads the embedding model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"]

# Start the API. `fastapi run` = production mode (no autoreload) on uvicorn.
# --host 0.0.0.0 makes it reachable from outside the container (not just localhost).
CMD ["fastapi", "run", "serving_api/main.py", "--host", "0.0.0.0", "--port", "8000"]
