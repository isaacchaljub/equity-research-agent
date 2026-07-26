"""Observability setup: structured logging (structlog) and error reporting (Sentry).

Kept separate from the domain so the wiring lives in one place and entrypoints just call
`setup_logging()` and `setup_sentry()` at startup."""

import logging
import os

import sentry_sdk
import structlog
from sentry_sdk.integrations.logging import LoggingIntegration


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog over stdlib logging: contextvar-aware, level/timestamp-stamped lines.

    Because it routes through stdlib logging, Sentry's LoggingIntegration still sees the records,
    and %-style calls (`logger.info("x=%s", x)`) keep working via PositionalArgumentsFormatter.
    Swap ConsoleRenderer for structlog.processors.JSONRenderer() to emit JSON logs in production."""
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def setup_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN is set (else a no-op, so local runs and tests need nothing).

    With LoggingIntegration, any ERROR-level log — notably `logger.exception(...)` — auto-reports as
    a Sentry event, and unhandled exceptions (incl. FastAPI request context) are captured too. Errors
    only: performance tracing is left to LangSmith, so traces_sample_rate is 0."""
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return
    sentry_sdk.init(
        dsn=dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        send_default_pii=True,
        traces_sample_rate=0.0,
        integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
    )
    structlog.get_logger(__name__).info("Sentry initialized")
