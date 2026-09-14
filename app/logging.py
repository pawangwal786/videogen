import logging
import sys
from typing import Any
from uuid import UUID

import structlog


def setup_logging(env: str = "development", log_level: str = "INFO") -> None:
    """Configure structlog and standard logging based on environment."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if env == "production":
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]

    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )


def bind_correlation(
    workflow_id: UUID | str | None = None,
    task_id: UUID | str | None = None,
) -> None:
    """Bind workflow and task correlation IDs into logging context."""
    kwargs: dict[str, Any] = {}
    if workflow_id is not None:
        kwargs["workflow_id"] = str(workflow_id)
    if task_id is not None:
        kwargs["task_id"] = str(task_id)
    if kwargs:
        structlog.contextvars.bind_contextvars(**kwargs)


def clear_correlation() -> None:
    """Clear correlation IDs from logging context."""
    structlog.contextvars.unbind_contextvars("workflow_id", "task_id")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
