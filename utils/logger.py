"""
Module: utils/logger.py
Purpose: Structured logging setup via structlog. Single configuration point for
         the entire system. All modules call get_logger(__name__).
Phase: All
Dependencies: structlog, config/settings.py
Last modified: 2026-06-10
"""

import logging
import sys
from typing import Any

import structlog

from config.settings import LOG_LEVEL

_configured: bool = False


def _configure() -> None:
    """Configure structlog once. Idempotent."""
    global _configured
    if _configured:
        return

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Use stdlib logger factory so add_logger_name can read .name from the logger
    structlog.configure(
        processors=shared_processors + [structlog.dev.ConsoleRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, LOG_LEVEL, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, LOG_LEVEL, logging.INFO),
    )
    _configured = True


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """
    Return a configured structlog logger bound to the given module name.

    Usage:
        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("event_name", key="value", ...)

    Args:
        name: Module name, typically __name__

    Returns:
        A structlog BoundLogger instance.
    """
    _configure()
    return structlog.get_logger(name)
