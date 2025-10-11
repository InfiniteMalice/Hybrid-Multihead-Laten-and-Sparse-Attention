"""Lightweight logging wrappers for latent attention experiments."""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with MLA-specific format."""

    logger = logging.getLogger("deepseek_latent_attention")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


@contextmanager
def log_section(logger: logging.Logger, message: str) -> Iterator[None]:
    """Context manager logging entry/exit for code sections."""

    logger.info("BEGIN: %s", message)
    try:
        yield
    finally:
        logger.info("END: %s", message)


__all__ = ["setup_logging", "log_section"]
