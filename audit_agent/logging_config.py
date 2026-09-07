"""Logging setup shared by the CLI and example scripts."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(verbose: bool = False) -> None:
    """Install a Rich console handler on the package logger.

    Idempotent: calling it twice will not duplicate handlers.
    """
    logger = logging.getLogger("audit_agent")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if logger.handlers:
        return
    handler = RichHandler(rich_tracebacks=True, show_path=verbose, omit_repeated_times=False)
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
