"""Logging helpers (console + rotating file)."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: str = "INFO", log_dir: Optional[Path] = None) -> None:
    """Configure the ``deeptrace`` logger once (safe to call repeatedly)."""
    global _CONFIGURED
    root = logging.getLogger("deeptrace")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _CONFIGURED:
        return
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                Path(log_dir) / "deeptrace.log",
                maxBytes=2_000_000,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # logging must never crash the app, but say so
            root.warning("File logging disabled (%s)", exc)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``deeptrace`` namespace."""
    if not name.startswith("deeptrace"):
        name = f"deeptrace.{name}"
    return logging.getLogger(name)
