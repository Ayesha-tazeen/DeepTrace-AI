"""Custom exceptions used across DeepTrace AI."""

from __future__ import annotations

from typing import Any, List, Optional


class DeepTraceError(Exception):
    """Base class for all DeepTrace errors."""


class MediaValidationError(DeepTraceError):
    """Raised when an input file fails validation."""

    def __init__(
        self,
        message: str,
        errors: Optional[List[str]] = None,
        result: Any = None,
    ) -> None:
        super().__init__(message)
        self.errors: List[str] = errors or []
        self.result = result


class DependencyUnavailableError(DeepTraceError):
    """Raised when an optional runtime dependency (FFmpeg, MediaPipe...) is missing."""


class ProcessingError(DeepTraceError):
    """Raised when media processing fails after validation succeeded."""
