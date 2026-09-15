from __future__ import annotations

from typing import Any
from uuid import UUID


class MediaProcessingError(Exception):
    """Base exception for all media assembly and processing failures."""

    def __init__(
        self,
        message: str,
        *,
        workflow_id: UUID | None = None,
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.workflow_id = workflow_id
        self.operation = operation
        self.context = context or {}


class MediaConfigurationError(MediaProcessingError):
    """Raised when external media binaries (ffmpeg, ffprobe) are missing or misconfigured."""


class MediaProbeError(MediaProcessingError):
    """Raised when probing media fails, returns malformed output, or lacks required streams."""


class MediaValidationError(MediaProcessingError):
    """Raised when input media files are missing, unreadable, or fail validation checks."""


class MediaEncodingError(MediaProcessingError):
    """Raised when FFmpeg normalization or re-encoding fails."""


class MediaConcatenationError(MediaProcessingError):
    """Raised when concatenating media clips fails."""


class MediaAudioMuxError(MediaProcessingError):
    """Raised when audio track processing, looping, volume adjustment, or muxing fails."""


class MediaTimeoutError(MediaProcessingError):
    """Raised when a media processing subprocess execution exceeds its configured timeout."""
