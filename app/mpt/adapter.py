from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from app.mpt.models import AudioTrack, MediaInfo, MediaProfile


class MediaProcessor(Protocol):
    """Tool-neutral protocol for media probing, normalization, concatenation, and audio muxing.

    Invariant: At the MediaProcessor boundary, all source paths must exist locally
    and be readable. The processor does not resolve cloud artifacts or storage references.
    """

    async def probe(self, source: Path) -> MediaInfo:
        """Probe a local media file and return structured technical metadata."""
        ...

    async def normalize_clip(
        self,
        source: Path,
        output: Path,
        profile: MediaProfile,
        target_duration: float | None = None,
        tolerance_seconds: float = 0.5,
    ) -> MediaInfo:
        """Normalize a video clip to target profile dimensions, fps, format, and reconcile duration."""
        ...

    async def concatenate(
        self,
        clips: Sequence[Path],
        output: Path,
    ) -> MediaInfo:
        """Concatenate normalized video clips in the provided order using the concat demuxer."""
        ...

    async def add_audio(
        self,
        video: Path,
        audio: AudioTrack,
        output: Path,
        video_duration: float,
    ) -> MediaInfo:
        """Mux an audio track with a video, applying looping or silence padding to match video duration."""
        ...


# Backward-compatible boundary alias
MoneyPrinterTurboAdapter = MediaProcessor
