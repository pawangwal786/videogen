from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.artifacts.models import ArtifactRef


class MediaInfo(BaseModel):
    """Probed metadata for a media file."""

    model_config = ConfigDict(frozen=True)

    duration_seconds: float = Field(ge=0.0, description="Duration in fractional seconds")
    width: int = Field(ge=0, description="Video width in pixels (0 if audio-only)")
    height: int = Field(ge=0, description="Video height in pixels (0 if audio-only)")
    frame_rate: float = Field(ge=0.0, description="Video frame rate in fps (0.0 if audio-only)")
    video_codec: str | None = Field(default=None, description="Video codec name")
    audio_codec: str | None = Field(default=None, description="Audio codec name")
    has_video: bool = True
    has_audio: bool = False
    bitrate: int | None = Field(default=None, ge=0, description="Overall bitrate in bps")
    size_bytes: int | None = Field(default=None, ge=0, description="File size in bytes")


class MediaProfile(BaseModel):
    """Target technical encoding profile for normalization and assembly."""

    model_config = ConfigDict(frozen=True)

    width: int = Field(default=1080, ge=1)
    height: int = Field(default=1920, ge=1)
    frame_rate: int = Field(default=30, ge=1)
    video_codec: str = "libx264"
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_sample_rate: int = Field(default=44100, ge=8000)
    audio_channels: int = Field(default=2, ge=1, le=8)
    container: str = "mp4"

    @classmethod
    def from_aspect_ratio(cls, aspect_ratio: str = "9:16", fps: int = 30) -> MediaProfile:
        """Resolve a standard aspect ratio string to a concrete MediaProfile."""
        normalized = aspect_ratio.strip()
        resolutions: dict[str, tuple[int, int]] = {
            "9:16": (1080, 1920),
            "16:9": (1920, 1080),
            "1:1": (1080, 1080),
            "4:5": (1080, 1350),
            "2:3": (1080, 1620),
        }
        if normalized not in resolutions:
            raise ValueError(
                f"Unsupported aspect ratio: '{aspect_ratio}'. Supported ratios: {list(resolutions.keys())}"
            )
        w, h = resolutions[normalized]
        return cls(width=w, height=h, frame_rate=fps)


class MediaClip(BaseModel):
    """Represents a local video clip entering media assembly."""

    model_config = ConfigDict(frozen=True)

    shot_number: int = Field(ge=1, description="1-indexed shot sequence number")
    source_path: Path = Field(description="Local filesystem path to the clip")
    target_duration_seconds: float = Field(
        gt=0.0, description="Creative storyboard target duration"
    )
    actual_duration_seconds: float | None = Field(
        default=None, gt=0.0, description="Probed actual duration of the source clip"
    )
    artifact_ref: ArtifactRef | None = Field(
        default=None, description="Optional associated storage artifact reference"
    )


class AudioTrack(BaseModel):
    """Configuration for an optional audio track to mux with the video."""

    model_config = ConfigDict(frozen=True)

    source_path: Path = Field(description="Local filesystem path to audio file")
    volume: float = Field(default=1.0, ge=0.0, le=2.0, description="Audio volume scale factor")
    loop: bool = Field(
        default=True,
        description="If True, loop audio to match video duration; if False, play once and pad silence",
    )


class MediaAssemblyRequest(BaseModel):
    """Validated input specification for media assembly."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    clips: list[MediaClip] = Field(
        min_length=1, description="Ordered list of shot clips to assemble"
    )
    profile: MediaProfile = Field(description="Target media profile")
    audio_track: AudioTrack | None = Field(default=None, description="Optional audio track to mux")
    duration_tolerance_seconds: float = Field(
        default=0.5, ge=0.0, description="Acceptable duration discrepancy tolerance in seconds"
    )

    @field_validator("clips")
    @classmethod
    def validate_clips_order_and_uniqueness(cls, clips: list[MediaClip]) -> list[MediaClip]:
        shot_numbers = [c.shot_number for c in clips]
        if len(shot_numbers) != len(set(shot_numbers)):
            duplicates = [num for num in shot_numbers if shot_numbers.count(num) > 1]
            raise ValueError(
                f"Duplicate shot numbers found in media assembly request: {set(duplicates)}"
            )
        sorted_clips = sorted(clips, key=lambda c: c.shot_number)
        expected = list(range(1, len(clips) + 1))
        actual = [c.shot_number for c in sorted_clips]
        if actual != expected:
            raise ValueError(
                f"Clips must form a continuous 1-indexed sequence. Expected {expected}, got {actual}."
            )
        return sorted_clips


class MediaAssemblyResult(BaseModel):
    """Authoritative result returned by media assembly."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    output_path: Path = Field(description="Local filesystem path to the assembled final video")
    duration_seconds: float = Field(gt=0.0, description="Authoritatively probed output duration")
    media_info: MediaInfo = Field(description="Probed metadata of the final video")
    artifact_ref: ArtifactRef | None = Field(
        default=None, description="Cloud storage artifact reference if uploaded"
    )
