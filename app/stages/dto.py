"""Strongly typed DTOs for all VideoGen pipeline stages.

Every model in this module is permanently frozen (immutable) and uses
strictly typed fields with zero `Any` usage.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.agents.research.models import ResearchResult
from app.agents.script.models import ScriptResult
from app.agents.storyboard.models import StoryboardResult


class PipelineStage(StrEnum):
    """Canonical sequence of autonomous video generation pipeline stages."""

    RESEARCH = "RESEARCH"
    SCRIPT = "SCRIPT"
    STORYBOARD = "STORYBOARD"
    MEDIA_SYNTHESIS = "MEDIA_SYNTHESIS"
    AUDIO = "AUDIO"
    ASSEMBLY = "ASSEMBLY"
    QC = "QC"
    PUBLISH = "PUBLISH"


class ArtifactKind(StrEnum):
    """Typed classification of pipeline artifacts.

    Every artifact record must be tagged with one of these kinds.
    This prevents arbitrary strings from entering the durable artifact
    contract and ensures an MP4 cannot be accidentally consumed as
    a research result or QC report.
    """

    RESEARCH_RESULT = "RESEARCH_RESULT"
    SCRIPT_RESULT = "SCRIPT_RESULT"
    STORYBOARD_RESULT = "STORYBOARD_RESULT"
    VIDEO_SCENE = "VIDEO_SCENE"
    NARRATION = "NARRATION"
    BACKGROUND_MUSIC = "BACKGROUND_MUSIC"
    MASTER_VIDEO = "MASTER_VIDEO"
    QC_REPORT = "QC_REPORT"
    PUBLICATION_RECORD = "PUBLICATION_RECORD"


# --- Stage 4: Media Synthesis DTOs ---


class VideoClipArtifact(BaseModel):
    """Descriptor for a synthesized video shot clip."""

    model_config = ConfigDict(frozen=True)

    shot_number: int = Field(ge=1, description="1-indexed storyboard shot sequence")
    artifact_id: UUID = Field(description="Unique artifact identifier")
    storage_path: str = Field(min_length=1, description="Relative storage path of video file")
    duration_seconds: float = Field(gt=0.0, description="Measured clip duration in seconds")
    checksum_sha256: str = Field(
        min_length=64, max_length=64, description="Hexadecimal SHA-256 hash"
    )
    provider_operation_id: str | None = Field(
        default=None, description="External provider operation identifier if available"
    )


class MediaSynthesisResult(BaseModel):
    """Output DTO of the MEDIA_SYNTHESIS pipeline stage."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    total_clips: int = Field(ge=1, description="Total number of synthesized shot clips")
    clips: list[VideoClipArtifact] = Field(min_length=1, description="Ordered clip descriptors")
    total_duration_seconds: float = Field(gt=0.0, description="Sum duration of all clips")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Stage 5: Audio DTOs ---


class AudioClipArtifact(BaseModel):
    """Descriptor for an audio clip artifact (speech, music, or SFX)."""

    model_config = ConfigDict(frozen=True)

    clip_type: Literal["narration", "music", "sfx"]
    artifact_id: UUID
    storage_path: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0.0)
    format: Literal["mp3", "wav", "aac"]
    checksum_sha256: str = Field(min_length=64, max_length=64)
    sample_rate: int = Field(ge=8000, description="Sample rate in Hz (e.g., 44100, 48000)")


class AudioStageResult(BaseModel):
    """Output DTO of the AUDIO pipeline stage."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    narration: AudioClipArtifact
    background_music: AudioClipArtifact | None = None
    total_duration_seconds: float = Field(gt=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Stage 6: Assembly DTOs ---


class AssemblyResult(BaseModel):
    """Output DTO of the ASSEMBLY pipeline stage."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    master_artifact_id: UUID
    storage_path: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float = Field(gt=0.0)
    video_codec: str = Field(min_length=1)
    audio_codec: str = Field(min_length=1)
    checksum_sha256: str = Field(min_length=64, max_length=64)
    file_size_bytes: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Stage 7: Quality Control (QC) DTOs ---


class QCMetrics(BaseModel):
    """Granular media quality metrics collected during automated probe."""

    model_config = ConfigDict(frozen=True)

    actual_duration_seconds: float = Field(gt=0.0)
    target_duration_seconds: float = Field(gt=0.0)
    duration_drift_ratio: float = Field(ge=0.0)
    loudness_lufs: float = Field(description="Integrated loudness in LUFS (-24 to -6 typical)")
    has_audio_stream: bool
    has_video_stream: bool
    resolution_width: int = Field(gt=0)
    resolution_height: int = Field(gt=0)
    frame_rate: float = Field(gt=0.0)
    black_frame_count: int = Field(ge=0)
    frozen_frame_count: int = Field(ge=0)


class QCResult(BaseModel):
    """Output DTO of the QC pipeline stage.

    Serves as the durable quality gate prerequisite for PUBLISH.
    """

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    verdict: Literal["PASS", "FAIL"]
    metrics: QCMetrics
    failure_reasons: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# --- Stage 8: Publish DTOs ---


class PublishResult(BaseModel):
    """Output DTO of the PUBLISH pipeline stage."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    provider: str = Field(min_length=1, description="Storage provider used (e.g., 'google_drive')")
    remote_file_id: str = Field(min_length=1, description="Provider file or object identifier")
    public_url: str | None = Field(default=None, description="Publicly accessible viewing URL")
    checksum_sha256: str = Field(min_length=64, max_length=64)
    file_size_bytes: int = Field(gt=0)
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "ArtifactKind",
    "AssemblyResult",
    "AudioClipArtifact",
    "AudioStageResult",
    "MediaSynthesisResult",
    "PipelineStage",
    "PublishResult",
    "QCMetrics",
    "QCResult",
    "ResearchResult",
    "ScriptResult",
    "StoryboardResult",
    "VideoClipArtifact",
]
