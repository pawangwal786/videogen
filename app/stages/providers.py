"""Typed provider capability interfaces and contracts.

Defines abstract request/response models and protocols at the capability
boundary (Text, Video, Speech, Storage). Zero vendor-specific types and
zero `Any` are permitted across this boundary.
"""

from datetime import UTC, datetime
from typing import Literal, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# --- Text Generation Capability ---


class TextGenerationRequest(BaseModel):
    """Input contract for text generation requests."""

    model_config = ConfigDict(frozen=True)

    user_prompt: str = Field(min_length=1, description="Primary user instruction prompt")
    system_prompt: str | None = Field(
        default=None, description="System instructions establishing persona or constraints"
    )
    response_model: type[BaseModel] | None = Field(
        default=None, description="Target Pydantic schema model for structured output validation"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=32768)


class TextGenerationResult(BaseModel):
    """Output contract for text generation responses."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(min_length=1, description="Generated raw text or serialized JSON string")
    provider: str = Field(min_length=1, description="Capability provider name (e.g., 'gemini')")
    model: str = Field(min_length=1, description="Underlying model identifier")
    usage_prompt_tokens: int = Field(default=0, ge=0)
    usage_completion_tokens: int = Field(default=0, ge=0)


@runtime_checkable
class TextGenerationProvider(Protocol):
    """Abstract protocol for text generation capability providers."""

    async def generate(self, request: TextGenerationRequest) -> TextGenerationResult:
        """Execute text generation from prompt instructions."""
        ...


# --- Video Generation Capability ---


class VideoGenerationRequest(BaseModel):
    """Input contract for shot video synthesis requests."""

    model_config = ConfigDict(frozen=True)

    client_request_id: str = Field(
        min_length=1, description="Deterministic idempotency key for deduplication"
    )
    prompt: str = Field(min_length=10, max_length=1000, description="Visual video prompt")
    negative_prompt: str | None = Field(
        default=None, max_length=500, description="Visual elements to avoid"
    )
    duration_seconds: int = Field(default=5, ge=1, le=60, description="Duration in seconds")
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    seed: int | None = None


class VideoSubmission(BaseModel):
    """Initial asynchronous response upon submitting work to a video provider."""

    model_config = ConfigDict(frozen=True)

    provider_operation_id: str = Field(
        min_length=1, description="External provider operation identifier"
    )
    status: Literal["PENDING", "PROCESSING"]
    estimated_duration_seconds: int | None = None


class VideoGenerationResult(BaseModel):
    """Result of an asynchronous video generation operation."""

    model_config = ConfigDict(frozen=True)

    provider_operation_id: str = Field(min_length=1)
    video_bytes: bytes | None = None
    download_url: str | None = None
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    error_message: str | None = None


class VideoReconciliationResult(BaseModel):
    """Result of reconciling an in-flight operation by client request ID."""

    model_config = ConfigDict(frozen=True)

    status: Literal["RESOLVED", "CONFIRMED_ABSENT", "UNRESOLVED"]
    provider_operation_id: str | None = None
    error_message: str | None = None


@runtime_checkable
class VideoGenerationProvider(Protocol):
    """Abstract protocol for video generation capability providers."""

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        """Asynchronously submit a video generation request to the provider."""
        ...

    async def poll(self, operation_id: str) -> VideoGenerationResult:
        """Poll status of an in-flight video generation operation."""
        ...

    async def reconcile(self, client_request_id: str) -> VideoReconciliationResult:
        """Reconcile in-flight operation by client request ID for crash recovery."""
        ...


# --- Text-to-Speech (TTS) Capability ---


class TTSRequest(BaseModel):
    """Input contract for speech synthesis requests."""

    model_config = ConfigDict(frozen=True)

    client_request_id: str = Field(
        min_length=1, description="Deterministic idempotency key for deduplication"
    )
    text: str = Field(min_length=1, description="Spoken narration text")
    voice_id: str = Field(min_length=1, description="Voice identifier")
    stability: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)


class TTSResult(BaseModel):
    """Output contract for synthesized speech audio."""

    model_config = ConfigDict(frozen=True)

    audio_bytes: bytes = Field(min_length=1)
    audio_format: Literal["mp3", "wav", "aac"]
    duration_seconds: float = Field(gt=0.0)
    sample_rate: int = Field(ge=8000)


@runtime_checkable
class TextToSpeechProvider(Protocol):
    """Abstract protocol for text-to-speech capability providers."""

    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """Synthesize spoken audio from text."""
        ...


# --- Cloud Storage Capability ---


class StorageUploadRequest(BaseModel):
    """Input contract for uploading an artifact to durable cloud storage."""

    model_config = ConfigDict(frozen=True)

    client_request_id: str = Field(
        min_length=1, description="Deterministic idempotency key for upload"
    )
    workflow_id: UUID
    local_file_path: str = Field(min_length=1, description="Resolved path of local file")
    destination_folder_id: str = Field(min_length=1, description="Remote root or folder ID")
    file_name: str = Field(min_length=1, description="Target remote filename")
    mime_type: str = Field(min_length=1, description="MIME content type")


class StorageResult(BaseModel):
    """Output contract confirming remote storage publication."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1, description="Storage provider (e.g., 'google_drive')")
    remote_file_id: str = Field(min_length=1, description="Remote file or object ID")
    public_url: str | None = Field(default=None, description="Public viewing URL if available")
    checksum_sha256: str = Field(
        min_length=64, max_length=64, description="Verified remote file SHA-256 hash"
    )
    file_size_bytes: int = Field(ge=0)
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@runtime_checkable
class StorageProvider(Protocol):
    """Abstract protocol for remote storage capability providers."""

    async def upload(self, request: StorageUploadRequest) -> StorageResult:
        """Upload a local file to remote storage and return publication metadata."""
        ...


__all__ = [
    "StorageProvider",
    "StorageResult",
    "StorageUploadRequest",
    "TTSRequest",
    "TTSResult",
    "TextGenerationProvider",
    "TextGenerationRequest",
    "TextGenerationResult",
    "TextToSpeechProvider",
    "VideoGenerationProvider",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoReconciliationResult",
    "VideoSubmission",
]
