"""Artifact response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from app.orchestration.state_machine import ArtifactLifecycleStatus


class PublicArtifactMetadata(BaseModel):
    """Explicit public metadata projection for artifacts, preventing arbitrary internal leakage.

    Only explicitly modeled public fields are allowed; all internal or provider-specific
    metadata attributes are dropped by construction.
    """

    model_config = ConfigDict(extra="ignore")

    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    frame_rate: float | None = None
    codec: str | None = None
    audio_codec: str | None = None
    sample_rate: int | None = None
    bitrate_kbps: int | None = None
    format: str | None = None


class ArtifactResponse(BaseModel):
    """Public representation of an artifact produced by a workflow."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    workflow_id: str
    job_id: str | None = None
    artifact_type: str
    storage_path: str
    mime_type: str | None = None
    file_size_bytes: int | None = None
    checksum: str | None = None
    status: ArtifactLifecycleStatus = ArtifactLifecycleStatus.PENDING
    created_at: datetime
    artifact_metadata: PublicArtifactMetadata | None = None

    @field_validator("storage_path", mode="after")
    @classmethod
    def sanitize_storage_path(cls, v: str) -> str:
        """Sanitize storage path to avoid leaking server filesystem roots."""
        normalized = v.replace("\\", "/")
        if ":" in normalized or normalized.startswith("/"):
            return normalized.split("/")[-1]
        return normalized


class ArtifactListResponse(BaseModel):
    """List envelope for workflow artifacts."""

    model_config = ConfigDict(extra="forbid")

    artifacts: list[ArtifactResponse]
    total: int
