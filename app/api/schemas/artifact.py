"""Artifact response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from app.orchestration.state_machine import ArtifactLifecycleStatus


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
    artifact_metadata: dict[str, Any] | None = None

    @field_validator("storage_path", mode="after")
    @classmethod
    def sanitize_storage_path(cls, v: str) -> str:
        """Sanitize storage path to avoid leaking server filesystem roots."""
        normalized = v.replace("\\", "/")
        if ":" in normalized or normalized.startswith("/"):
            return normalized.split("/")[-1]
        return normalized

    @field_validator("artifact_metadata", mode="after")
    @classmethod
    def sanitize_metadata(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """Filter internal provider tokens and folder IDs from public response."""
        if v is None:
            return None
        sensitive_substrings = ("token", "secret", "credential", "gdrive", "auth")
        return {
            k: val
            for k, val in v.items()
            if not any(sub in k.lower() for sub in sensitive_substrings)
        }


class ArtifactListResponse(BaseModel):
    """List envelope for workflow artifacts."""

    model_config = ConfigDict(extra="forbid")

    artifacts: list[ArtifactResponse]
    total: int
