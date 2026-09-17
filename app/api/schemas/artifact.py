"""Artifact response schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

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


class ArtifactListResponse(BaseModel):
    """List envelope for workflow artifacts."""

    model_config = ConfigDict(extra="forbid")

    artifacts: list[ArtifactResponse]
    total: int
