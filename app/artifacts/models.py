from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ArtifactRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)

    workflow_id: UUID

    artifact_type: str = Field(min_length=1, max_length=100)
    filename: str = Field(min_length=1, max_length=500)
    mime_type: str = Field(min_length=1, max_length=200)

    drive_file_id: str = Field(min_length=1)

    size_bytes: int | None = Field(default=None, ge=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
