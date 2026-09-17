"""Job response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.orchestration.state_machine import JobStage, JobStatus


class JobResponse(BaseModel):
    """Public representation of a workflow job."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    workflow_id: str
    logical_key: str
    job_type: str
    stage: JobStage
    status: JobStatus
    max_attempts: int
    current_attempt_id: str | None = None
    available_at: datetime
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


class JobListResponse(BaseModel):
    """List envelope for workflow jobs."""

    model_config = ConfigDict(extra="forbid")

    jobs: list[JobResponse]
    total: int
