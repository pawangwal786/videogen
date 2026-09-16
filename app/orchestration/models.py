"""Domain models for workflow orchestration."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.orchestration.state_machine import (
    ArtifactLifecycleStatus,
    AttemptStatus,
    JobStage,
    JobStatus,
    WorkflowStatus,
)


class ReconciliationStatus(StrEnum):
    """Status outcomes for in-flight external provider reconciliation."""

    RESOLVED = "RESOLVED"
    CONFIRMED_ABSENT = "CONFIRMED_ABSENT"
    UNRESOLVED = "UNRESOLVED"


class ReconciliationOutcome(BaseModel):
    """Immutable result of reconciling an in-flight external provider submission."""

    model_config = ConfigDict(frozen=True)

    status: ReconciliationStatus
    provider_operation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None



class Workflow(BaseModel):
    """Domain model representing a video generation workflow."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    topic: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_stage: JobStage = JobStage.RESEARCH
    idempotency_key: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class JobAttempt(BaseModel):
    """Domain model representing a single execution attempt of a job."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    attempt_number: int
    worker_id: str | None = None
    lease_token: str | None = None
    status: AttemptStatus = AttemptStatus.CLAIMED
    started_at: datetime
    heartbeat_at: datetime
    completed_at: datetime | None = None
    provider: str | None = None
    provider_operation_id: str | None = None
    submission_token: str | None = None
    request_payload: dict[str, Any] | None = None
    response_metadata: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None


class Job(BaseModel):
    """Domain model representing a logical unit of work in a workflow."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    workflow_id: str
    logical_key: str
    job_type: str
    stage: JobStage
    status: JobStatus = JobStatus.PENDING
    max_attempts: int = 3
    current_attempt_id: str | None = None
    available_at: datetime
    input_payload: dict[str, Any] = Field(default_factory=dict)
    output_payload: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    version: int = 1
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class Artifact(BaseModel):
    """Domain model representing a persisted artifact."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    workflow_id: str
    job_id: str | None = None
    artifact_type: str
    storage_path: str
    gdrive_file_id: str | None = None
    file_size_bytes: int | None = None
    checksum: str | None = None
    mime_type: str | None = None
    lifecycle_status: ArtifactLifecycleStatus = ArtifactLifecycleStatus.AVAILABLE
    artifact_metadata: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class ClaimedJob(BaseModel):
    """Encapsulates a successfully claimed job and its corresponding active attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    job: Job
    attempt: JobAttempt
