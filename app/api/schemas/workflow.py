"""Workflow request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.orchestration.state_machine import JobStage, WorkflowStatus


class WorkflowCreateRequest(BaseModel):
    """Payload for creating a new video generation workflow."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The topic or prompt for video generation.",
        examples=["Quantum computing explained in 60 seconds"],
    )

    @field_validator("topic")
    @classmethod
    def validate_non_whitespace_topic(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Topic must not be empty or whitespace only.")
        return cleaned


class WorkflowResponse(BaseModel):
    """Public representation of a video generation workflow."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: str
    topic: str
    status: WorkflowStatus
    current_stage: JobStage
    idempotency_key: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
