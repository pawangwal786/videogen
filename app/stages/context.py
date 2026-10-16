"""Immutable stage execution context."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.config.settings import Settings


class EmptyPayload(BaseModel):
    """Empty payload representation for stages without custom input payloads."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class StageContext[TPayload: BaseModel](BaseModel):
    """Immutable execution context provided to a StageHandler.

    Guarantees read-only access to identity, lease, environment parameters,
    and typed job input payload during stage execution. Any attempt to mutate
    fields raises a ValidationError.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    workflow_id: UUID
    job_id: str = Field(min_length=1, description="Durable job identifier")
    attempt_id: str = Field(min_length=1, description="Durable attempt identifier")
    worker_id: str = Field(min_length=1, description="Claiming worker identifier")
    lease_token: str = Field(min_length=1, description="Active lease token")
    attempt_number: int = Field(ge=1, description="1-indexed attempt number")
    settings: Settings = Field(description="Immutable application configuration")
    payload: TPayload = Field(
        default_factory=EmptyPayload,
        description="Validated immutable job input payload model with zero untyped dict leakage",
    )
