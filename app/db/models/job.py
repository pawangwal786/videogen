"""Job and JobAttempt ORM models."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JSONVariant, utc_now

if TYPE_CHECKING:
    from app.db.models.artifact import ArtifactModel
    from app.db.models.workflow import WorkflowModel


class JobModel(Base):
    """Represents a logical unit of execution in a workflow.
    
    A job has a unique logical identity within a workflow (workflow_id, logical_key).
    Execution attempts are tracked separately in JobAttemptModel.
    """

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    workflow_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    logical_key: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
        index=True,
    )

    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    current_attempt_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )

    input_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant,
        default=dict,
        nullable=False,
    )
    output_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONVariant,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    workflow: Mapped["WorkflowModel"] = relationship(
        "WorkflowModel",
        back_populates="jobs",
    )
    attempts: Mapped[list["JobAttemptModel"]] = relationship(
        "JobAttemptModel",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="JobAttemptModel.attempt_number",
    )
    artifacts: Mapped[list["ArtifactModel"]] = relationship(
        "ArtifactModel",
        back_populates="job",
    )

    __table_args__ = (
        UniqueConstraint("workflow_id", "logical_key", name="uq_jobs_workflow_logical_key"),
        Index("ix_jobs_claim_poll", "status", "available_at"),
    )


class JobAttemptModel(Base):
    """Represents a single execution attempt of a Job.
    
    Contains worker lease ownership (worker_id, lease_token) and external
    provider operation linkage (provider, provider_operation_id, submission_token).
    """

    __tablename__ = "job_attempts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)

    worker_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    lease_token: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="CLAIMED",
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)
    provider_operation_id: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )
    submission_token: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )

    request_payload: Mapped[dict[str, Any] | None] = mapped_column(
        JSONVariant,
        nullable=True,
    )
    response_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONVariant,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    job: Mapped["JobModel"] = relationship(
        "JobModel",
        back_populates="attempts",
    )

    __table_args__ = (
        UniqueConstraint("job_id", "attempt_number", name="uq_job_attempts_job_attempt"),
        Index("ix_job_attempts_lease", "job_id", "worker_id", "lease_token"),
    )
