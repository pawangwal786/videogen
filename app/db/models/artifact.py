"""Artifact ORM model."""

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, JSONVariant, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.job import JobModel
    from app.db.models.workflow import WorkflowModel


class ArtifactModel(Base, TimestampMixin):
    """Represents a persisted pipeline artifact linked to a workflow and optional job."""

    __tablename__ = "artifacts"

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
    job_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    artifact_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    storage_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )
    gdrive_file_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    checksum: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    lifecycle_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="AVAILABLE",
        index=True,
    )
    artifact_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONVariant,
        nullable=True,
    )

    # Relationships
    workflow: Mapped["WorkflowModel"] = relationship(
        "WorkflowModel",
        back_populates="artifacts",
    )
    job: Mapped["JobModel | None"] = relationship(
        "JobModel",
        back_populates="artifacts",
    )

    __table_args__ = (Index("ix_artifacts_workflow_type", "workflow_id", "artifact_type"),)
