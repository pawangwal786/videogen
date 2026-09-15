"""Workflow ORM model."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.db.models.artifact import ArtifactModel
    from app.db.models.job import JobModel


class WorkflowModel(Base, TimestampMixin):
    """Represents a top-level video generation workflow."""

    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
        index=True,
    )
    current_stage: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="RESEARCH",
        index=True,
    )
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
    )
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Relationships
    jobs: Mapped[list["JobModel"]] = relationship(
        "JobModel",
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="JobModel.created_at",
    )
    artifacts: Mapped[list["ArtifactModel"]] = relationship(
        "ArtifactModel",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_workflows_status_stage", "status", "current_stage"),
    )
