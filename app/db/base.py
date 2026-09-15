"""Base SQLAlchemy DeclarativeBase and common column definitions."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


# Cross-dialect JSON type that maps to JSONB in PostgreSQL
JSONVariant = JSON().with_variant(postgresql.JSONB(), "postgresql")


class Base(DeclarativeBase):
    """Root DeclarativeBase for all ORM models."""

    type_annotation_map = {
        dict[str, Any]: JSONVariant,
        datetime: DateTime(timezone=True),
    }


class TimestampMixin:
    """Standard audit timestamps for ORM entities."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
