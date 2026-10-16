"""add_phase9_artifact_columns

Revision ID: b5259680b287
Revises: f69e41b953b6
Create Date: 2026-09-17 20:49:37.586018

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5259680b287"
down_revision: str | Sequence[str] | None = "f69e41b953b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add Phase 9 artifact infrastructure columns and indexes."""
    # New columns (all nullable for backward compatibility with Phase 8 artifacts)
    op.add_column("artifacts", sa.Column("stage", sa.String(length=50), nullable=True))
    op.add_column("artifacts", sa.Column("stage_fingerprint", sa.String(length=64), nullable=True))
    op.add_column("artifacts", sa.Column("artifact_kind", sa.String(length=50), nullable=True))

    # Per-column indexes for individual lookups
    op.create_index(op.f("ix_artifacts_stage"), "artifacts", ["stage"], unique=False)
    op.create_index(
        op.f("ix_artifacts_stage_fingerprint"), "artifacts", ["stage_fingerprint"], unique=False
    )

    # Composite index for Phase 9 fingerprint lookup queries:
    #   SELECT ... WHERE workflow_id = ? AND stage = ?
    #     AND stage_fingerprint = ? AND lifecycle_status = 'AVAILABLE'
    op.create_index(
        "ix_artifacts_stage_fingerprint_lookup",
        "artifacts",
        ["workflow_id", "stage", "stage_fingerprint", "lifecycle_status"],
        unique=False,
    )

    # Partial unique index: claim exclusion mechanism for idempotency safety.
    # Prevents concurrent workers from creating duplicate PENDING or AVAILABLE
    # artifacts for the same (workflow_id, stage, stage_fingerprint).
    op.create_index(
        "uq_artifacts_active_fingerprint",
        "artifacts",
        ["workflow_id", "stage", "stage_fingerprint"],
        unique=True,
        postgresql_where=sa.text("lifecycle_status IN ('PENDING', 'AVAILABLE')"),
    )


def downgrade() -> None:
    """Remove Phase 9 artifact infrastructure columns and indexes."""
    op.drop_index(
        "uq_artifacts_active_fingerprint",
        table_name="artifacts",
        postgresql_where=sa.text("lifecycle_status IN ('PENDING', 'AVAILABLE')"),
    )
    op.drop_index("ix_artifacts_stage_fingerprint_lookup", table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_stage_fingerprint"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_stage"), table_name="artifacts")
    op.drop_column("artifacts", "artifact_kind")
    op.drop_column("artifacts", "stage_fingerprint")
    op.drop_column("artifacts", "stage")
