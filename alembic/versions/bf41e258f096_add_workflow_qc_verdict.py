"""add_workflow_qc_verdict

Revision ID: bf41e258f096
Revises: b5259680b287
Create Date: 2026-10-14 17:33:49.532755

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "bf41e258f096"
down_revision: str | Sequence[str] | None = "b5259680b287"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add qc_verdict column to workflows table."""
    op.add_column("workflows", sa.Column("qc_verdict", sa.String(length=50), nullable=True))


def downgrade() -> None:
    """Remove qc_verdict column from workflows table."""
    op.drop_column("workflows", "qc_verdict")
