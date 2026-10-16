"""Integration tests for Alembic database migrations."""

import asyncio

import pytest
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import command


@pytest.mark.asyncio
async def test_qc_verdict_migration_lifecycle(pg_engine: AsyncEngine, postgres_url: str):
    """Verify upgrade and downgrade lifecycle of bf41e258f096_add_workflow_qc_verdict."""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", postgres_url)

    # 1. Downgrade to b5259680b287 (run in thread to allow its own event loop)
    await asyncio.to_thread(command.downgrade, alembic_cfg, "b5259680b287")

    def get_workflow_columns(connection):
        insp = inspect(connection)
        return {col["name"] for col in insp.get_columns("workflows")}

    async with pg_engine.connect() as conn:
        cols_before = await conn.run_sync(get_workflow_columns)
        assert "qc_verdict" not in cols_before

    # 2. Upgrade to head (bf41e258f096)
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")

    async with pg_engine.connect() as conn:
        cols_after = await conn.run_sync(get_workflow_columns)
        assert "qc_verdict" in cols_after

        # Verify insertion and selection with qc_verdict works
        await conn.execute(
            text(
                "INSERT INTO workflows (id, topic, status, current_stage, qc_verdict, created_at, updated_at) "
                "VALUES ('mig-test-wf', 'Migration Test', 'PENDING', 'RESEARCH', 'PASS', NOW(), NOW())"
            )
        )
        row = (
            await conn.execute(text("SELECT qc_verdict FROM workflows WHERE id = 'mig-test-wf'"))
        ).scalar_one()
        assert row == "PASS"

        await conn.execute(text("DELETE FROM workflows WHERE id = 'mig-test-wf'"))
        await conn.commit()
