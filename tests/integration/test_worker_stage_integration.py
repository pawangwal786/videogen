"""Integration tests for JobWorker and StageHandler dispatching on PostgreSQL 16."""

import asyncio
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config.settings import Settings
from app.orchestration import (
    JobStage,
    JobStatus,
    JobWorker,
)
from app.repositories import JobRepository, WorkflowRepository
from app.stages.artifact_store import ArtifactStore
from app.stages.base import StageHandler
from app.stages.context import StageContext
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.stage_artifact_repository import StageArtifactRepository


class ScriptInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    topic: str
    target_duration_seconds: int = 60


class ScriptOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    workflow_id: UUID
    headline: str
    narration_script: str
    word_count: int


class SlowScriptStageHandler(StageHandler[ScriptInput, ScriptOutput]):
    def __init__(self, delay_seconds: float = 0.0, should_fail: bool = False) -> None:
        self.delay_seconds = delay_seconds
        self.should_fail = should_fail
        self.execute_count = 0

    @property
    def stage_name(self) -> str:
        return "script"

    async def load_input(self, context: StageContext) -> ScriptInput:
        return ScriptInput(topic="Quantum Mechanics Explained")

    async def execute(self, input_data: ScriptInput, context: StageContext) -> ScriptOutput:
        self.execute_count += 1
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)
        if self.should_fail:
            raise RuntimeError("Stage execution planned failure")
        return ScriptOutput(
            workflow_id=context.workflow_id,
            headline=f"Script: {input_data.topic}",
            narration_script="Welcome to this video on quantum physics...",
            word_count=120,
        )


@pytest.mark.asyncio
async def test_job_worker_dispatches_stage_handler_and_persists_artifact(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Verify JobWorker dispatches a registered StageHandler and persists the artifact via StageRuntime."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_worker", fsync=False)
    settings = Settings(_env_file=None)
    worker = JobWorker(
        session_factory=session_factory,
        artifact_store=store,
        settings=settings,
        heartbeat_interval_seconds=0.5,
    )

    stage_handler = SlowScriptStageHandler()
    worker.register_stage_handler(
        job_type="script",
        handler=stage_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptOutput,
    )

    # 1. Setup workflow and script job
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)
        wf = await wf_repo.create_workflow(topic="Quantum Mechanics Explained")
        await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="script",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={"topic": wf.topic, "fingerprint": "1" * 64},
        )
        await session.commit()

    # 2. Worker runs once to process the script job
    worked = await worker.run_once()
    assert worked is True
    assert stage_handler.execute_count == 1

    # 3. Assert job was completed and artifact was persisted
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        script_job = next(j for j in jobs if j.logical_key == "script")
        assert script_job.status == JobStatus.COMPLETED.value
        assert script_job.output_payload is not None
        assert script_job.output_payload["headline"] == "Script: Quantum Mechanics Explained"
        assert "_artifact_id" in script_job.output_payload

        # Assert durable artifact exists in DB and is AVAILABLE
        art_repo = StageArtifactRepository(session, store)
        art = await art_repo.get_available(
            UUID(script_job.output_payload["_artifact_id"]),
            verify_integrity=True,
        )
        assert art is not None
        assert art.stage == PipelineStage.SCRIPT.value
        assert art.stage_fingerprint == "1" * 64

    store.close()


@pytest.mark.asyncio
async def test_job_worker_heartbeat_during_stage_execution(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Verify worker's lease heartbeat loop continuously updates heartbeat_at during long stage execution."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_hb", fsync=False)
    settings = Settings(_env_file=None)
    worker = JobWorker(
        session_factory=session_factory,
        artifact_store=store,
        settings=settings,
        heartbeat_interval_seconds=0.2,  # fast heartbeat interval
    )

    # Handler sleeps for 0.8s
    stage_handler = SlowScriptStageHandler(delay_seconds=0.8)
    worker.register_stage_handler(
        job_type="script",
        handler=stage_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptOutput,
    )

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)
        wf = await wf_repo.create_workflow(topic="Heartbeat Test")
        await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="script",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={"topic": wf.topic, "fingerprint": "2" * 64},
        )
        await session.commit()

    worked = await worker.run_once()
    assert worked is True

    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        script_job = await job_repo.get_job(jobs[0].id, load_attempts=True)
        assert script_job is not None
        assert len(script_job.attempts) == 1
        att = script_job.attempts[0]
        # Heartbeat was updated after started_at
        assert att.heartbeat_at >= att.started_at

    store.close()


@pytest.mark.asyncio
async def test_job_worker_stage_handler_failure_handling(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
    tmp_path: Path,
):
    """Verify worker catches stage execution failure, records attempt error, and cleans up active state."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_fail", fsync=False)
    settings = Settings(_env_file=None)
    worker = JobWorker(
        session_factory=session_factory,
        artifact_store=store,
        settings=settings,
    )

    # Handler deliberately fails
    failing_handler = SlowScriptStageHandler(should_fail=True)
    worker.register_stage_handler(
        job_type="script",
        handler=failing_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptOutput,
    )

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)
        wf = await wf_repo.create_workflow(topic="Failing Stage Test")
        await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="script",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={"topic": wf.topic, "fingerprint": "3" * 64},
        )
        await session.commit()

    worked = await worker.run_once()
    assert worked is True

    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        script_job = await job_repo.get_job(jobs[0].id, load_attempts=True)
        assert script_job is not None
        assert len(script_job.attempts) == 1
        assert script_job.attempts[0].status == "FAILED"
        assert "Stage execution planned failure" in (script_job.attempts[0].error_message or "")

    store.close()
