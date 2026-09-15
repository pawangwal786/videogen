"""Integration tests for WorkflowOrchestrator and JobWorker against PostgreSQL 16."""

import asyncio
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration import (
    JobStage,
    JobStatus,
    JobWorker,
    WorkflowOrchestrator,
    WorkflowStatus,
)
from app.repositories import JobRepository, WorkflowRepository


class DummyHandler:
    """Mock job handler returning deterministic output."""

    def __init__(self, output: dict[str, Any], fail: bool = False, two_phase: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.two_phase = two_phase

    async def execute(
        self, job: JobModel, attempt: JobAttemptModel, worker: JobWorker
    ) -> dict[str, Any]:
        if self.two_phase:
            await worker.record_submission_pending(
                provider="test_provider", request_payload=job.input_payload
            )
            await worker.record_submitted(provider_operation_id="test-op-12345")

        if self.fail:
            raise RuntimeError("Handler deliberate failure")

        return self.output


@pytest.mark.asyncio
async def test_orchestrator_workflow_creation_and_cancellation(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)

    wf = await orchestrator.create_workflow(topic="Quantum Encryption")
    assert wf.topic == "Quantum Encryption"
    assert wf.status == WorkflowStatus.PENDING
    assert wf.current_stage == JobStage.RESEARCH

    # Check that initial research job was created
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        assert len(jobs) == 1
        assert jobs[0].logical_key == "research"
        assert jobs[0].stage == JobStage.RESEARCH.value

    # Cancel workflow
    cancelled_wf = await orchestrator.cancel_workflow(wf.id)
    assert cancelled_wf.status == WorkflowStatus.CANCELLED

    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        assert jobs[0].status == JobStatus.CANCELLED.value


@pytest.mark.asyncio
async def test_worker_execution_and_two_phase_submission(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)
    worker = JobWorker(session_factory=session_factory, heartbeat_interval_seconds=1.0)

    # Register two-phase handler
    handler = DummyHandler(output={"result": "ok"}, two_phase=True)
    worker.register_handler("research", handler)

    wf = await orchestrator.create_workflow(topic="Deep Learning History")

    # Run worker once
    worked = await worker.run_once()
    assert worked is True

    # Verify job completed and workflow advanced to SCRIPT stage
    async with session_factory() as session:
        job_repo = JobRepository(session)
        wf_repo = WorkflowRepository(session)

        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        job_summary = next(j for j in jobs if j.logical_key == "research")
        assert job_summary.status == JobStatus.COMPLETED.value
        assert job_summary.output_payload == {"result": "ok"}

        # Attempt should have provider operation ID recorded
        full_job = await job_repo.get_job(job_summary.id, load_attempts=True)
        assert full_job is not None
        assert len(full_job.attempts) == 1
        assert full_job.attempts[0].provider_operation_id == "test-op-12345"

        updated_wf = await wf_repo.get_workflow(wf.id)
        assert updated_wf is not None
        assert updated_wf.current_stage == JobStage.SCRIPT.value


@pytest.mark.asyncio
async def test_worker_missing_handler_fails_safely(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)
    worker = JobWorker(session_factory=session_factory)  # No handlers registered

    wf = await orchestrator.create_workflow(topic="Unregistered Job Test")
    worked = await worker.run_once()
    assert worked is True

    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        assert jobs[0].status == JobStatus.FAILED.value
        assert jobs[0].error_code == "NO_HANDLER_REGISTERED"


@pytest.mark.asyncio
async def test_full_pipeline_stage_progression(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Architectural Directive Verification:
    Workflow progresses through:
    RESEARCH -> SCRIPT -> STORYBOARD -> parallel VIDEO_GENERATION -> MEDIA_ASSEMBLY -> COMPLETED.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)
    worker = JobWorker(session_factory=session_factory)

    # Register handlers for all pipeline stages
    worker.register_handler("research", DummyHandler(output={"summary": "Key research points"}))
    worker.register_handler("script", DummyHandler(output={"narration": "Script narration"}))
    worker.register_handler(
        "storyboard",
        DummyHandler(
            output={
                "aspect_ratio": "9:16",
                "shots": [
                    {"shot_number": 1, "video_prompt": "Prompt 1"},
                    {"shot_number": 2, "video_prompt": "Prompt 2"},
                ],
            }
        ),
    )
    worker.register_handler(
        "video_generation", DummyHandler(output={"video_path": "/tmp/shot.mp4"})
    )
    worker.register_handler(
        "media_assembly", DummyHandler(output={"final_video_path": "/tmp/final.mp4"})
    )

    wf = await orchestrator.create_workflow(topic="Autonomous Video Orchestration")

    # Step 1: Execute RESEARCH
    assert await worker.run_once() is True
    wf_state = await orchestrator.get_workflow(wf.id)
    assert wf_state.current_stage == JobStage.SCRIPT

    # Step 2: Execute SCRIPT
    assert await worker.run_once() is True
    wf_state = await orchestrator.get_workflow(wf.id)
    assert wf_state.current_stage == JobStage.STORYBOARD

    # Step 3: Execute STORYBOARD (spawns 2 video shot jobs)
    assert await worker.run_once() is True
    wf_state = await orchestrator.get_workflow(wf.id)
    assert wf_state.current_stage == JobStage.VIDEO_GENERATION

    # Step 4: Execute Shot 1
    assert await worker.run_once() is True

    # Step 5: Execute Shot 2
    assert await worker.run_once() is True
    wf_state = await orchestrator.get_workflow(wf.id)
    assert wf_state.current_stage == JobStage.MEDIA_ASSEMBLY

    # Step 6: Execute MEDIA_ASSEMBLY
    assert await worker.run_once() is True
    wf_state = await orchestrator.get_workflow(wf.id)
    assert wf_state.status == WorkflowStatus.COMPLETED
    assert wf_state.current_stage == JobStage.COMPLETED


@pytest.mark.asyncio
async def test_worker_start_stop_lifecycle(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    worker = JobWorker(session_factory=session_factory)
    await worker.start(poll_interval_seconds=0.05)
    await asyncio.sleep(0.1)
    await worker.stop()
    assert worker._running is False
