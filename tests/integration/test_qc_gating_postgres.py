"""Integration tests for QC-to-PUBLISH gating and invariants on PostgreSQL 16."""

import asyncio
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration import (
    InvariantViolationError,
    JobStage,
    JobStatus,
    JobWorker,
    WorkflowOrchestrator,
    WorkflowStatus,
)
from app.orchestration.state_machine import QcVerdict
from app.repositories import JobRepository, WorkflowRepository


class DeterministicHandler:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output

    async def execute(
        self, job: JobModel, attempt: JobAttemptModel, worker: JobWorker
    ) -> dict[str, Any]:
        return self.output


@pytest.mark.asyncio
async def test_qc_gating_direct_repository_enforcement(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify that JobRepository.create_job blocks PUBLISH job unless qc_verdict is PASS."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)

        # 1. Workflow with no QC verdict (None)
        wf = await wf_repo.create_workflow(topic="QC Gating Direct Repo Test")
        await session.commit()

        with pytest.raises(InvariantViolationError) as exc_info:
            await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="publish",
                job_type="publish",
                stage=JobStage.PUBLISH.value,
                input_payload={"topic": wf.topic},
            )
        assert "QC verdict is 'None', must be 'PASS'" in str(exc_info.value)

        # 2. Workflow with QC verdict FAIL
        await wf_repo.update_status(
            wf.id,
            status=WorkflowStatus.FAILED.value,
            qc_verdict=QcVerdict.FAIL.value,
        )
        await session.commit()

        with pytest.raises(InvariantViolationError) as exc_info:
            await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="publish",
                job_type="publish",
                stage=JobStage.PUBLISH.value,
                input_payload={"topic": wf.topic},
            )
        assert "QC verdict is 'FAIL', must be 'PASS'" in str(exc_info.value)

        # 3. Workflow with QC verdict PASS
        # Re-create fresh workflow with PASS verdict
        wf_pass = await wf_repo.create_workflow(topic="QC Gating Pass Workflow")
        await wf_repo.update_status(
            wf_pass.id,
            status=WorkflowStatus.RUNNING.value,
            qc_verdict=QcVerdict.PASS.value,
        )
        await session.commit()

        publish_job = await job_repo.create_job(
            workflow_id=wf_pass.id,
            logical_key="publish",
            job_type="publish",
            stage=JobStage.PUBLISH.value,
            input_payload={"topic": wf_pass.topic},
        )
        assert publish_job is not None
        assert publish_job.stage == JobStage.PUBLISH.value
        await session.commit()


@pytest.mark.asyncio
async def test_orchestrator_qc_fail_blocks_publish_and_fails_workflow(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify that when a QC job finishes with FAIL verdict, PUBLISH is never scheduled."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    worker = JobWorker(session_factory=session_factory)

    # Register QC handler returning FAIL verdict
    worker.register_handler(
        "qc",
        DeterministicHandler(
            {
                "verdict": "FAIL",
                "failure_reasons": [
                    "Loudness exceeded tolerance (-12 LUFS)",
                    "Frozen frames detected",
                ],
            }
        ),
    )

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)

        wf = await wf_repo.create_workflow(topic="QC Rejection Test")
        await wf_repo.update_status(
            wf.id,
            status=WorkflowStatus.RUNNING.value,
            current_stage=JobStage.QC.value,
        )
        # Create QC job
        await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="qc",
            job_type="qc",
            stage=JobStage.QC.value,
            input_payload={"test": True},
        )
        await session.commit()

    # Run worker to process QC job
    worked = await worker.run_once()
    assert worked is True

    # Verify workflow state
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)

        updated_wf = await wf_repo.get_workflow(wf.id)
        assert updated_wf is not None
        assert updated_wf.status == WorkflowStatus.FAILED.value
        assert updated_wf.qc_verdict == "FAIL"
        assert updated_wf.error_code == "QC_VERDICT_FAILED"
        assert "Loudness exceeded tolerance" in (updated_wf.error_message or "")

        # Verify PUBLISH job was NOT created
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        job_keys = {j.logical_key for j in jobs}
        assert "publish" not in job_keys


@pytest.mark.asyncio
async def test_orchestrator_qc_pass_schedules_publish(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify that when a QC job finishes with PASS verdict, workflow advances to PUBLISH."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    worker = JobWorker(session_factory=session_factory)

    # Register QC handler returning PASS verdict
    worker.register_handler(
        "qc",
        DeterministicHandler(
            {
                "verdict": "PASS",
                "metrics": {"loudness_lufs": -16.0, "duration": 30.0},
            }
        ),
    )
    # Register Publish handler
    worker.register_handler(
        "publish",
        DeterministicHandler(
            {
                "public_url": "https://storage.googleapis.com/videos/final.mp4",
                "status": "published",
            }
        ),
    )

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)

        wf = await wf_repo.create_workflow(topic="QC Pass Pipeline Test")
        await wf_repo.update_status(
            wf.id,
            status=WorkflowStatus.RUNNING.value,
            current_stage=JobStage.QC.value,
        )
        await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="qc",
            job_type="qc",
            stage=JobStage.QC.value,
            input_payload={"test": True},
        )
        await session.commit()

    # 1. Run worker to process QC job
    worked = await worker.run_once()
    assert worked is True

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)

        updated_wf = await wf_repo.get_workflow(wf.id)
        assert updated_wf is not None
        assert updated_wf.status == WorkflowStatus.RUNNING.value
        assert updated_wf.current_stage == JobStage.PUBLISH.value
        assert updated_wf.qc_verdict == "PASS"

        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        publish_job = next((j for j in jobs if j.logical_key == "publish"), None)
        assert publish_job is not None
        assert publish_job.stage == JobStage.PUBLISH.value
        assert publish_job.status == JobStatus.PENDING.value

    # 2. Run worker to process Publish job
    worked2 = await worker.run_once()
    assert worked2 is True

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        final_wf = await wf_repo.get_workflow(wf.id)
        assert final_wf is not None
        assert final_wf.status == WorkflowStatus.COMPLETED.value
        assert final_wf.current_stage == JobStage.COMPLETED.value


@pytest.mark.asyncio
async def test_concurrent_advancement_and_qc_gating_safety(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify concurrent advance_workflow calls on a QC job do not duplicate publish jobs or deadlock."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)

        wf = await wf_repo.create_workflow(topic="Concurrent QC Test")
        await wf_repo.update_status(
            wf.id,
            status=WorkflowStatus.RUNNING.value,
            current_stage=JobStage.QC.value,
        )
        # Create QC job and complete it
        qc_job = await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="qc",
            job_type="qc",
            stage=JobStage.QC.value,
            input_payload={},
        )
        claim = await job_repo.claim_next_job(worker_id="worker-concurrent-qc")
        assert claim is not None
        _, att = claim
        await job_repo.complete_job(
            job_id=qc_job.id,
            worker_id="worker-concurrent-qc",
            lease_token=att.lease_token,
            output_payload={"verdict": "PASS"},
        )
        await session.commit()

    # Concurrently advance workflow 5 times
    tasks = [orchestrator.advance_workflow(wf.id) for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        assert not isinstance(r, Exception), f"Concurrent advance failed with error: {r}"

    # Verify only exactly 1 publish job was created
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        publish_jobs = [j for j in jobs if j.logical_key == "publish"]
        assert len(publish_jobs) == 1
