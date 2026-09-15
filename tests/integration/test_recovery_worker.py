"""Integration tests for RecoveryWorker against PostgreSQL 16."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.orchestration import AttemptStatus, JobStage, JobStatus
from app.orchestration.recovery import RecoveryWorker
from app.repositories import JobRepository, WorkflowRepository


class MockVeoReconciler:
    """Mock reconciler simulating external Veo operation checking."""

    def __init__(self, mode: str = "found") -> None:
        self.mode = mode

    async def reconcile_submission(
        self,
        submission_token: str | None,
        provider_operation_id: str | None,
    ) -> str | None:
        if self.mode == "found":
            return "operations/reconciled-veo-999"
        elif self.mode == "not_found":
            return None
        raise RuntimeError("External API timeout during reconciliation")


@pytest.mark.asyncio
async def test_recovery_worker_reclaims_crashed_worker_lease(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Recovery Normal Crash")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="script",
        job_type="script",
        stage=JobStage.SCRIPT.value,
        input_payload={"topic": "Recovery"},
        max_attempts=2,
    )
    job_id = job.id
    await db_session.commit()

    # Claim job
    claim = await job_repo.claim_next_job(worker_id="crashed-worker-1")
    assert claim is not None
    _, attempt = claim

    # Backdate heartbeat
    attempt.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    await db_session.commit()

    # Run recovery
    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    report = await worker.run_once()

    assert report.expired_attempts_detected == 1
    assert report.jobs_rescheduled == 1
    assert report.jobs_terminally_failed == 0

    # Expire all cached objects in db_session to pull latest committed rows from PostgreSQL
    db_session.expire_all()
    updated_job = await job_repo.get_job(job_id, load_attempts=True)
    assert updated_job is not None
    assert updated_job.status == JobStatus.PENDING.value
    assert updated_job.attempts[0].status == AttemptStatus.EXPIRED.value


@pytest.mark.asyncio
async def test_recovery_worker_reconciles_submission_pending_found(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Veo Reconcile Found")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="video:shot:1",
        job_type="video_generation",
        stage=JobStage.VIDEO_GENERATION.value,
        input_payload={"prompt": "prompt"},
        max_attempts=3,
    )
    job_id = job.id
    await db_session.commit()

    # Claim job and record submission pending
    claim = await job_repo.claim_next_job(worker_id="veo-worker-1")
    assert claim is not None
    _, attempt = claim

    await job_repo.record_submission_pending(
        job_id=job_id,
        worker_id="veo-worker-1",
        lease_token=attempt.lease_token,
        provider="veo",
    )
    attempt.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    await db_session.commit()

    # Setup recovery with Veo reconciler that finds the operation
    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    worker.register_reconciler("veo", MockVeoReconciler(mode="found"))

    report = await worker.run_once()
    assert report.expired_attempts_detected == 1
    assert report.jobs_rescheduled == 1

    db_session.expire_all()
    updated_job = await job_repo.get_job(job_id, load_attempts=True)
    assert updated_job is not None
    assert updated_job.status == JobStatus.RUNNING.value
    assert updated_job.attempts[0].status == AttemptStatus.RUNNING.value
    assert updated_job.attempts[0].provider_operation_id == "operations/reconciled-veo-999"


@pytest.mark.asyncio
async def test_recovery_worker_handles_ambiguous_submission_pending_safely(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Architectural Directive Verification:
    Ambiguous submissions do NOT blindly generate duplicates; they require reconciliation or fail safely.
    """
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Veo Ambiguous Test")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="video:shot:1",
        job_type="video_generation",
        stage=JobStage.VIDEO_GENERATION.value,
        input_payload={"prompt": "prompt"},
        max_attempts=3,
    )
    job_id = job.id
    await db_session.commit()

    claim = await job_repo.claim_next_job(worker_id="veo-worker-2")
    assert claim is not None
    _, attempt = claim

    await job_repo.record_submission_pending(
        job_id=job_id,
        worker_id="veo-worker-2",
        lease_token=attempt.lease_token,
        provider="veo",
    )
    attempt.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    await db_session.commit()

    # Recovery with ambiguous error
    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    worker.register_reconciler("veo", MockVeoReconciler(mode="error"))

    report = await worker.run_once()
    assert report.expired_attempts_detected == 1
    assert report.ambiguous_reconciliations == 1
    assert report.jobs_terminally_failed == 1

    db_session.expire_all()
    updated_job = await job_repo.get_job(job_id, load_attempts=True)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED.value
    assert updated_job.error_code == "PROVIDER_RECONCILIATION_REQUIRED"


@pytest.mark.asyncio
async def test_recovery_worker_start_stop(pg_engine: AsyncEngine):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=1.0)
    await worker.start(interval_seconds=0.05)
    await asyncio.sleep(0.1)
    await worker.stop()
    assert worker._running is False
