import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration import (
    AttemptStatus,
    JobStage,
    JobStatus,
    ReconciliationOutcome,
    ReconciliationStatus,
    WorkflowStatus,
)
from app.orchestration.errors import LeaseConflictError
from app.orchestration.recovery import RecoveryWorker
from app.orchestration.worker import JobWorker
from app.repositories import JobRepository, WorkflowRepository


class MockVeoReconciler:
    """Mock reconciler simulating external operation checking."""

    def __init__(self, mode: str = "found", operation_id: str | None = None) -> None:
        self.mode = mode
        self.operation_id = operation_id or "operations/reconciled-veo-999"

    async def reconcile_submission(
        self,
        submission_token: str | None,
        provider_operation_id: str | None,
    ) -> ReconciliationOutcome:
        if self.mode == "found":
            return ReconciliationOutcome(
                status=ReconciliationStatus.RESOLVED,
                provider_operation_id=self.operation_id,
                metadata={"status": "processing"},
            )
        elif self.mode == "confirmed_absent":
            return ReconciliationOutcome(
                status=ReconciliationStatus.CONFIRMED_ABSENT,
            )
        elif self.mode == "unresolved":
            return ReconciliationOutcome(
                status=ReconciliationStatus.UNRESOLVED,
                error_message="Ambiguous external submission",
            )
        raise RuntimeError("External API timeout during reconciliation")


@pytest.mark.asyncio
async def test_recovery_worker_reclaims_crashed_worker_lease(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
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
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
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
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
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
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
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
    assert updated_job.status == JobStatus.PENDING.value
    assert updated_job.attempts[0].status == AttemptStatus.EXPIRED.value
    assert updated_job.attempts[0].provider_operation_id == "operations/reconciled-veo-999"
    assert updated_job.available_at <= datetime.now(UTC)


@pytest.mark.asyncio
async def test_recovery_worker_handles_ambiguous_submission_pending_safely(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Architectural Directive Verification:
    Ambiguous submissions do NOT blindly generate duplicates; they require reconciliation or fail safely.
    Zero-retry terminal ambiguity must remain persistent across repeated recovery invocations.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
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
    wf_id = wf.id
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
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
    await db_session.commit()

    # Recovery invocation #1: ambiguous error yields terminal failure
    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    worker.register_reconciler("veo", MockVeoReconciler(mode="unresolved"))

    report1 = await worker.run_once()
    assert report1.expired_attempts_detected == 1
    assert report1.ambiguous_reconciliations == 1
    assert report1.jobs_terminally_failed == 1

    db_session.expire_all()
    updated_job = await job_repo.get_job(job_id, load_attempts=True)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED.value
    assert updated_job.attempts[0].provider_operation_id is None
    assert updated_job.error_code == "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"
    assert len(updated_job.attempts) == 1
    assert updated_job.attempts[0].status == AttemptStatus.FAILED.value

    updated_wf = await wf_repo.get_workflow(wf_id)
    assert updated_wf is not None
    assert updated_wf.status == WorkflowStatus.FAILED.value
    assert updated_wf.error_code == "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"

    # Recovery invocation #2: persistent zero-retry invariant verification
    report2 = await worker.run_once()
    assert report2.expired_attempts_detected == 0
    assert report2.jobs_rescheduled == 0

    db_session.expire_all()
    job_after_second_run = await job_repo.get_job(job_id, load_attempts=True)
    assert job_after_second_run is not None
    assert job_after_second_run.status == JobStatus.FAILED.value
    assert len(job_after_second_run.attempts) == 1
    assert job_after_second_run.attempts[0].status == AttemptStatus.FAILED.value


@pytest.mark.asyncio
async def test_recovery_worker_handles_confirmed_absent_reschedules(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Confirmed Absent Test")
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

    claim = await job_repo.claim_next_job(worker_id="absent-worker")
    assert claim is not None
    _, attempt = claim

    await job_repo.record_submission_pending(
        job_id=job_id,
        worker_id="absent-worker",
        lease_token=attempt.lease_token,
        provider="mock_absent_provider",
    )
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
    await db_session.commit()

    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    worker.register_reconciler("mock_absent_provider", MockVeoReconciler(mode="confirmed_absent"))

    report = await worker.run_once()
    assert report.expired_attempts_detected == 1
    assert report.jobs_rescheduled == 1

    db_session.expire_all()
    updated_job = await job_repo.get_job(job_id, load_attempts=True)
    assert updated_job is not None
    assert updated_job.status == JobStatus.PENDING.value
    assert updated_job.available_at is not None
    assert updated_job.attempts[0].status == AttemptStatus.EXPIRED.value


@pytest.mark.asyncio
async def test_recovery_worker_start_stop(pg_engine: AsyncEngine):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=1.0)
    await worker.start(interval_seconds=0.05)
    await asyncio.sleep(0.1)
    await worker.stop()
    assert worker._running is False


@pytest.mark.asyncio
async def test_resolved_recovery_executes_to_completion(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Architectural Directive Verification:
    Full recovery lifecycle proof:
    Worker A crashes in SUBMISSION_PENDING
    -> Recovery worker reconciles RESOLVED
    -> Job becomes PENDING with available_at <= now
    -> Worker B claims job
    -> Worker B resumes persisted provider operation (zero duplicate submissions)
    -> Provider operation completes
    -> Job marked COMPLETED
    -> Workflow advances
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Full Recovery Lifecycle")
    wf_id = wf.id
    job = await job_repo.create_job(
        workflow_id=wf_id,
        logical_key="research",
        job_type="research",
        stage=JobStage.RESEARCH.value,
        input_payload={"topic": "Full Recovery Lifecycle"},
        max_attempts=3,
    )
    job_id = job.id
    await db_session.commit()

    # 1. Worker A claims and enters SUBMISSION_PENDING
    claim_a = await job_repo.claim_next_job(worker_id="worker-A")
    assert claim_a is not None
    _, attempt_a = claim_a

    await job_repo.record_submission_pending(
        job_id=job_id,
        worker_id="worker-A",
        lease_token=attempt_a.lease_token,
        provider="veo",
    )
    # Simulate Worker A crash by backdating heartbeat
    attempt_a.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
    await db_session.commit()

    # 2. RecoveryWorker runs: finds operation 'operations/reconciled-e2e-123'
    recovery_worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    recovery_worker.register_reconciler(
        "veo", MockVeoReconciler(mode="found", operation_id="operations/reconciled-e2e-123")
    )
    report = await recovery_worker.run_once()
    assert report.expired_attempts_detected == 1
    assert report.jobs_rescheduled == 1

    # Verify attempt is EXPIRED with operation ID and job is PENDING
    db_session.expire_all()
    rescheduled_job = await job_repo.get_job(job_id, load_attempts=True)
    assert rescheduled_job is not None
    assert rescheduled_job.status == JobStatus.PENDING.value
    assert rescheduled_job.attempts[0].status == AttemptStatus.EXPIRED.value
    assert rescheduled_job.attempts[0].provider_operation_id == "operations/reconciled-e2e-123"

    # 3. Worker B claims the rescheduled job
    provider_submissions = 0
    poll_count = 0

    class RecoverableHandler:
        async def execute(self, j: JobModel, a: JobAttemptModel, w: JobWorker) -> dict[str, Any]:
            nonlocal provider_submissions, poll_count
            recovered_op = await w.get_recoverable_provider_operation_id(j.id)
            if recovered_op is None:
                provider_submissions += 1
                return {"status": "fresh_submission"}
            else:
                poll_count += 1
                return {
                    "operation_id": recovered_op,
                    "status": "completed",
                    "brief": {"topic": "Full Recovery Lifecycle", "sections": []},
                }

    worker_b = JobWorker(session_factory=session_factory, worker_id="worker-B")
    worker_b.register_handler("research", RecoverableHandler())

    processed = await worker_b.run_once()
    assert processed is True
    assert provider_submissions == 0  # ZERO duplicate submission!
    assert poll_count == 1

    # Verify final database state: job completed and workflow advanced
    db_session.expire_all()
    completed_job = await job_repo.get_job(job_id, load_attempts=True)
    assert completed_job is not None
    assert completed_job.status == JobStatus.COMPLETED.value
    assert len(completed_job.attempts) == 2
    assert completed_job.attempts[1].status == AttemptStatus.COMPLETED.value

    # Verify workflow advanced to next stage (SCRIPT)
    updated_wf = await wf_repo.get_workflow(wf_id)
    assert updated_wf is not None
    assert updated_wf.current_stage == JobStage.SCRIPT.value


@pytest.mark.asyncio
async def test_concurrent_recovery_workers_claim_safely(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify that two recovery workers concurrently scanning the same expired attempt
    safely serialize via FOR UPDATE SKIP LOCKED and recovery ownership fencing.
    Exactly one recovery worker claims and transitions the attempt; zero conflicts,
    zero duplicate attempts or reschedules.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Concurrent Recovery Contention")
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

    claim = await job_repo.claim_next_job(worker_id="initial-worker")
    assert claim is not None
    _, attempt = claim

    await job_repo.record_submission_pending(
        job_id=job_id,
        worker_id="initial-worker",
        lease_token=attempt.lease_token,
        provider="veo",
    )
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
    await db_session.commit()

    # Create two recovery workers with distinct worker_ids
    recovery_1 = RecoveryWorker(
        session_factory=session_factory,
        lease_timeout_seconds=30.0,
        worker_id="recovery-worker-1",
    )
    recovery_1.register_reconciler("veo", MockVeoReconciler(mode="found"))

    recovery_2 = RecoveryWorker(
        session_factory=session_factory,
        lease_timeout_seconds=30.0,
        worker_id="recovery-worker-2",
    )
    recovery_2.register_reconciler("veo", MockVeoReconciler(mode="found"))

    # Execute both concurrently
    report_1, report_2 = await asyncio.gather(
        recovery_1.run_once(),
        recovery_2.run_once(),
    )

    total_rescheduled = report_1.jobs_rescheduled + report_2.jobs_rescheduled
    assert total_rescheduled == 1, (
        f"Expected exactly 1 job rescheduled across workers, got {total_rescheduled} "
        f"(Worker 1: {report_1.jobs_rescheduled}, Worker 2: {report_2.jobs_rescheduled})"
    )

    db_session.expire_all()
    final_job = await job_repo.get_job(job_id, load_attempts=True)
    assert final_job is not None
    assert final_job.status == JobStatus.PENDING.value
    assert len(final_job.attempts) == 1
    assert final_job.attempts[0].status == AttemptStatus.EXPIRED.value


@pytest.mark.asyncio
async def test_heartbeat_and_recovery_race_protection(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify heartbeat / recovery race protection:
    Case A: A worker with a fresh heartbeat is not expired by the recovery worker.
    Case B: Once genuinely expired, recovery claims the attempt and fences the original worker,
            causing subsequent mutations with the old lease token to fail with LeaseConflictError.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Heartbeat Race Protection")
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

    # --- Case A: Fresh heartbeat wins ---
    claim = await job_repo.claim_next_job(worker_id="active-worker")
    assert claim is not None
    _, attempt = claim
    original_lease_token = attempt.lease_token

    # Heartbeat is fresh (0 seconds old)
    recovery_worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    report = await recovery_worker.run_once()
    assert report.expired_attempts_detected == 0
    assert report.jobs_rescheduled == 0

    db_session.expire_all()
    job_check = await job_repo.get_job(job_id, load_attempts=True)
    assert job_check is not None
    assert job_check.attempts[0].status == AttemptStatus.CLAIMED.value
    assert job_check.attempts[0].worker_id == "active-worker"

    # --- Case B: Recovery wins after genuine expiry, fencing stale worker ---
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
    await db_session.commit()

    report_expired = await recovery_worker.run_once()
    assert report_expired.expired_attempts_detected == 1
    assert report_expired.jobs_rescheduled == 1

    # Active worker now attempts heartbeat with its stale lease token
    async with session_factory() as stale_session:
        stale_repo = JobRepository(stale_session)
        with pytest.raises(LeaseConflictError):
            await stale_repo.heartbeat_attempt(
                job_id=job_id,
                worker_id="active-worker",
                lease_token=original_lease_token,
            )
