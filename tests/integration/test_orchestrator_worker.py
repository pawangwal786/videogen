import asyncio
import uuid
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.base import utc_now
from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration import (
    AttemptStatus,
    JobStage,
    JobStatus,
    JobWorker,
    LeaseConflictError,
    RecoveryWorker,
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


@pytest.mark.asyncio
async def test_advance_workflow_concurrent_idempotency(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)

    wf = await orchestrator.create_workflow(topic="Parallel Advance Test")
    wf_id = wf.id

    # Complete the research job specifically for this workflow
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf_id)
        r_job = next(j for j in jobs if j.logical_key == "research")
        now = utc_now()
        token = str(uuid.uuid4())
        attempt = JobAttemptModel(
            job_id=r_job.id,
            attempt_number=1,
            worker_id="test-worker",
            lease_token=token,
            status=AttemptStatus.CLAIMED.value,
            started_at=now,
            heartbeat_at=now,
            submission_token=f"{r_job.workflow_id}:{r_job.logical_key}:1",
            request_payload=r_job.input_payload,
        )
        session.add(attempt)
        await session.flush()
        r_job.status = JobStatus.CLAIMED.value
        r_job.current_attempt_id = attempt.id
        await session.flush()
        await job_repo.complete_job(r_job.id, "test-worker", token, {"findings": "verified"})
        await session.commit()

    # Advance workflow concurrently from 10 independent callers
    async def advance_task():
        caller_orch = WorkflowOrchestrator(session_factory)
        return await caller_orch.advance_workflow(wf_id)

    results = await asyncio.gather(*[advance_task() for _ in range(10)])

    # All callers receive coherent workflow state
    for res in results:
        assert res.status == WorkflowStatus.RUNNING
        assert res.current_stage == JobStage.SCRIPT

    # Verify exactly one script job was created in the database
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf_id)
        script_jobs = [j for j in jobs if j.logical_key == "script"]
        assert len(script_jobs) == 1


@pytest.mark.asyncio
async def test_worker_restart_resumes_persisted_provider_operation(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Prove that an in-flight provider operation survives a worker crash,
    and a subsequent worker reclaims the job, sees the persisted operation ID,
    reconciles/polls it, and advances the workflow without duplicate submission.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create workflow and video generation job directly
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)
        wf = await wf_repo.create_workflow(topic="Crash Recovery Video Shot")
        wf_id = wf.id
        job = await job_repo.create_job(
            workflow_id=wf_id,
            logical_key="video:shot:1",
            job_type="video_generation",
            stage=JobStage.VIDEO_GENERATION.value,
            input_payload={"prompt": "A futuristic city in neon rain"},
            max_attempts=3,
        )
        job_id = job.id
        await session.commit()

    provider_submission_count = 0
    poll_count = 0
    crash_sync_event = asyncio.Event()

    class ResumableVideoHandler:
        async def execute(
            self, job: JobModel, attempt: JobAttemptModel, worker: JobWorker
        ) -> dict[str, Any]:
            nonlocal provider_submission_count, poll_count

            existing_op_id = await worker.get_recoverable_provider_operation_id(job.id)

            if existing_op_id is None:
                # First attempt: submit to provider
                await worker.record_submission_pending("veo", job.input_payload)
                provider_submission_count += 1
                op_id = "operations/veo-real-4567"
                await worker.record_submitted(op_id)
                crash_sync_event.set()
                # Simulate abrupt crash/interdiction right after committing operation ID to DB
                raise SystemExit("Simulated Worker A crash/SIGKILL")
            else:
                # Resumed attempt: do NOT re-submit to provider! Poll existing operation
                poll_count += 1
                return {
                    "operation_id": existing_op_id,
                    "video_path": "tmp/staging/shot_1.mp4",
                    "status": "completed",
                }

    # Worker A executes and crashes right after recording submitted
    worker_a = JobWorker(session_factory=session_factory, worker_id="worker-A")
    worker_a.register_handler("video_generation", ResumableVideoHandler())

    with pytest.raises(SystemExit):
        await worker_a.run_once()

    assert crash_sync_event.is_set()
    assert provider_submission_count == 1

    # Verify that operation ID survived Worker A's crash in PostgreSQL
    async with session_factory() as session:
        job_repo = JobRepository(session)
        job_with_att = await job_repo.get_job(job_id, load_attempts=True)
        assert job_with_att is not None and len(job_with_att.attempts) > 0
        assert job_with_att.attempts[0].provider_operation_id == "operations/veo-real-4567"

        # Backdate heartbeat of Worker A's attempt to simulate lease expiration
        attempt_a = job_with_att.attempts[0]
        attempt_a.heartbeat_at = utc_now() - timedelta(seconds=120)
        await session.commit()

    # Recovery worker runs: detects expired lease and reschedules job for retry
    recovery_worker = RecoveryWorker(session_factory=session_factory, lease_timeout_seconds=30.0)
    report = await recovery_worker.run_once()
    assert report.expired_attempts_detected >= 1
    assert report.jobs_rescheduled >= 1

    # Clear retry backoff for immediate test assertion
    async with session_factory() as session:
        job_repo = JobRepository(session)
        rescheduled_job = await job_repo.get_job(job_id)
        assert rescheduled_job is not None
        rescheduled_job.available_at = utc_now()
        await session.commit()

    # Worker B starts, claims the rescheduled job, sees the persisted operation ID, and finishes it
    worker_b = JobWorker(session_factory=session_factory, worker_id="worker-B")
    worker_b.register_handler("video_generation", ResumableVideoHandler())

    processed = await worker_b.run_once()
    assert processed is True
    assert provider_submission_count == 1  # ZERO duplicate submission!
    assert poll_count == 1

    # Verify database state
    async with session_factory() as session:
        job_repo = JobRepository(session)
        recovered_job = await job_repo.get_job(job_id, load_attempts=True)
        assert recovered_job is not None
        assert recovered_job.status == JobStatus.COMPLETED.value
        assert recovered_job.output_payload["operation_id"] == "operations/veo-real-4567"
        assert len(recovered_job.attempts) == 2
        assert recovered_job.attempts[0].status == AttemptStatus.EXPIRED.value
        assert recovered_job.attempts[1].status == AttemptStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_worker_shutdown_preserves_attempt_and_blocks_stale_mutation(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Prove that when Worker A's lease expires and Worker B reclaims the job,
    any stale mutations from Worker A are rejected with LeaseConflictError at DB level,
    and Worker B's active ownership is preserved untouched.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Fencing Test")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="script",
        job_type="script",
        stage=JobStage.SCRIPT.value,
        input_payload={"topic": "Fencing"},
        max_attempts=3,
    )
    job_id = job.id
    await db_session.commit()

    # 1. Worker A claims the job
    worker_id_a = "worker-A"
    claim_a = await job_repo.claim_next_job(worker_id=worker_id_a)
    assert claim_a is not None
    job_a, attempt_a = claim_a
    lease_token_a = attempt_a.lease_token
    await db_session.commit()

    # 2. Worker A's lease expires (marked EXPIRED in DB)
    attempt_a.status = AttemptStatus.EXPIRED.value
    attempt_a.completed_at = utc_now()
    job_a.status = JobStatus.PENDING.value
    job_a.version += 1
    await db_session.commit()

    # 3. Worker B claims the job
    worker_id_b = "worker-B"
    claim_b = await job_repo.claim_next_job(worker_id=worker_id_b)
    assert claim_b is not None
    _job_b, attempt_b = claim_b
    lease_token_b = attempt_b.lease_token
    await db_session.commit()

    assert lease_token_a != lease_token_b
    assert attempt_b.worker_id == worker_id_b
    assert attempt_b.status == AttemptStatus.CLAIMED.value

    # 4. Stale Worker A attempts mutation with its old lease_token_a
    async with session_factory() as stale_session:
        stale_repo = JobRepository(stale_session)

        # Heartbeat attempt with stale token -> rejected
        with pytest.raises(LeaseConflictError):
            await stale_repo.heartbeat_attempt(job_id, worker_id_a, lease_token_a)

        # Record submission pending with stale token -> rejected
        with pytest.raises(LeaseConflictError):
            await stale_repo.record_submission_pending(job_id, worker_id_a, lease_token_a, "veo")

        # Record submitted with stale token -> rejected
        with pytest.raises(LeaseConflictError):
            await stale_repo.record_submitted(job_id, worker_id_a, lease_token_a, "op-stale")

        # Complete job with stale token -> rejected
        with pytest.raises(LeaseConflictError):
            await stale_repo.complete_job(job_id, worker_id_a, lease_token_a, {"output": "stale"})

    # 5. Assert Worker B's state in PostgreSQL was NOT modified
    db_session.expire_all()
    current_job = await job_repo.get_job(job_id, load_attempts=True)
    assert current_job is not None
    assert current_job.current_attempt_id == attempt_b.id
    assert current_job.status == JobStatus.CLAIMED.value
    assert current_job.output_payload is None

    active_b = await job_repo.get_active_attempt(job_id, worker_id_b, lease_token_b)
    assert active_b.id == attempt_b.id
    assert active_b.worker_id == worker_id_b
    assert active_b.lease_token == lease_token_b
    assert active_b.status == AttemptStatus.CLAIMED.value


@pytest.mark.asyncio
async def test_failed_attempt_does_not_reuse_provider_operation(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify that a provider operation belonging to a FAILED attempt is NEVER reused on retry.
    Attempt 1 submits op-old and fails terminally for that attempt (status=FAILED).
    Attempt 2 claims the retried job:
    - worker.get_recoverable_provider_operation_id() returns None
    - op-old poll count is 0 after new attempt begins
    - provider receives a brand new submission (op-new).
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Failed Attempt Reuse Guard")
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

    created_operations: list[str] = []
    polled_operations: list[str] = []

    class MockFailingThenSucceedingHandler:
        async def execute(self, j: JobModel, a: JobAttemptModel, w: JobWorker) -> dict[str, Any]:
            recoverable_op = await w.get_recoverable_provider_operation_id(j.id)
            if a.attempt_number == 1:
                # First attempt: submits op-old, then fails with retryable error
                assert recoverable_op is None
                await w.record_submission_pending("veo", j.input_payload)
                op_old = "operations/veo-old-fail-123"
                await w.record_submitted(op_old)
                created_operations.append(op_old)

                class ProviderTransientError(Exception):
                    retryable = True
                    code = "PROVIDER_503"

                raise ProviderTransientError("Temporary provider 503")
            else:
                # Second attempt: must NOT discover op-old!
                assert recoverable_op is None
                await w.record_submission_pending("veo", j.input_payload)
                op_new = "operations/veo-new-success-456"
                await w.record_submitted(op_new)
                created_operations.append(op_new)
                polled_operations.append(op_new)
                return {
                    "operation_id": op_new,
                    "video_path": "tmp/staging/shot_1.mp4",
                    "status": "completed",
                }

    # Run Worker 1 for Attempt 1
    worker_1 = JobWorker(session_factory=session_factory, worker_id="worker-1")
    worker_1.register_handler("video_generation", MockFailingThenSucceedingHandler())

    processed_1 = await worker_1.run_once()
    assert processed_1 is True

    # Clear retry backoff for immediate test assertion
    async with session_factory() as session:
        j_repo = JobRepository(session)
        j = await j_repo.get_job(job_id)
        assert j is not None
        assert j.status == JobStatus.PENDING.value
        j.available_at = utc_now()
        await session.commit()

    # Run Worker 2 for Attempt 2
    worker_2 = JobWorker(session_factory=session_factory, worker_id="worker-2")
    worker_2.register_handler("video_generation", MockFailingThenSucceedingHandler())

    processed_2 = await worker_2.run_once()
    assert processed_2 is True

    # Assertions:
    # 1. op-old was never polled after attempt 1 failed!
    assert "operations/veo-old-fail-123" not in polled_operations
    # 2. Both operations were created (1 old, 1 new fresh submission)
    assert created_operations == [
        "operations/veo-old-fail-123",
        "operations/veo-new-success-456",
    ]
    assert polled_operations == ["operations/veo-new-success-456"]

    # 3. Final database state: Attempt 1 is FAILED, Attempt 2 is COMPLETED
    async with session_factory() as session:
        j_repo = JobRepository(session)
        final_j = await j_repo.get_job(job_id, load_attempts=True)
        assert final_j is not None
        assert final_j.status == JobStatus.COMPLETED.value
        assert len(final_j.attempts) == 2
        assert final_j.attempts[0].status == AttemptStatus.FAILED.value
        assert final_j.attempts[0].provider_operation_id == "operations/veo-old-fail-123"
        assert final_j.attempts[1].status == AttemptStatus.COMPLETED.value
        assert final_j.attempts[1].provider_operation_id == "operations/veo-new-success-456"


@pytest.mark.asyncio
async def test_workflow_cancellation_fences_active_worker_mutations(
    pg_engine: AsyncEngine,
    db_session: AsyncSession,
):
    """Verify that cancelling a workflow cancels active job attempts and fences subsequent worker mutations."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    orchestrator = WorkflowOrchestrator(session_factory)

    # 1. Create a workflow with an initial pending job
    wf = await orchestrator.create_workflow(topic="Test Cancellation Fencing")
    wf_id = wf.id

    # 2. Worker claims the job
    worker_id = "test-worker-cancellation"
    token = str(uuid.uuid4())
    async with session_factory() as session:
        job_repo = JobRepository(session)
        claim = await job_repo.claim_next_job(worker_id=worker_id, lease_token=token)
        assert claim is not None
        job, attempt = claim
        job_id = job.id
        attempt_id = attempt.id
        await session.commit()

    # Verify job and attempt are initially active
    async with session_factory() as session:
        j = await session.get(JobModel, job_id)
        assert j is not None
        assert j.status == JobStatus.CLAIMED.value
        att = await session.get(JobAttemptModel, attempt_id)
        assert att is not None
        assert att.status == AttemptStatus.CLAIMED.value

    # 3. Cancel the workflow via orchestrator
    cancelled_wf = await orchestrator.cancel_workflow(wf_id)
    assert cancelled_wf.status == WorkflowStatus.CANCELLED

    # Verify both job and attempt are CANCELLED in DB
    async with session_factory() as session:
        j = await session.get(JobModel, job_id)
        assert j is not None
        assert j.status == JobStatus.CANCELLED.value
        att = await session.get(JobAttemptModel, attempt_id)
        assert att is not None
        assert att.status == AttemptStatus.CANCELLED.value

    # 4. Worker tries heartbeat -> rejected with LeaseConflictError
    async with session_factory() as session:
        job_repo = JobRepository(session)
        with pytest.raises(LeaseConflictError):
            await job_repo.heartbeat_attempt(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=token,
            )

    # 5. Worker tries completion -> rejected with LeaseConflictError
    async with session_factory() as session:
        job_repo = JobRepository(session)
        with pytest.raises(LeaseConflictError):
            await job_repo.complete_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=token,
                output_payload={"status": "done"},
            )

    # 6. Worker tries failure -> rejected with LeaseConflictError
    async with session_factory() as session:
        job_repo = JobRepository(session)
        with pytest.raises(LeaseConflictError):
            await job_repo.fail_job(
                job_id=job_id,
                worker_id=worker_id,
                lease_token=token,
                error_code="WORKER_FAIL",
                error_message="Worker error",
                retryable=False,
            )

    # 7. Verify CANCELLED state remains authoritative in the DB
    async with session_factory() as session:
        j = await session.get(JobModel, job_id)
        assert j is not None
        assert j.status == JobStatus.CANCELLED.value
        att = await session.get(JobAttemptModel, attempt_id)
        assert att is not None
        assert att.status == AttemptStatus.CANCELLED.value
