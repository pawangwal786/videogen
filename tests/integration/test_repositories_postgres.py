"""Integration tests for repositories against real PostgreSQL 16."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.orchestration import (
    ArtifactLifecycleStatus,
    AttemptStatus,
    IdempotencyConflictError,
    InvalidStateTransitionError,
    JobStage,
    JobStatus,
    LeaseConflictError,
    WorkflowStatus,
)
from app.repositories import ArtifactRepository, JobRepository, WorkflowRepository
from tests.integration.conftest import require_postgres


@pytest.mark.asyncio
async def test_require_postgres_contract():
    url = require_postgres()
    assert "postgresql" in url or "postgres" in url


@pytest.mark.asyncio
async def test_workflow_repository_idempotency(db_session: AsyncSession):
    repo = WorkflowRepository(db_session)
    key = f"idemp-{uuid.uuid4()}"

    # 1. First creation
    wf1 = await repo.create_workflow(topic="Quantum Mechanics", idempotency_key=key)
    await db_session.commit()
    assert wf1.id is not None
    assert wf1.topic == "Quantum Mechanics"
    assert wf1.status == WorkflowStatus.PENDING.value

    # 2. Same idempotency key + identical topic -> returns existing
    wf2 = await repo.create_workflow(topic="Quantum Mechanics", idempotency_key=key)
    assert wf2.id == wf1.id

    # 3. Same idempotency key + differing topic -> raises IdempotencyConflictError
    with pytest.raises(IdempotencyConflictError) as exc_info:
        await repo.create_workflow(topic="Different Topic", idempotency_key=key)
    assert exc_info.value.idempotency_key == key


@pytest.mark.asyncio
async def test_workflow_repository_lifecycle(db_session: AsyncSession):
    repo = WorkflowRepository(db_session)
    wf = await repo.create_workflow(topic="Solar Flare Dynamics")
    await db_session.commit()

    # Valid transitions
    updated = await repo.update_status(wf.id, WorkflowStatus.RUNNING.value, current_stage=JobStage.SCRIPT.value)
    await db_session.commit()
    assert updated.status == WorkflowStatus.RUNNING.value
    assert updated.current_stage == JobStage.SCRIPT.value

    completed = await repo.update_status(wf.id, WorkflowStatus.COMPLETED.value)
    await db_session.commit()
    assert completed.status == WorkflowStatus.COMPLETED.value
    assert completed.completed_at is not None

    # Invalid transition from COMPLETED -> RUNNING
    with pytest.raises(InvalidStateTransitionError):
        await repo.update_status(wf.id, WorkflowStatus.RUNNING.value)


@pytest.mark.asyncio
async def test_concurrency_10_workers_competing_for_1_job(pg_engine: AsyncEngine):
    """Architectural Directive Verification:
    10 concurrent workers competing for 1 pending job -> exactly 1 worker claims it, 9 receive None.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Setup 1 workflow and 1 pending job
    workflow_id = str(uuid.uuid4())
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        job_repo = JobRepository(session)
        wf = await wf_repo.create_workflow(topic="Concurrency Race Check")
        workflow_id = wf.id
        await job_repo.create_job(
            workflow_id=workflow_id,
            logical_key="video:shot:1",
            job_type="video_generation",
            stage=JobStage.VIDEO_GENERATION.value,
            input_payload={"prompt": "cinematic neon cityscape"},
            max_attempts=3,
        )
        await session.commit()

    # Worker claim function running in its own session
    async def worker_claim(worker_num: int):
        async with session_factory() as session:
            repo = JobRepository(session)
            claim = await repo.claim_next_job(worker_id=f"worker-{worker_num:02d}")
            if claim is not None:
                await session.commit()
                return claim[0].id, claim[1].worker_id, claim[1].lease_token
            return None

    # Launch 10 workers simultaneously
    results = await asyncio.gather(*(worker_claim(i) for i in range(10)))

    successful_claims = [r for r in results if r is not None]
    failed_claims = [r for r in results if r is None]

    assert len(successful_claims) == 1, f"Expected exactly 1 claim, got {len(successful_claims)}"
    assert len(failed_claims) == 9, f"Expected 9 None claims, got {len(failed_claims)}"


@pytest.mark.asyncio
async def test_job_repository_attempt_lifecycle_and_lease_token(db_session: AsyncSession):
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Attempt Lifecycle Test")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="research",
        job_type="research",
        stage=JobStage.RESEARCH.value,
        input_payload={"query": "Generative Video Architecture"},
    )
    await db_session.commit()

    # Claim job
    claim = await job_repo.claim_next_job(worker_id="worker-alpha")
    assert claim is not None
    claimed_job, attempt = claim
    assert claimed_job.id == job.id
    assert attempt.attempt_number == 1
    assert attempt.worker_id == "worker-alpha"
    assert attempt.lease_token is not None
    assert attempt.submission_token == f"{wf.id}:research:1"
    assert attempt.status == AttemptStatus.CLAIMED.value
    await db_session.commit()

    # Heartbeat with correct worker and lease token
    hb_attempt = await job_repo.heartbeat_attempt(
        job_id=job.id,
        worker_id="worker-alpha",
        lease_token=attempt.lease_token,
    )
    assert hb_attempt.heartbeat_at >= attempt.heartbeat_at

    # Heartbeat with rogue worker or incorrect lease token raises LeaseConflictError
    with pytest.raises(LeaseConflictError):
        await job_repo.heartbeat_attempt(
            job_id=job.id,
            worker_id="worker-rogue",
            lease_token=attempt.lease_token,
        )

    with pytest.raises(LeaseConflictError):
        await job_repo.heartbeat_attempt(
            job_id=job.id,
            worker_id="worker-alpha",
            lease_token=str(uuid.uuid4()),
        )

    # Pre-submission attempt boundary: record_submission_pending
    sub_pending = await job_repo.record_submission_pending(
        job_id=job.id,
        worker_id="worker-alpha",
        lease_token=attempt.lease_token,
        provider="veo",
    )
    assert sub_pending.status == AttemptStatus.SUBMISSION_PENDING.value
    assert sub_pending.provider == "veo"

    # Persist provider operation id
    submitted = await job_repo.record_submitted(
        job_id=job.id,
        worker_id="worker-alpha",
        lease_token=attempt.lease_token,
        provider_operation_id="operations/veo-98765",
        response_metadata={"submitted": True},
    )
    assert submitted.status == AttemptStatus.RUNNING.value
    assert submitted.provider_operation_id == "operations/veo-98765"

    # Complete job
    completed_job, completed_att = await job_repo.complete_job(
        job_id=job.id,
        worker_id="worker-alpha",
        lease_token=attempt.lease_token,
        output_payload={"summary": "Research findings generated."},
    )
    assert completed_job.status == JobStatus.COMPLETED.value
    assert completed_job.output_payload == {"summary": "Research findings generated."}
    assert completed_att.status == AttemptStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_job_retry_and_max_attempts(db_session: AsyncSession):
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Retry Test")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="script",
        job_type="script",
        stage=JobStage.SCRIPT.value,
        input_payload={"topic": "Retry Test"},
        max_attempts=2,
    )
    await db_session.commit()

    # Attempt 1: fail retryable
    claim1 = await job_repo.claim_next_job(worker_id="worker-1")
    assert claim1 is not None
    j1, a1 = claim1
    assert a1.attempt_number == 1

    failed_job, _ = await job_repo.fail_job(
        job_id=job.id,
        worker_id="worker-1",
        lease_token=a1.lease_token,
        error_code="RATE_LIMITED",
        error_message="Provider 429",
        retryable=True,
    )
    assert failed_job.status == JobStatus.PENDING.value
    await db_session.commit()

    # Attempt 2: claim again
    claim2 = await job_repo.claim_next_job(worker_id="worker-2")
    assert claim2 is not None
    j2, a2 = claim2
    assert a2.attempt_number == 2

    # Fail attempt 2 (reaches max_attempts 2) -> marks job terminally FAILED
    terminal_job, term_att = await job_repo.fail_job(
        job_id=job.id,
        worker_id="worker-2",
        lease_token=a2.lease_token,
        error_code="PERMANENT_ERROR",
        error_message="Unrecoverable",
        retryable=True,
    )
    assert terminal_job.status == JobStatus.FAILED.value
    assert term_att.status == AttemptStatus.FAILED.value

    # Subsequent claim should return None
    claim3 = await job_repo.claim_next_job(worker_id="worker-3")
    assert claim3 is None


@pytest.mark.asyncio
async def test_find_expired_leases(db_session: AsyncSession):
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Expired Lease Test")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="storyboard",
        job_type="storyboard",
        stage=JobStage.STORYBOARD.value,
        input_payload={},
    )
    await db_session.commit()

    claim = await job_repo.claim_next_job(worker_id="worker-ghost")
    assert claim is not None
    _, attempt = claim

    # Backdate heartbeat_at by 120 seconds
    attempt.heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    await db_session.commit()

    expired = await job_repo.find_expired_leases(lease_timeout_seconds=60.0)
    assert len(expired) == 1
    assert expired[0].id == attempt.id


@pytest.mark.asyncio
async def test_artifact_repository_operations(db_session: AsyncSession):
    wf_repo = WorkflowRepository(db_session)
    art_repo = ArtifactRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Artifact Spec")
    await db_session.commit()

    # Create artifact
    art = await art_repo.create_artifact(
        workflow_id=wf.id,
        artifact_type="FINAL_VIDEO",
        storage_path="/tmp/output.mp4",
        gdrive_file_id="gdrive-12345",
        file_size_bytes=5242880,
        checksum="sha256-abcdef",
        mime_type="video/mp4",
        artifact_metadata={"fps": 30, "duration": 10.0},
    )
    await db_session.commit()

    assert art.id is not None
    assert art.workflow_id == wf.id
    assert art.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value

    # Query artifact
    fetched = await art_repo.get_artifact(art.id)
    assert fetched is not None
    assert fetched.storage_path == "/tmp/output.mp4"

    # List for workflow
    wf_artifacts = await art_repo.list_artifacts_for_workflow(wf.id)
    assert len(wf_artifacts) == 1
    assert wf_artifacts[0].artifact_type == "FINAL_VIDEO"

    # List for job
    job_artifacts = await art_repo.list_artifacts_for_job(str(uuid.uuid4()))
    assert len(job_artifacts) == 0

    # Update status
    updated = await art_repo.update_lifecycle_status(art.id, ArtifactLifecycleStatus.PURGED.value)
    assert updated is not None
    assert updated.lifecycle_status == ArtifactLifecycleStatus.PURGED.value

    # Update non-existent artifact
    assert await art_repo.update_lifecycle_status("non-existent", ArtifactLifecycleStatus.FAILED.value) is None


@pytest.mark.asyncio
async def test_repository_edge_cases_and_queries(db_session: AsyncSession):
    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    # Workflow not found
    from app.orchestration.errors import JobNotFoundError, WorkflowNotFoundError

    with pytest.raises(WorkflowNotFoundError):
        await wf_repo.update_status(str(uuid.uuid4()), WorkflowStatus.RUNNING.value)

    # Create workflow and verify queries
    wf = await wf_repo.create_workflow(topic="Queries Test")
    await db_session.commit()

    # Load relations
    loaded_wf = await wf_repo.get_workflow(wf.id, load_relations=True)
    assert loaded_wf is not None
    assert len(loaded_wf.jobs) == 0
    assert len(loaded_wf.artifacts) == 0

    # Count workflows
    count_all = await wf_repo.count_workflows()
    assert count_all >= 1
    count_pending = await wf_repo.count_workflows(status=WorkflowStatus.PENDING.value)
    assert count_pending >= 1
    count_completed = await wf_repo.count_workflows(status=WorkflowStatus.COMPLETED.value)
    assert count_completed == 0

    # List workflows with filter
    list_filtered = await wf_repo.list_workflows(status=WorkflowStatus.PENDING.value)
    assert len(list_filtered) >= 1

    # Create duplicate job idempotently
    job1 = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="unique:step",
        job_type="research",
        stage=JobStage.RESEARCH.value,
        input_payload={"k": "v"},
    )
    job2 = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="unique:step",
        job_type="research",
        stage=JobStage.RESEARCH.value,
        input_payload={"k": "v"},
    )
    assert job1.id == job2.id
    await db_session.commit()

    # Load job with attempts
    loaded_job = await job_repo.get_job(job1.id, load_attempts=True)
    assert loaded_job is not None
    assert len(loaded_job.attempts) == 0

    # List jobs for workflow
    wf_jobs = await job_repo.list_jobs_for_workflow(wf.id)
    assert len(wf_jobs) == 1

    # Claim job
    claim = await job_repo.claim_next_job(worker_id="worker-tester")
    assert claim is not None
    _, attempt = claim
    await db_session.commit()

    # Non-existent job ID raises JobNotFoundError
    non_existent_id = str(uuid.uuid4())
    with pytest.raises(JobNotFoundError):
        await job_repo.complete_job(non_existent_id, "worker-tester", attempt.lease_token, {})

    with pytest.raises(JobNotFoundError):
        await job_repo.fail_job(non_existent_id, "worker-tester", attempt.lease_token, "ERR", "msg")


@pytest.mark.asyncio
async def test_concurrent_create_same_logical_job(pg_engine: AsyncEngine):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Concurrent Job Test")
        await session.commit()
        wf_id = wf.id

    async def create_job_task():
        async with session_factory() as session:
            repo = JobRepository(session)
            job = await repo.create_job(
                workflow_id=wf_id,
                logical_key="parallel:step",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={"test": 1},
            )
            await session.commit()
            return job.id

    results = await asyncio.gather(*[create_job_task() for _ in range(5)])
    assert len(set(results)) == 1

    async with session_factory() as session:
        repo = JobRepository(session)
        jobs = await repo.list_jobs_for_workflow(wf_id)
        assert len(jobs) == 1
        assert jobs[0].logical_key == "parallel:step"


@pytest.mark.asyncio
async def test_concurrent_create_same_idempotency_key(pg_engine: AsyncEngine):
    session_factory = async_sessionmaker(bind=pg_engine, class_=AsyncSession, expire_on_commit=False)
    key = f"idemp-concurrent-{uuid.uuid4()}"

    async def create_wf_task():
        async with session_factory() as session:
            repo = WorkflowRepository(session)
            wf = await repo.create_workflow(
                topic="Same Topic Concurrency",
                idempotency_key=key,
            )
            await session.commit()
            return wf.id

    results = await asyncio.gather(*[create_wf_task() for _ in range(5)])
    assert len(set(results)) == 1

    async with session_factory() as session:
        repo = WorkflowRepository(session)
        wf = await repo.get_workflow(results[0])
        assert wf is not None
        assert wf.idempotency_key == key


@pytest.mark.asyncio
async def test_create_job_unrelated_integrity_error_propagates(db_session: AsyncSession):
    repo = JobRepository(db_session)
    non_existent_wf_id = str(uuid.uuid4())

    with pytest.raises(IntegrityError):
        await repo.create_job(
            workflow_id=non_existent_wf_id,
            logical_key="orphan_step",
            job_type="research",
            stage=JobStage.RESEARCH.value,
            input_payload={},
        )
