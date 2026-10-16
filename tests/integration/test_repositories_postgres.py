"""Integration tests for repositories against real PostgreSQL 16."""

import asyncio
import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.db.models.artifact import ArtifactModel
from app.db.models.job import JobAttemptModel
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
from app.stages import (
    ArtifactAlreadyExistsError,
    ArtifactKind,
    ArtifactStore,
    FingerprintConflictError,
    InvalidArtifactLifecycleError,
    PipelineStage,
    SecurityViolationError,
    StageArtifactRepository,
)
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
    updated = await repo.update_status(
        wf.id, WorkflowStatus.RUNNING.value, current_stage=JobStage.SCRIPT.value
    )
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
    _j1, a1 = claim1
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
    _j2, a2 = claim2
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
    await job_repo.create_job(
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
    attempt.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)
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
    assert (
        await art_repo.update_lifecycle_status("non-existent", ArtifactLifecycleStatus.FAILED.value)
        is None
    )


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
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
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
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
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


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_lifecycle(db_session: AsyncSession, tmp_path):
    store = ArtifactStore(tmp_path, fsync=False)
    repo = StageArtifactRepository(db_session, store)

    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Artifact Test Topic")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="research_1",
        job_type="research",
        stage=JobStage.RESEARCH.value,
        input_payload={},
    )
    await db_session.commit()

    wf_id = uuid.UUID(wf.id)
    job_id = uuid.UUID(job.id)
    fp = "e" * 64
    content = b'{"research": "completed"}'

    # Claim job so valid attempt exists with active worker credentials
    claimed = await job_repo.claim_next_job(worker_id="test-worker")
    assert claimed is not None
    job, attempt = claimed
    await db_session.commit()

    # 1. create_pending with automatically derived canonical storage path
    pending = await repo.create_pending(
        workflow_id=wf_id,
        job_id=job_id,
        stage=PipelineStage.RESEARCH,
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
        stage_fingerprint=fp,
    )
    art_id = uuid.UUID(pending.id)
    path = Path(pending.storage_path)
    await db_session.commit()

    # 2. Write file
    await store.write_artifact(path, content)

    # 3. Finalize with mandatory worker lease credentials
    finalized = await repo.finalize(
        art_id,
        worker_id=attempt.worker_id,
        lease_token=attempt.lease_token,
    )
    await db_session.commit()
    assert finalized.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert finalized.file_size_bytes == len(content)

    # Verify physical file existence and integrity check
    available_art = await repo.get_available(art_id, verify_integrity=True)
    assert available_art.id == str(art_id)

    # 4. find_by_fingerprint finds it
    found = await repo.find_by_fingerprint(wf_id, PipelineStage.RESEARCH, fp)
    assert found is not None
    assert found.id == str(art_id)

    # 5. supersede
    superseded = await repo.supersede(art_id)
    await db_session.commit()
    assert superseded.lifecycle_status == ArtifactLifecycleStatus.SUPERSEDED.value

    # 6. find_by_fingerprint no longer finds superseded artifact
    assert await repo.find_by_fingerprint(wf_id, PipelineStage.RESEARCH, fp) is None


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_fingerprint_conflict(
    db_session: AsyncSession, tmp_path
):
    store = ArtifactStore(tmp_path, fsync=False)
    repo = StageArtifactRepository(db_session, store)

    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Conflict Test Topic")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="script_1",
        job_type="script",
        stage=JobStage.SCRIPT.value,
        input_payload={},
    )
    await db_session.commit()

    wf_id = uuid.UUID(wf.id)
    job_id = uuid.UUID(job.id)
    fp = "f" * 64

    # 1. Create first pending artifact
    art1 = await repo.create_pending(
        workflow_id=wf_id,
        job_id=job_id,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        stage_fingerprint=fp,
    )
    art1_id = uuid.UUID(art1.id)
    await db_session.commit()

    # 2. Attempt duplicate claim for same fingerprint -> FingerprintConflictError
    with pytest.raises(FingerprintConflictError):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            stage_fingerprint=fp,
        )

    # 3. Mark first failed -> partial unique index no longer blocks because FAILED is excluded!
    await repo.mark_failed(art1_id, error_message="Stage failure")
    await db_session.commit()

    # 4. Now a new pending artifact CAN be created with the same fingerprint!
    art2 = await repo.create_pending(
        workflow_id=wf_id,
        job_id=job_id,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        stage_fingerprint=fp,
    )
    await db_session.commit()
    assert art2.id != str(art1_id)
    assert art2.stage_fingerprint == fp


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_unrelated_integrity_error(
    db_session: AsyncSession, tmp_path
):
    """P1 test: non-fingerprint DB errors (e.g. invalid FK) propagate as IntegrityError."""
    store = ArtifactStore(tmp_path, fsync=False)
    repo = StageArtifactRepository(db_session, store)

    non_existent_wf_id = uuid.uuid4()
    non_existent_job_id = uuid.uuid4()
    fp = "0" * 64

    with pytest.raises(IntegrityError):
        await repo.create_pending(
            workflow_id=non_existent_wf_id,
            job_id=non_existent_job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint=fp,
        )


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_savepoint_isolation(
    db_session: AsyncSession, tmp_path
):
    """P1 test: savepoint ensures outer transaction is NOT rolled back on fingerprint conflict."""
    store = ArtifactStore(tmp_path, fsync=False)
    repo = StageArtifactRepository(db_session, store)

    wf_repo = WorkflowRepository(db_session)
    job_repo = JobRepository(db_session)

    wf = await wf_repo.create_workflow(topic="Savepoint Isolation Topic")
    job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="research_step",
        job_type="research",
        stage=JobStage.RESEARCH.value,
        input_payload={},
    )
    await db_session.commit()

    wf_id = uuid.UUID(wf.id)
    job_id = uuid.UUID(job.id)
    fp = "1" * 64

    # Create initial claim in separate transaction
    await repo.create_pending(
        workflow_id=wf_id,
        job_id=job_id,
        stage=PipelineStage.RESEARCH,
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
        stage_fingerprint=fp,
    )
    await db_session.commit()

    # Now in a new transaction: create a new job, flush it, then trigger fingerprint conflict
    new_job = await job_repo.create_job(
        workflow_id=wf.id,
        logical_key="other_step",
        job_type="script",
        stage=JobStage.SCRIPT.value,
        input_payload={},
    )
    await db_session.flush()

    # Claim conflict happens
    with pytest.raises(FingerprintConflictError):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint=fp,
        )

    # Verify new_job was NOT rolled back by the repository's conflict handling!
    await db_session.commit()
    persisted_job = await job_repo.get_job(new_job.id)
    assert persisted_job is not None
    assert persisted_job.logical_key == "other_step"


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_concurrent_lifecycle_transitions(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: concurrent finalize() vs mark_failed() serialized via row locking (FOR UPDATE).

    Two simultaneous workers attempt transitions on the same PENDING artifact.
    Exactly one must win and transition the state; the loser must raise InvalidArtifactLifecycleError.
    The final persisted state must not regress.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)

    # Setup pending artifact and physical file
    async with session_factory() as setup_session:
        wf_repo = WorkflowRepository(setup_session)
        job_repo = JobRepository(setup_session)
        wf = await wf_repo.create_workflow(topic="Concurrency Test")
        job = await job_repo.create_job(
            workflow_id=wf.id,
            logical_key="step_1",
            job_type="research",
            stage=JobStage.RESEARCH.value,
            input_payload={},
        )
        claimed = await job_repo.claim_next_job(worker_id="worker-fin")
        assert claimed is not None
        job, attempt = claimed
        repo = StageArtifactRepository(setup_session, store)
        wf_id = uuid.UUID(wf.id)
        job_id = uuid.UUID(job.id)
        art = await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint="2" * 64,
        )
        art_id = uuid.UUID(art.id)
        await store.write_artifact(Path(art.storage_path), b'{"concurrency": true}')
        await setup_session.commit()

    try:
        # Worker 1: finalize
        async def worker_finalize():
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                try:
                    res = await r.finalize(
                        art_id,
                        worker_id=attempt.worker_id,
                        lease_token=attempt.lease_token,
                    )
                    await session.commit()
                    return ("finalized", res.lifecycle_status)
                except InvalidArtifactLifecycleError:
                    await session.rollback()
                    # Verify losing transaction can cleanly recover and query without transaction poisoning
                    refreshed = await session.get(ArtifactModel, str(art_id))
                    assert refreshed is not None
                    assert refreshed.lifecycle_status in ("AVAILABLE", "FAILED")
                    raise

        # Worker 2: mark_failed
        async def worker_mark_failed():
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                try:
                    res = await r.mark_failed(art_id, error_message="Failed by concurrent worker")
                    await session.commit()
                    return ("failed", res.lifecycle_status)
                except InvalidArtifactLifecycleError:
                    await session.rollback()
                    # Verify losing transaction can cleanly recover and query without transaction poisoning
                    refreshed = await session.get(ArtifactModel, str(art_id))
                    assert refreshed is not None
                    assert refreshed.lifecycle_status in ("AVAILABLE", "FAILED")
                    raise

        # Execute simultaneously
        results = await asyncio.gather(
            worker_finalize(),
            worker_mark_failed(),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, tuple)]
        failures = [r for r in results if isinstance(r, InvalidArtifactLifecycleError)]

        assert len(successes) == 1, f"Expected exactly 1 winner, got: {results}"
        assert len(failures) == 1, f"Expected exactly 1 loser, got: {results}"

        # Verify persisted state matches winner
        _winner_type, winner_status = successes[0]
        async with session_factory() as verify_session:
            final_art = await verify_session.get(ArtifactModel, str(art_id))
            assert final_art is not None
            assert final_art.lifecycle_status == winner_status
            assert final_art.lifecycle_status in ("AVAILABLE", "FAILED")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_concurrent_simultaneous_finalize(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: concurrent finalize() vs finalize() serialized via row locking.

    Two workers simultaneously finalize the same PENDING artifact.
    Exactly one worker must succeed and transition PENDING -> AVAILABLE.
    The losing worker must see status is no longer PENDING under the row lock
    and raise InvalidArtifactLifecycleError without overwriting the winning state.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Concurrent Finalize")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            claimed = await job_repo.claim_next_job(worker_id="worker-shared")
            assert claimed is not None
            job, attempt = claimed
            repo = StageArtifactRepository(setup_session, store)
            wf_id = uuid.UUID(wf.id)
            job_id = uuid.UUID(job.id)
            art = await repo.create_pending(
                workflow_id=wf_id,
                job_id=job_id,
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="3" * 64,
            )
            art_id = uuid.UUID(art.id)
            payload = b'{"concurrent_finalize": true}'
            await store.write_artifact(Path(art.storage_path), payload)
            await setup_session.commit()

        async def worker_finalize(worker_id: int):
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                res = await r.finalize(
                    art_id,
                    worker_id=attempt.worker_id,
                    lease_token=attempt.lease_token,
                )
                await session.commit()
                return (f"worker_{worker_id}", res.lifecycle_status)

        results = await asyncio.gather(
            worker_finalize(1),
            worker_finalize(2),
            return_exceptions=True,
        )

        successes = [r for r in results if isinstance(r, tuple)]
        failures = [r for r in results if isinstance(r, InvalidArtifactLifecycleError)]

        assert len(successes) == 1, f"Expected exactly 1 winner, got: {results}"
        assert len(failures) == 1, (
            f"Expected exactly 1 loser with InvalidArtifactLifecycleError, got: {results}"
        )

        async with session_factory() as verify_session:
            final_art = await verify_session.get(ArtifactModel, str(art_id))
            assert final_art is not None
            assert final_art.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert final_art.checksum == hashlib.sha256(payload).hexdigest()
            assert final_art.file_size_bytes == len(payload)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_ambiguous_publication_recovery(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: recovery of PENDING artifact after ambiguous publication failure.

    Simulates an ambiguous storage failure where the file was successfully hard-linked,
    but directory sync raised an OSError. The worker process terminated/failed, leaving
    the DB record in PENDING.
    Post-crash reconciliation inspects the physical file, verifies its checksum and size,
    and safely reconciles the record PENDING -> AVAILABLE without deleting the file.
    """
    import errno
    from unittest.mock import patch

    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=True)
    try:
        payload = b'{"video_id": "vid-123", "frame_count": 300}'
        expected_hash = hashlib.sha256(payload).hexdigest()

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Ambiguous Publication Recovery")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="4" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await setup_session.commit()

        # Simulate ambiguous write failure: link succeeds, but sync raises OSError
        with (
            patch(
                "app.stages.artifact_store._sync_dir_if_supported",
                side_effect=OSError(errno.EIO, "Hardware sync failure"),
            ),
            pytest.raises(OSError, match="Hardware sync failure"),
        ):
            await store.write_artifact(storage_path, payload)

        # File exists physically on disk
        assert storage_path.exists()

        # Run reconciliation on PENDING artifact
        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            reconciled = await r.reconcile_pending(art_id, expected_checksum=expected_hash)
            await reconcile_session.commit()
            assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value

        # Verify database state
        async with session_factory() as verify_session:
            final_art = await verify_session.get(ArtifactModel, str(art_id))
            assert final_art is not None
            assert final_art.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert final_art.checksum == expected_hash
            assert final_art.file_size_bytes == len(payload)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_concurrent_reconciliation(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: two concurrent reconcilers racing on the same PENDING artifact.

    Both reconcilers call reconcile_pending() simultaneously.
    Because file checksum is computed outside the row lock and locked state transition is
    serialized idempotently, both reconcilers must succeed and return AVAILABLE status.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        payload = b'{"concurrent_reconciliation": true}'
        expected_hash = hashlib.sha256(payload).hexdigest()

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Concurrent Reconciliation")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="5" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        async def worker_reconcile(worker_id: int):
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                res = await r.reconcile_pending(art_id, expected_checksum=expected_hash)
                await session.commit()
                return (f"worker_{worker_id}", res.lifecycle_status)

        results = await asyncio.gather(
            worker_reconcile(1),
            worker_reconcile(2),
            return_exceptions=True,
        )

        for res in results:
            assert isinstance(res, tuple), f"Unexpected exception during reconciliation: {res}"
            assert res[1] == ArtifactLifecycleStatus.AVAILABLE.value

        async with session_factory() as verify_session:
            final_art = await verify_session.get(ArtifactModel, str(art_id))
            assert final_art is not None
            assert final_art.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert final_art.checksum == expected_hash
            assert final_art.file_size_bytes == len(payload)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_reconciliation_missing_file_and_retry(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: reconciliation transitions missing file to FAILED and is safe to retry."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Missing File Reconciliation")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="6" * 64,
            )
            art_id = uuid.UUID(art.id)
            await setup_session.commit()

        # 1. First reconciliation: file does not exist -> FAILED
        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            reconciled = await r.reconcile_pending(art_id)
            await reconcile_session.commit()
            assert reconciled.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
            assert (
                reconciled.artifact_metadata.get("error_message")
                == "Reconciliation failed: physical artifact file does not exist"
            )

        # 2. Retry reconciliation: must be idempotent and return FAILED record cleanly
        async with session_factory() as retry_session:
            r2 = StageArtifactRepository(retry_session, store)
            retry_res = await r2.reconcile_pending(art_id)
            await retry_session.commit()
            assert retry_res.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_reconciliation_checksum_mismatch(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: corrupted file payload marks record FAILED without deleting physical data."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        corrupted_payload = b'{"corrupted": true}'

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Corrupt Payload Reconciliation")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="7" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, corrupted_payload)
            await setup_session.commit()

        # Reconcile with expected checksum for a different/complete file
        expected_valid_checksum = "e" * 64
        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            reconciled = await r.reconcile_pending(
                art_id, expected_checksum=expected_valid_checksum
            )
            await reconcile_session.commit()
            assert reconciled.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
            assert "checksum mismatch" in reconciled.artifact_metadata.get("error_message", "")

        # Physical file MUST NOT be deleted (preserved for post-mortem forensics)
        assert storage_path.exists()
        assert storage_path.read_bytes() == corrupted_payload
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_reconciliation_concurrent_mutation(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: concurrent file mutation during reconciliation hashing raises SecurityViolationError."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        payload = b'{"original": true}'

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Mutation Reconciliation")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="8" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        # Simulate tampering during checksum computation
        orig_compute = store.compute_checksum

        async def mutating_compute(path: Path):
            res = await orig_compute(path)
            # External actor mutates file
            path.write_bytes(b'{"mutated": true, "extra": 12345}')
            return res

        store.compute_checksum = mutating_compute

        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            with pytest.raises(
                SecurityViolationError, match="modified concurrently during reconciliation"
            ):
                await r.reconcile_pending(art_id)
            await reconcile_session.rollback()

        # Database record remains safely in PENDING status
        async with session_factory() as verify_session:
            rec = await verify_session.get(ArtifactModel, str(art_id))
            assert rec is not None
            assert rec.lifecycle_status == ArtifactLifecycleStatus.PENDING.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_reconciliation_active_lease_rejected(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: an active worker lease must NOT be treated as abandoned by recovery.

    Third-party reconciler must be rejected with LeaseConflictError.
    The active owning worker possessing the matching lease_token must be permitted.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        payload = b'{"active_lease_test": true}'
        expected_hash = hashlib.sha256(payload).hexdigest()
        worker_id = "worker-active-42"
        lease_token = str(uuid.uuid4())

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Active Lease Reconciliation")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            # Create active execution attempt with fresh heartbeat
            attempt = JobAttemptModel(
                job_id=job.id,
                attempt_number=1,
                worker_id=worker_id,
                lease_token=lease_token,
                status=AttemptStatus.RUNNING.value,
                heartbeat_at=datetime.now(UTC),
            )
            setup_session.add(attempt)
            await setup_session.flush()
            job.current_attempt_id = attempt.id
            job.status = JobStatus.RUNNING.value

            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="a" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        # 1. Third-party recovery process attempts reconciliation -> must raise LeaseConflictError
        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            with pytest.raises(LeaseConflictError, match="actively leased by worker"):
                await r.reconcile_pending(
                    art_id,
                    worker_id="third-party-recovery",
                    lease_timeout_seconds=60.0,
                )
            await reconcile_session.rollback()

        # Artifact must remain in PENDING status (not abandoned)
        async with session_factory() as verify_session:
            rec = await verify_session.get(ArtifactModel, str(art_id))
            assert rec is not None
            assert rec.lifecycle_status == ArtifactLifecycleStatus.PENDING.value

        # 2. Active owning worker reconciles its own ambiguous publication -> permitted
        async with session_factory() as owner_session:
            r_owner = StageArtifactRepository(owner_session, store)
            reconciled = await r_owner.reconcile_pending(
                art_id,
                expected_checksum=expected_hash,
                worker_id=worker_id,
                lease_token=lease_token,
                lease_timeout_seconds=60.0,
            )
            await owner_session.commit()
            assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_reconciliation_expired_lease_permitted(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: an expired worker lease permits third-party recovery."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        payload = b'{"expired_lease_test": true}'
        expected_hash = hashlib.sha256(payload).hexdigest()

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Expired Lease Recovery")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            # Create attempt with expired heartbeat (15 minutes old)
            attempt = JobAttemptModel(
                job_id=job.id,
                attempt_number=1,
                worker_id="dead-worker-99",
                lease_token=str(uuid.uuid4()),
                status=AttemptStatus.RUNNING.value,
                heartbeat_at=datetime.now(UTC) - timedelta(minutes=15),
            )
            setup_session.add(attempt)
            await setup_session.flush()
            job.current_attempt_id = attempt.id
            job.status = JobStatus.RUNNING.value

            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="b" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        # Third-party recovery coordinator reconciles expired job
        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            reconciled = await r.reconcile_pending(
                art_id,
                expected_checksum=expected_hash,
                worker_id="recovery-coordinator",
                lease_timeout_seconds=60.0,
            )
            await reconcile_session.commit()
            assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert reconciled.checksum == expected_hash
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_reconciliation_renewed_lease_blocks_stale_recovery(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: a renewed lease or competing claim blocks a stale recovery decision under row lock."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        payload = b'{"renewed_lease_test": true}'

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Renewed Lease Test")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            attempt = JobAttemptModel(
                job_id=job.id,
                attempt_number=1,
                worker_id="worker-old",
                lease_token=str(uuid.uuid4()),
                status=AttemptStatus.RUNNING.value,
                # Expired initially
                heartbeat_at=datetime.now(UTC) - timedelta(minutes=10),
            )
            setup_session.add(attempt)
            await setup_session.flush()
            job.current_attempt_id = attempt.id
            job.status = JobStatus.RUNNING.value

            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="c" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        # Competing worker takes over and renews the lease before recovery transaction acquires row lock
        async with session_factory() as renew_session:
            att = await renew_session.get(JobAttemptModel, attempt.id)
            assert att is not None
            att.worker_id = "worker-renewed-hero"
            att.lease_token = str(uuid.uuid4())
            att.heartbeat_at = datetime.now(UTC)  # Fresh active lease
            await renew_session.commit()

        # Recovery process serialized under row lock checks the attempt, sees renewed lease, and rejects recovery
        async with session_factory() as reconcile_session:
            r = StageArtifactRepository(reconcile_session, store)
            with pytest.raises(
                LeaseConflictError, match="actively leased by worker 'worker-renewed-hero'"
            ):
                await r.reconcile_pending(
                    art_id,
                    worker_id="stale-recovery-coordinator",
                    lease_timeout_seconds=60.0,
                )
            await reconcile_session.rollback()

        # Artifact remains in PENDING status owned by the active worker
        async with session_factory() as verify_session:
            rec = await verify_session.get(ArtifactModel, str(art_id))
            assert rec is not None
            assert rec.lifecycle_status == ArtifactLifecycleStatus.PENDING.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_no_deadlock_between_competing_transactions(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: concurrent finalize() vs reconcile_pending() follow strict global lock order.

    Both transactions touch JobModel, JobAttemptModel, and ArtifactModel for the same job and artifact.
    Because both strictly acquire row locks in:
      JobModel (FOR UPDATE) -> JobAttemptModel (FOR UPDATE) -> ArtifactModel (FOR UPDATE),
    PostgreSQL serializes the transactions cleanly without deadlocks (error 40P01 / DeadlockDetected).
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        payload = b'{"deadlock_free_test": true}'
        worker_token = str(uuid.uuid4())

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Deadlock Free Hierarchy Test")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="deadlock_step",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            attempt = JobAttemptModel(
                job_id=job.id,
                attempt_number=1,
                worker_id="worker-deadlock-1",
                lease_token=worker_token,
                status=AttemptStatus.RUNNING.value,
                heartbeat_at=datetime.now(UTC),
            )
            setup_session.add(attempt)
            await setup_session.flush()
            job.current_attempt_id = attempt.id
            job.status = JobStatus.RUNNING.value

            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="d" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        # Competing Transaction 1: Worker 1 finalizes with valid lease token
        async def run_finalize():
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                try:
                    res = await r.finalize(
                        art_id,
                        worker_id="worker-deadlock-1",
                        lease_token=worker_token,
                    )
                    await session.commit()
                    return ("finalize_ok", res.lifecycle_status)
                except Exception as e:  # noqa: BLE001
                    await session.rollback()
                    return ("finalize_err", type(e).__name__)

        # Competing Transaction 2: Reconciler reconciles concurrently
        async def run_reconcile():
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                try:
                    res = await r.reconcile_pending(
                        art_id,
                        worker_id="reconciler-daemon",
                        lease_timeout_seconds=60.0,
                    )
                    await session.commit()
                    return ("reconcile_ok", res.lifecycle_status)
                except Exception as e:  # noqa: BLE001
                    await session.rollback()
                    return ("reconcile_err", type(e).__name__)

        # Execute both transactions concurrently against PostgreSQL
        results = await asyncio.gather(run_finalize(), run_reconcile())

        # CRITICAL ASSERTION: No deadlock occurred (no 40P01 / DeadlockDetected errors)
        for outcome, detail in results:
            assert "Deadlock" not in detail, f"Deadlock detected in transaction: {results}"
            assert "deadlock" not in detail.lower(), f"Deadlock detected in transaction: {results}"

        # At least one succeeded, and the artifact is in AVAILABLE status
        async with session_factory() as verify_session:
            final_art = await verify_session.get(ArtifactModel, str(art_id))
            assert final_art is not None
            assert final_art.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert final_art.checksum is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_stale_worker_fenced_after_lease_expiry_and_recovery(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: stale worker is strictly fenced off after lease expiration and recovery.

    Verifies the end-to-end fencing invariants across both DB and filesystem:
    1. Worker 1 starts, writes to a temp file, but pauses; its lease expires.
    2. Recovery detects expired lease and reconciles (fails PENDING because file was not published).
    3. Worker 1 resumes:
       - Calling finalize() with old worker/token is rejected with LeaseConflictError / InvalidArtifactLifecycleError.
       - Worker 1 cannot transition or mutate the artifact in the database.
       - Even if worker 1 attempts to publish its stale temp file to an already-available canonical path,
         ArtifactStore raises ArtifactAlreadyExistsError and prevents overwrite.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        stale_token = str(uuid.uuid4())

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Stale Worker Fencing Test")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="fenced_step",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            # Worker 1's attempt is expired (heartbeat 10 minutes ago)
            attempt_1 = JobAttemptModel(
                job_id=job.id,
                attempt_number=1,
                worker_id="stale-worker-1",
                lease_token=stale_token,
                status=AttemptStatus.RUNNING.value,
                heartbeat_at=datetime.now(UTC) - timedelta(minutes=10),
            )
            setup_session.add(attempt_1)
            await setup_session.flush()
            job.current_attempt_id = attempt_1.id
            job.status = JobStatus.RUNNING.value

            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="e" * 64,
            )
            art_id = uuid.UUID(art.id)
            canonical_path = Path(art.storage_path)
            # Worker 1 has not published the file yet (simulating paused worker during write/publish)
            assert not canonical_path.exists()
            await setup_session.commit()

        # Step 2: Recovery coordinator reconciles the abandoned artifact
        async with session_factory() as recovery_session:
            r = StageArtifactRepository(recovery_session, store)
            reconciled = await r.reconcile_pending(
                art_id,
                worker_id="recovery-coordinator",
                lease_timeout_seconds=60.0,
            )
            # Physical file was not present, so recovery marked it FAILED
            assert reconciled.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
            await recovery_session.commit()

        # Job is reassigned to Attempt 2 with new worker
        new_token = str(uuid.uuid4())
        async with session_factory() as reassign_session:
            job_db = await reassign_session.get(job.__class__, job.id)
            assert job_db is not None
            attempt_2 = JobAttemptModel(
                job_id=job.id,
                attempt_number=2,
                worker_id="active-worker-2",
                lease_token=new_token,
                status=AttemptStatus.RUNNING.value,
                heartbeat_at=datetime.now(UTC),
            )
            reassign_session.add(attempt_2)
            await reassign_session.flush()
            job_db.current_attempt_id = attempt_2.id
            await reassign_session.commit()

        # Step 3: Stale Worker 1 resumes from pause!
        # It creates its file and tries to finalize the recovered artifact
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_bytes(b'{"stale_payload": true}')

        async with session_factory() as resume_session:
            r = StageArtifactRepository(resume_session, store)

            # Invariant A (Terminal Lifecycle Fencing):
            # Because recovery already marked art_id as FAILED, finalize() immediately rejects
            # the stale transition with InvalidArtifactLifecycleError (FAILED -> AVAILABLE is illegal)
            with pytest.raises(
                InvalidArtifactLifecycleError, match="Invalid artifact lifecycle transition"
            ):
                await r.finalize(
                    art_id,
                    worker_id="stale-worker-1",
                    lease_token=stale_token,
                )
            await resume_session.rollback()

        # Invariant B (Active Lease Fencing under Row Lock):
        # Even if a new PENDING artifact was created under the same job, the stale worker cannot finalize it
        # because the active lease now belongs to active-worker-2!
        async with session_factory() as pending_session:
            r = StageArtifactRepository(pending_session, store)
            art2 = await r.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="f" * 64,
            )
            art2_id = uuid.UUID(art2.id)
            art2_path = Path(art2.storage_path)
            art2_path.parent.mkdir(parents=True, exist_ok=True)
            art2_path.write_bytes(b'{"pending_new": true}')
            await pending_session.commit()

        async with session_factory() as stale_session:
            r = StageArtifactRepository(stale_session, store)
            with pytest.raises(LeaseConflictError, match="does not own active lease on job"):
                await r.finalize(
                    art2_id,
                    worker_id="stale-worker-1",
                    lease_token=stale_token,
                )
            await stale_session.rollback()

        # Invariant C (Filesystem Immutability / Collision Fencing):
        # If an authoritative canonical file already exists, publishing another temp file fails
        temp_file = tmp_path / "stale_temp.json"
        temp_file.write_bytes(b'{"stale_overwrite_attempt": true}')
        temp_fd = 0  # not used by error path because fsync=False
        with pytest.raises(ArtifactAlreadyExistsError, match="already exists"):
            store._publish_temp_sync(temp_fd, str(temp_file), art2_path)

        # Authoritative state remains intact: art_id is FAILED, art2 is PENDING
        async with session_factory() as verify_session:
            art_db = await verify_session.get(ArtifactModel, str(art_id))
            assert art_db is not None
            assert art_db.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
            art2_db = await verify_session.get(ArtifactModel, str(art2_id))
            assert art2_db is not None
            assert art2_db.lifecycle_status == ArtifactLifecycleStatus.PENDING.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_stage_artifact_repository_postgres_finalize_expired_heartbeat_rejected(
    pg_engine: AsyncEngine, tmp_path
):
    """P1 test: finalize rejects worker whose lease heartbeat has expired even if worker_id and token match."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path, fsync=False)
    try:
        worker_token = str(uuid.uuid4())
        payload = b'{"expired_finalize": true}'

        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Expired Heartbeat Finalize Test")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="expired_step",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            # Heartbeat was 5 minutes ago (300s > 60s lease timeout)
            attempt = JobAttemptModel(
                job_id=job.id,
                attempt_number=1,
                worker_id="worker-paused-1",
                lease_token=worker_token,
                status=AttemptStatus.RUNNING.value,
                heartbeat_at=datetime.now(UTC) - timedelta(minutes=5),
            )
            setup_session.add(attempt)
            await setup_session.flush()
            job.current_attempt_id = attempt.id
            job.status = JobStatus.RUNNING.value

            repo = StageArtifactRepository(setup_session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="1" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, payload)
            await setup_session.commit()

        # Worker resumes without refreshing heartbeat and calls finalize
        async with session_factory() as finalize_session:
            r = StageArtifactRepository(finalize_session, store)
            with pytest.raises(LeaseConflictError, match="has expired"):
                await r.finalize(
                    art_id,
                    worker_id="worker-paused-1",
                    lease_token=worker_token,
                    lease_timeout_seconds=60.0,
                )
            await finalize_session.rollback()

        # Artifact remains in PENDING status
        async with session_factory() as verify_session:
            art_db = await verify_session.get(ArtifactModel, str(art_id))
            assert art_db is not None
            assert art_db.lifecycle_status == ArtifactLifecycleStatus.PENDING.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_finalize_rejects_missing_credentials_on_job_artifact_postgres(
    pg_engine: AsyncEngine, tmp_path: Path
):
    """P1 test: finalize rejects calls omitting worker credentials on job-associated artifacts."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_fencing_bypass", fsync=False)
    try:
        async with session_factory() as session:
            wf_repo = WorkflowRepository(session)
            job_repo = JobRepository(session)
            wf = await wf_repo.create_workflow(topic="Lease Fencing Bypass Test")
            await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            claimed = await job_repo.claim_next_job(worker_id="test-worker")
            assert claimed is not None
            job, _attempt = claimed
            repo = StageArtifactRepository(session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="a" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, b'{"secure": true}')
            await session.commit()

        # Call finalize omitting worker credentials
        async with session_factory() as finalize_session:
            repo = StageArtifactRepository(finalize_session, store)
            with pytest.raises(
                ValueError,
                match="worker_id and lease_token are mandatory to finalize job-associated artifact",
            ):
                await repo.finalize(art_id)
            await finalize_session.rollback()

        # Artifact remains PENDING
        async with session_factory() as verify_session:
            art_db = await verify_session.get(ArtifactModel, str(art_id))
            assert art_db is not None
            assert art_db.lifecycle_status == ArtifactLifecycleStatus.PENDING.value
    finally:
        store.close()


@pytest.mark.asyncio
async def test_finalize_administrative_postgres(pg_engine: AsyncEngine, tmp_path: Path):
    """P1 test: administrative finalization allows explicitly authorized bypass and records audit metadata."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_admin_finalization", fsync=False)
    try:
        async with session_factory() as session:
            wf_repo = WorkflowRepository(session)
            job_repo = JobRepository(session)
            wf = await wf_repo.create_workflow(topic="Admin Finalization Test")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_admin",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="b" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)
            await store.write_artifact(storage_path, b'{"admin": "override"}')
            await session.commit()

        # Call finalize_administrative with empty reason -> raises ValueError
        async with session_factory() as err_session:
            repo = StageArtifactRepository(err_session, store)
            with pytest.raises(ValueError, match="admin_reason must be a non-empty string"):
                await repo.finalize_administrative(art_id, admin_reason="")

        # Call finalize_administrative with valid reason
        async with session_factory() as admin_session:
            repo = StageArtifactRepository(admin_session, store)
            finalized = await repo.finalize_administrative(
                art_id, admin_reason="Trusted operator recovery repair"
            )
            await admin_session.commit()
            assert finalized.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert (
                finalized.artifact_metadata["administrative_finalization"]["reason"]
                == "Trusted operator recovery repair"
            )

        # Verification: get_available works
        async with session_factory() as verify_session:
            repo = StageArtifactRepository(verify_session, store)
            available = await repo.get_available(art_id, verify_integrity=True)
            assert available.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
            assert available.checksum is not None
    finally:
        store.close()


@pytest.mark.asyncio
async def test_quarantine_stale_file_on_failed_artifact_postgres(
    pg_engine: AsyncEngine, tmp_path: Path
):
    """P1 test: physical file published after recovery marks artifact FAILED is safely quarantined."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_stale_quarantine", fsync=False)
    try:
        async with session_factory() as session:
            wf_repo = WorkflowRepository(session)
            job_repo = JobRepository(session)
            wf = await wf_repo.create_workflow(topic="Stale Quarantine Test")
            job = await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="step_quarantine",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            repo = StageArtifactRepository(session, store)
            art = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(job.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="c" * 64,
            )
            art_id = uuid.UUID(art.id)
            storage_path = Path(art.storage_path)

            # Recovery marks artifact FAILED because worker lease expired and file was missing
            await repo.mark_failed(art_id, error_message="Worker lease expired before file publish")
            await session.commit()

        # Stale worker resumes and publishes file to canonical path
        await store.write_artifact(storage_path, b'{"stale_data": true}')
        assert storage_path.exists()

        # Reconciler runs and detects orphan file for FAILED artifact
        async with session_factory() as rec_session:
            repo = StageArtifactRepository(rec_session, store)
            rec = await repo.reconcile_pending(art_id)
            await rec_session.commit()

            # Record remains FAILED
            assert rec.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
            # Canonical path is no longer occupied
            assert not storage_path.exists()

            # Metadata records quarantined stale file
            metadata = rec.artifact_metadata or {}
            quarantined_files = metadata.get("quarantined_stale_files", [])
            assert len(quarantined_files) == 1
            quarantine_path = Path(quarantined_files[0])
            assert quarantine_path.exists()
            assert quarantine_path.read_bytes() == b'{"stale_data": true}'
    finally:
        store.close()


@pytest.mark.asyncio
async def test_lock_order_inventory_concurrency_postgres(pg_engine: AsyncEngine, tmp_path: Path):
    """P1 test: concurrent transactions across JobModel -> JobAttemptModel -> ArtifactModel

    Executes:
    - Transaction A: Job claim (JobModel lock)
    - Transaction B: Finalize (JobModel -> JobAttemptModel -> ArtifactModel lock)
    - Transaction C: Reconcile (JobModel -> JobAttemptModel -> ArtifactModel lock)

    Verifies deadlock-free concurrent execution across all multi-row paths conforming to hierarchy.
    """
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_lock_hierarchy", fsync=False)
    try:
        async with session_factory() as setup_session:
            wf_repo = WorkflowRepository(setup_session)
            job_repo = JobRepository(setup_session)
            wf = await wf_repo.create_workflow(topic="Lock Order Test")

            # Job 1 for finalize
            await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="job_hier_1",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            c1 = await job_repo.claim_next_job(worker_id="worker-hier-1")
            assert c1 is not None
            j1, att1 = c1

            # Job 2 for claim_next_job
            await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="job_hier_2",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )

            # Job 3 for reconcile
            await job_repo.create_job(
                workflow_id=wf.id,
                logical_key="job_hier_3",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={},
            )
            c3 = await job_repo.claim_next_job(worker_id="worker-hier-3")
            assert c3 is not None
            j3, att3 = c3

            # Artifact 1
            repo = StageArtifactRepository(setup_session, store)
            art1 = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(j1.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="d" * 64,
            )
            art1_id = uuid.UUID(art1.id)
            await store.write_artifact(Path(art1.storage_path), b'{"hier": 1}')

            # Artifact 3
            art3 = await repo.create_pending(
                workflow_id=uuid.UUID(wf.id),
                job_id=uuid.UUID(j3.id),
                stage=PipelineStage.RESEARCH,
                artifact_kind=ArtifactKind.RESEARCH_RESULT,
                stage_fingerprint="e" * 64,
            )
            art3_id = uuid.UUID(art3.id)
            await store.write_artifact(Path(art3.storage_path), b'{"hier": 3}')

            await setup_session.commit()

        async def tx_finalize():
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                res = await r.finalize(
                    art1_id,
                    worker_id=att1.worker_id,
                    lease_token=att1.lease_token,
                )
                await session.commit()
                return ("finalize", res.lifecycle_status)

        async def tx_claim_job():
            async with session_factory() as session:
                jr = JobRepository(session)
                claimed = await jr.claim_next_job(worker_id="worker-hier-2")
                await session.commit()
                return ("claim_job", claimed is not None)

        async def tx_reconcile():
            async with session_factory() as session:
                r = StageArtifactRepository(session, store)
                res = await r.reconcile_pending(
                    art3_id,
                    worker_id=att3.worker_id,
                    lease_token=att3.lease_token,
                )
                await session.commit()
                return ("reconcile", res.lifecycle_status)

        results = await asyncio.gather(
            tx_finalize(),
            tx_claim_job(),
            tx_reconcile(),
            return_exceptions=True,
        )

        for res in results:
            assert not isinstance(res, Exception), f"Unexpected failure in lock order test: {res}"

        assert results[0] == ("finalize", ArtifactLifecycleStatus.AVAILABLE.value)
        assert results[1] == ("claim_job", True)
        assert results[2] == ("reconcile", ArtifactLifecycleStatus.AVAILABLE.value)
    finally:
        store.close()
