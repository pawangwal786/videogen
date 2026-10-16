"""Unit tests for StageArtifactRepository."""

import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.base import utc_now
from app.db.models.artifact import ArtifactModel
from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration.errors import LeaseConflictError
from app.orchestration.state_machine import ArtifactLifecycleStatus, AttemptStatus
from app.stages.artifact_errors import (
    ArtifactUnavailableError,
    FingerprintConflictError,
    InvalidArtifactLifecycleError,
    SecurityViolationError,
)
from app.stages.artifact_store import ArtifactStore
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.stage_artifact_repository import StageArtifactRepository


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.get = AsyncMock()
    session.execute = AsyncMock()

    # Support async with session.begin_nested()
    nested_cm = AsyncMock()
    nested_cm.__aenter__.return_value = None
    nested_cm.__aexit__.return_value = None
    session.begin_nested = MagicMock(return_value=nested_cm)

    return session


@pytest.fixture
def mock_store() -> AsyncMock:
    store = AsyncMock(spec=ArtifactStore)
    return store


@pytest.fixture
def repo(mock_session: AsyncMock, mock_store: AsyncMock) -> StageArtifactRepository:
    return StageArtifactRepository(mock_session, mock_store)


@pytest.mark.asyncio
async def test_create_pending_success_canonical_path_derived(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock
):
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    art_id = uuid.uuid4()
    valid_fp = "a" * 64
    canonical_path = Path("/artifacts") / str(wf_id) / "RESEARCH" / f"{art_id}.json"
    mock_store.build_path.return_value = canonical_path

    artifact = await repo.create_pending(
        workflow_id=wf_id,
        job_id=job_id,
        stage=PipelineStage.RESEARCH,
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
        stage_fingerprint=valid_fp,
        extension="json",
        mime_type="application/json",
        artifact_id=art_id,
    )

    assert artifact.workflow_id == str(wf_id)
    assert artifact.job_id == str(job_id)
    assert artifact.id == str(art_id)
    assert artifact.stage == PipelineStage.RESEARCH.value
    assert artifact.artifact_kind == ArtifactKind.RESEARCH_RESULT.value
    assert artifact.stage_fingerprint == valid_fp
    assert artifact.storage_path == str(canonical_path)
    assert artifact.lifecycle_status == ArtifactLifecycleStatus.PENDING.value
    mock_session.add.assert_called_once()
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_pending_runtime_enum_validation(repo: StageArtifactRepository):
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    valid_fp = "a" * 64

    # Raw string instead of PipelineStage enum
    with pytest.raises(TypeError, match="PipelineStage enum member"):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage="RESEARCH",  # type: ignore[arg-type]
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint=valid_fp,
        )

    # Raw string instead of ArtifactKind enum
    with pytest.raises(TypeError, match="ArtifactKind enum member"):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind="RESEARCH_RESULT",  # type: ignore[arg-type]
            stage_fingerprint=valid_fp,
        )


@pytest.mark.asyncio
async def test_create_pending_storage_path_canonical_validation(
    repo: StageArtifactRepository, mock_store: AsyncMock
):
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    art_id = uuid.uuid4()
    valid_fp = "a" * 64
    canonical_path = Path("/artifacts") / str(wf_id) / "RESEARCH" / f"{art_id}.json"
    mock_store.build_path.return_value = canonical_path
    mock_store._validate_confinement.return_value = canonical_path

    # Matching canonical path -> succeeds
    art = await repo.create_pending(
        workflow_id=wf_id,
        job_id=job_id,
        stage=PipelineStage.RESEARCH,
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
        stage_fingerprint=valid_fp,
        storage_path=str(canonical_path),
        artifact_id=art_id,
    )
    assert art.storage_path == str(canonical_path.resolve())

    # Non-matching canonical layout -> raises SecurityViolationError
    wrong_layout_path = Path("/artifacts") / str(wf_id) / "SCRIPT" / f"{art_id}.json"
    with pytest.raises(SecurityViolationError, match="does not match canonical layout"):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint=valid_fp,
            storage_path=str(wrong_layout_path),
            artifact_id=art_id,
        )


@pytest.mark.asyncio
async def test_create_pending_invalid_fingerprint(repo: StageArtifactRepository):
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()

    with pytest.raises(ValueError, match="64-char lowercase hex"):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint="not-a-valid-hex",
        )


@pytest.mark.asyncio
async def test_create_pending_fingerprint_integrity_conflict(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock
):
    """P1 test: only uq_artifacts_active_fingerprint is converted to FingerprintConflictError."""
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    valid_fp = "b" * 64
    mock_store.build_path.return_value = Path("/artifacts/test.json")

    mock_session.flush.side_effect = IntegrityError(
        "statement", {}, Exception("violates unique constraint uq_artifacts_active_fingerprint")
    )

    with pytest.raises(FingerprintConflictError) as exc_info:
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            stage_fingerprint=valid_fp,
        )

    assert exc_info.value.workflow_id == str(wf_id)
    assert exc_info.value.stage == PipelineStage.SCRIPT.value
    assert exc_info.value.stage_fingerprint == valid_fp


@pytest.mark.asyncio
async def test_create_pending_unrelated_integrity_error_propagates(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock
):
    """P1 test: unrelated integrity error (e.g. FK violation) is re-raised."""
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    valid_fp = "b" * 64
    mock_store.build_path.return_value = Path("/artifacts/test.json")

    mock_session.flush.side_effect = IntegrityError(
        "statement", {}, Exception("violates foreign key constraint fk_artifacts_workflow_id")
    )

    with pytest.raises(IntegrityError, match="fk_artifacts_workflow_id"):
        await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            stage_fingerprint=valid_fp,
        )


@pytest.mark.asyncio
async def test_finalize_success_with_row_lock(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    art_id = uuid.uuid4()
    test_file = tmp_path / "test.json"
    test_file.write_bytes(b'{"key": "value"}')

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result
    mock_store.compute_checksum.return_value = ("computed-sha256", 16)

    finalized = await repo.finalize(art_id)

    assert finalized.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert finalized.checksum == "computed-sha256"
    assert finalized.file_size_bytes == 16
    assert (
        mock_session.execute.await_count == 2
    )  # Step 1: read without lock, Step 3: row lock FOR UPDATE
    mock_store.compute_checksum.assert_awaited_once_with(test_file)
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_detects_concurrent_external_file_mutation(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P2 test: verifies that external mutation between hash calculation and lock aborts finalization."""
    art_id = uuid.uuid4()
    test_file = tmp_path / "test.json"
    test_file.write_bytes(b'{"key": "value"}')

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    async def mutate_file_during_hash(path: Path):
        # External actor mutates file during checksum calculation
        path.write_bytes(b'{"tampered": true}')
        return ("hash-of-original", 16)

    mock_store.compute_checksum.side_effect = mutate_file_during_hash

    with pytest.raises(SecurityViolationError, match="modified concurrently during finalization"):
        await repo.finalize(art_id)


@pytest.mark.asyncio
async def test_reconcile_pending_success(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 test: reconciliation transitions PENDING -> AVAILABLE when physical file is intact."""
    art_id = uuid.uuid4()
    test_file = tmp_path / "recovered.mp4"
    test_file.write_bytes(b"intact_media_payload")

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result
    mock_store.compute_checksum.return_value = ("expected_hash_123", 20)

    reconciled = await repo.reconcile_pending(art_id, expected_checksum="expected_hash_123")

    assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert reconciled.checksum == "expected_hash_123"
    assert reconciled.file_size_bytes == 20
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_pending_checksum_mismatch_marks_failed(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 test: reconciliation transitions PENDING -> FAILED when checksum mismatches."""
    art_id = uuid.uuid4()
    test_file = tmp_path / "corrupt.mp4"
    test_file.write_bytes(b"corrupt_partial_data")

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result
    mock_store.compute_checksum.return_value = ("actual_corrupt_hash", 20)

    reconciled = await repo.reconcile_pending(art_id, expected_checksum="expected_valid_hash")

    assert reconciled.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
    assert "checksum mismatch" in (reconciled.artifact_metadata or {}).get("error_message", "")
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_pending_missing_file_marks_failed(
    repo: StageArtifactRepository, mock_session: AsyncMock, tmp_path: Path
):
    """P1 test: reconciliation transitions PENDING -> FAILED when physical file does not exist."""
    art_id = uuid.uuid4()
    missing_file = tmp_path / "nonexistent.mp4"

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(missing_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    reconciled = await repo.reconcile_pending(art_id)

    assert reconciled.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
    assert "physical artifact file does not exist" in (reconciled.artifact_metadata or {}).get(
        "error_message", ""
    )
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_pending_already_available_returns_idempotently(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    """P1 test: reconciliation returns immediately without mutation if already AVAILABLE."""
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/already_done.json",
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
        checksum="hash123",
        file_size_bytes=42,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    reconciled = await repo.reconcile_pending(art_id)
    assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    # flush should NOT be called since no mutation occurred
    mock_session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_pending_detects_concurrent_external_file_mutation(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 test: concurrent file mutation during reconciliation hashing raises SecurityViolationError."""
    art_id = uuid.uuid4()
    test_file = tmp_path / "tamper_target.mp4"
    test_file.write_bytes(b"original_bytes")

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    async def tamper_file(path: Path):
        path.write_bytes(b"tampered_bytes_mutation")
        return ("original_hash", 14)

    mock_store.compute_checksum.side_effect = tamper_file

    with pytest.raises(SecurityViolationError, match="modified concurrently during reconciliation"):
        await repo.reconcile_pending(art_id)


@pytest.mark.asyncio
async def test_reconcile_pending_active_lease_rejected(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 unit test: reconciliation fails with LeaseConflictError when owning attempt has active lease."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    att_id = uuid.uuid4()
    test_file = tmp_path / "active_leased.mp4"
    test_file.write_bytes(b"active_leased_payload")

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(wf_id),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(wf_id),
        logical_key="job_key",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=str(att_id),
    )
    attempt = JobAttemptModel(
        id=str(att_id),
        job_id=str(job_id),
        attempt_number=1,
        worker_id="worker-active-42",
        lease_token="token-abc-123",
        status=AttemptStatus.RUNNING.value,
        heartbeat_at=utc_now(),
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # initial _get_artifact(for_update=False)
        make_res(job),  # job query under lock
        make_res(attempt),  # attempt query under lock
    ]
    mock_store.compute_checksum.return_value = ("hash_456", 21)

    with pytest.raises(LeaseConflictError, match="actively leased by worker 'worker-active-42'"):
        await repo.reconcile_pending(
            art_id,
            worker_id="third-party-recovery",
            lease_timeout_seconds=60.0,
        )


@pytest.mark.asyncio
async def test_reconcile_pending_expired_lease_permitted(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 unit test: reconciliation succeeds when owning attempt lease has expired."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    att_id = uuid.uuid4()
    test_file = tmp_path / "expired_leased.mp4"
    test_file.write_bytes(b"expired_leased_payload")

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(wf_id),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(wf_id),
        logical_key="job_key",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=str(att_id),
    )
    attempt = JobAttemptModel(
        id=str(att_id),
        job_id=str(job_id),
        attempt_number=1,
        worker_id="dead-worker-1",
        lease_token="token-old",
        status=AttemptStatus.RUNNING.value,
        heartbeat_at=utc_now() - timedelta(minutes=10),
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # Step 1: initial _get_artifact(for_update=False)
        make_res(job),  # Step 3: job query under lock
        make_res(attempt),  # Step 3: attempt query under lock
        make_res(artifact),  # Step 4: locked _get_artifact(for_update=True)
    ]
    mock_store.compute_checksum.return_value = ("valid_hash_789", 22)

    reconciled = await repo.reconcile_pending(
        art_id,
        expected_checksum="valid_hash_789",
        worker_id="recovery-coordinator",
        lease_timeout_seconds=60.0,
    )

    assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert reconciled.checksum == "valid_hash_789"
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_pending_owner_worker_permitted(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 unit test: owning worker possessing matching lease_token can reconcile its own pending artifact."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    att_id = uuid.uuid4()
    test_file = tmp_path / "owner_reconcile.mp4"
    test_file.write_bytes(b"owner_reconcile_payload")

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(wf_id),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(wf_id),
        logical_key="job_key",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=str(att_id),
    )
    attempt = JobAttemptModel(
        id=str(att_id),
        job_id=str(job_id),
        attempt_number=1,
        worker_id="my-worker-id",
        lease_token="my-secret-token",
        status=AttemptStatus.RUNNING.value,
        heartbeat_at=utc_now(),  # Active lease
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # Step 1: initial _get_artifact(for_update=False)
        make_res(job),  # Step 3: job query under lock
        make_res(attempt),  # Step 3: attempt query under lock
        make_res(artifact),  # Step 4: locked _get_artifact(for_update=True)
    ]
    mock_store.compute_checksum.return_value = ("owner_hash_321", 23)

    reconciled = await repo.reconcile_pending(
        art_id,
        expected_checksum="owner_hash_321",
        worker_id="my-worker-id",
        lease_token="my-secret-token",
        lease_timeout_seconds=60.0,
    )

    assert reconciled.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert reconciled.checksum == "owner_hash_321"


@pytest.mark.asyncio
async def test_finalize_with_active_lease_owner_success(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 unit test: finalize succeeds when owning worker holds active, matching lease token."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    att_id = uuid.uuid4()
    test_file = tmp_path / "final_success.json"
    test_file.write_bytes(b'{"status": "ok"}')

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(wf_id),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(wf_id),
        logical_key="job_key",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=str(att_id),
    )
    attempt = JobAttemptModel(
        id=str(att_id),
        job_id=str(job_id),
        attempt_number=1,
        worker_id="owning-worker-1",
        lease_token="valid-tok-123",
        status=AttemptStatus.RUNNING.value,
        heartbeat_at=utc_now(),
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # Step 1: initial _get_artifact(for_update=False)
        make_res(job),  # Step 3: job query under lock
        make_res(attempt),  # Step 3: attempt query under lock
        make_res(artifact),  # Step 4: locked _get_artifact(for_update=True)
    ]
    mock_store.compute_checksum.return_value = ("hash-final-123", 16)

    finalized = await repo.finalize(
        art_id,
        worker_id="owning-worker-1",
        lease_token="valid-tok-123",
    )

    assert finalized.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert finalized.checksum == "hash-final-123"
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_stale_worker_lease_conflict(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 unit test: finalize rejects stale worker with LeaseConflictError when lease was reassigned."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    att_id = uuid.uuid4()
    test_file = tmp_path / "stale_finalize.json"
    test_file.write_bytes(b'{"status": "stale"}')

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(wf_id),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(wf_id),
        logical_key="job_key",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=str(att_id),
    )
    # Database attempt belongs to new worker!
    attempt = JobAttemptModel(
        id=str(att_id),
        job_id=str(job_id),
        attempt_number=2,
        worker_id="new-worker-2",
        lease_token="new-token-456",
        status=AttemptStatus.RUNNING.value,
        heartbeat_at=utc_now(),
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # Step 1: initial _get_artifact(for_update=False)
        make_res(job),  # Step 3: job query under lock
        make_res(attempt),  # Step 3: attempt query under lock
    ]
    mock_store.compute_checksum.return_value = ("hash-stale", 18)

    with pytest.raises(LeaseConflictError, match="does not own active lease on job"):
        await repo.finalize(
            art_id,
            worker_id="old-worker-1",
            lease_token="old-token-expired",
        )
    mock_session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_expired_lease_worker_conflict(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 unit test: finalize rejects worker whose lease heartbeat has expired even if worker_id and token match."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    att_id = uuid.uuid4()
    test_file = tmp_path / "expired_finalize.json"
    test_file.write_bytes(b'{"status": "expired"}')

    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(wf_id),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(wf_id),
        logical_key="job_key",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=str(att_id),
    )
    # Attempt has matching worker and token, but expired heartbeat!
    attempt = JobAttemptModel(
        id=str(att_id),
        job_id=str(job_id),
        attempt_number=1,
        worker_id="same-worker-1",
        lease_token="same-token-123",
        status=AttemptStatus.RUNNING.value,
        heartbeat_at=utc_now() - timedelta(minutes=5),  # 300s old > 60s
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # Step 1: initial _get_artifact(for_update=False)
        make_res(job),  # Step 3: job query under lock
        make_res(attempt),  # Step 3: attempt query under lock
    ]
    mock_store.compute_checksum.return_value = ("hash-expired", 19)

    with pytest.raises(LeaseConflictError, match="has expired"):
        await repo.finalize(
            art_id,
            worker_id="same-worker-1",
            lease_token="same-token-123",
            lease_timeout_seconds=60.0,
        )
    mock_session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_invalid_transition(repo: StageArtifactRepository, mock_session: AsyncMock):
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,  # Already AVAILABLE
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    with pytest.raises(InvalidArtifactLifecycleError) as exc_info:
        await repo.finalize(art_id)

    assert exc_info.value.current_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert exc_info.value.target_status == ArtifactLifecycleStatus.AVAILABLE.value


@pytest.mark.asyncio
async def test_finalize_not_found(repo: StageArtifactRepository, mock_session: AsyncMock):
    art_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    with pytest.raises(ValueError, match="not found"):
        await repo.finalize(art_id)


@pytest.mark.asyncio
async def test_mark_failed_success_with_row_lock(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    failed = await repo.mark_failed(art_id, error_message="Provider timeout")

    assert failed.lifecycle_status == ArtifactLifecycleStatus.FAILED.value
    assert failed.artifact_metadata == {"error_message": "Provider timeout"}
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_mark_failed_invalid_transition(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    with pytest.raises(InvalidArtifactLifecycleError):
        await repo.mark_failed(art_id)


@pytest.mark.asyncio
async def test_supersede_success_with_row_lock(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    superseded = await repo.supersede(art_id)

    assert superseded.lifecycle_status == ArtifactLifecycleStatus.SUPERSEDED.value
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_supersede_invalid_transition(repo: StageArtifactRepository, mock_session: AsyncMock):
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    with pytest.raises(InvalidArtifactLifecycleError):
        await repo.supersede(art_id)


@pytest.mark.asyncio
async def test_find_by_fingerprint_valid(repo: StageArtifactRepository, mock_session: AsyncMock):
    wf_id = uuid.uuid4()
    fp = "c" * 64
    artifact = ArtifactModel(
        id=str(uuid.uuid4()),
        workflow_id=str(wf_id),
        stage=PipelineStage.AUDIO.value,
        stage_fingerprint=fp,
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    found = await repo.find_by_fingerprint(wf_id, PipelineStage.AUDIO, fp)
    assert found == artifact


@pytest.mark.asyncio
async def test_find_by_fingerprint_enum_validation(repo: StageArtifactRepository):
    wf_id = uuid.uuid4()
    with pytest.raises(TypeError, match="PipelineStage enum member"):
        await repo.find_by_fingerprint(wf_id, "AUDIO", "c" * 64)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_get_available_success(
    repo: StageArtifactRepository, mock_session: AsyncMock, tmp_path: Path
):
    art_id = uuid.uuid4()
    test_file = tmp_path / "available_art.json"
    test_file.write_bytes(b'{"available": true}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_session.get.return_value = artifact

    res = await repo.get_available(art_id)
    assert res == artifact


@pytest.mark.asyncio
async def test_get_available_missing_physical_file_raises(
    repo: StageArtifactRepository, mock_session: AsyncMock, tmp_path: Path
):
    art_id = uuid.uuid4()
    missing_file = tmp_path / "missing_physical.json"
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(missing_file),
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_session.get.return_value = artifact

    with pytest.raises(ArtifactUnavailableError, match="physical artifact file does not exist"):
        await repo.get_available(art_id)


@pytest.mark.asyncio
async def test_get_available_verify_integrity_checksum_mismatch_raises(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    art_id = uuid.uuid4()
    test_file = tmp_path / "corrupted.json"
    test_file.write_bytes(b'{"mutated": true}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        checksum="original-expected-sha256",
        file_size_bytes=16,
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_session.get.return_value = artifact
    mock_store.read_artifact.return_value = test_file.read_bytes()
    mock_store.compute_checksum.return_value = ("tampered-disk-sha256", 16)

    with pytest.raises(
        SecurityViolationError, match="Integrity failure: artifact .* checksum mismatch"
    ):
        await repo.get_available(art_id, verify_integrity=True)


@pytest.mark.asyncio
async def test_get_available_not_found(repo: StageArtifactRepository, mock_session: AsyncMock):
    art_id = uuid.uuid4()
    mock_session.get.return_value = None

    with pytest.raises(ArtifactUnavailableError, match="not found"):
        await repo.get_available(art_id)


@pytest.mark.asyncio
async def test_get_available_not_available(repo: StageArtifactRepository, mock_session: AsyncMock):
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_session.get.return_value = artifact

    with pytest.raises(ArtifactUnavailableError, match="expected AVAILABLE, got PENDING"):
        await repo.get_available(art_id)


@pytest.mark.asyncio
async def test_get_available_verify_integrity_missing_checksum_fails_closed(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    art_id = uuid.uuid4()
    test_file = tmp_path / "valid.json"
    test_file.write_bytes(b'{"valid": true}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        checksum=None,  # Missing checksum metadata
        file_size_bytes=15,
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_session.get.return_value = artifact

    with pytest.raises(SecurityViolationError, match="missing or empty stored checksum"):
        await repo.get_available(art_id, verify_integrity=True)


@pytest.mark.asyncio
async def test_get_available_verify_integrity_missing_filesize_fails_closed(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    art_id = uuid.uuid4()
    test_file = tmp_path / "valid.json"
    test_file.write_bytes(b'{"valid": true}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path=str(test_file),
        checksum="some-valid-sha256",
        file_size_bytes=None,  # Missing file_size_bytes metadata
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE.value,
    )
    mock_session.get.return_value = artifact

    with pytest.raises(SecurityViolationError, match="missing or invalid stored file_size_bytes"):
        await repo.get_available(art_id, verify_integrity=True)


@pytest.mark.asyncio
async def test_find_by_fingerprint_filters_by_artifact_kind(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    wf_id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    res = await repo.find_by_fingerprint(
        wf_id,
        PipelineStage.RESEARCH,
        "test-fp-123",
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
    )
    assert res is None
    mock_session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_job_artifact_missing_credentials_raises_value_error(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    """P1 test: finalize rejects job-associated artifacts when worker credentials are omitted."""
    art_id = uuid.uuid4()
    job_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        job_id=str(job_id),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    with pytest.raises(ValueError, match="worker_id and lease_token are mandatory"):
        await repo.finalize(art_id)


@pytest.mark.asyncio
async def test_finalize_lease_timeout_validation(repo: StageArtifactRepository):
    """P1 test: lease_timeout_seconds must be positive and finite."""
    art_id = uuid.uuid4()
    with pytest.raises(ValueError, match="lease_timeout_seconds must be a positive finite number"):
        await repo.finalize(art_id, lease_timeout_seconds=0.0)

    with pytest.raises(ValueError, match="lease_timeout_seconds must be a positive finite number"):
        await repo.finalize(art_id, lease_timeout_seconds=float("inf"))


@pytest.mark.asyncio
async def test_finalize_referenced_job_missing_raises_lease_conflict(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 test: job-associated artifact whose job row is missing raises LeaseConflictError under lock."""
    art_id = uuid.uuid4()
    job_id = uuid.uuid4()
    test_file = tmp_path / "test.json"
    test_file.write_bytes(b'{"data": 1}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # Step 1: initial _get_artifact
        make_res(None),  # Step 3: job query under lock returns None!
    ]
    mock_store.compute_checksum.return_value = ("hash", 10)

    with pytest.raises(LeaseConflictError, match="referenced job .* does not exist"):
        await repo.finalize(
            art_id,
            worker_id="worker-1",
            lease_token="tok-1",
        )


@pytest.mark.asyncio
async def test_finalize_job_no_current_attempt_raises_lease_conflict(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 test: job with no current attempt raises LeaseConflictError."""
    art_id = uuid.uuid4()
    job_id = uuid.uuid4()
    test_file = tmp_path / "test.json"
    test_file.write_bytes(b'{"data": 1}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(uuid.uuid4()),
        logical_key="jk",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=None,  # No active attempt!
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),
        make_res(job),
    ]
    mock_store.compute_checksum.return_value = ("hash", 10)

    with pytest.raises(LeaseConflictError, match="job .* has no current attempt"):
        await repo.finalize(
            art_id,
            worker_id="worker-1",
            lease_token="tok-1",
        )


@pytest.mark.asyncio
async def test_finalize_administrative_success(
    repo: StageArtifactRepository, mock_session: AsyncMock, mock_store: AsyncMock, tmp_path: Path
):
    """P1 test: finalize_administrative allows authorized bypass with reason and audit metadata."""
    art_id = uuid.uuid4()
    job_id = uuid.uuid4()
    test_file = tmp_path / "admin_test.json"
    test_file.write_bytes(b'{"admin": true}')
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        job_id=str(job_id),
        storage_path=str(test_file),
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    job = JobModel(
        id=str(job_id),
        workflow_id=str(uuid.uuid4()),
        logical_key="jk",
        job_type="research",
        stage="research",
        input_payload={},
        current_attempt_id=None,
    )

    def make_res(val):
        m = MagicMock()
        m.scalar_one_or_none.return_value = val
        return m

    mock_session.execute.side_effect = [
        make_res(artifact),  # initial
        make_res(job),  # job lock
        make_res(artifact),  # locked artifact
    ]
    mock_store.compute_checksum.return_value = ("admin-hash", 15)

    finalized = await repo.finalize_administrative(
        art_id, admin_reason="Recovery after cluster crash"
    )

    assert finalized.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value
    assert (
        finalized.artifact_metadata["administrative_finalization"]["reason"]
        == "Recovery after cluster crash"
    )
    mock_session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_administrative_empty_reason_rejected(repo: StageArtifactRepository):
    """P1 test: finalize_administrative requires a non-empty admin_reason."""
    art_id = uuid.uuid4()
    with pytest.raises(ValueError, match="admin_reason must be a non-empty string"):
        await repo.finalize_administrative(art_id, admin_reason="   ")


@pytest.mark.asyncio
async def test_quarantine_stale_file_not_failed_or_superseded_rejected(
    repo: StageArtifactRepository, mock_session: AsyncMock
):
    """P1 test: quarantine_stale_file rejects artifacts that are still in PENDING or AVAILABLE."""
    art_id = uuid.uuid4()
    artifact = ArtifactModel(
        id=str(art_id),
        workflow_id=str(uuid.uuid4()),
        storage_path="/artifacts/test.json",
        lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
    )
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = artifact
    mock_session.execute.return_value = mock_result

    with pytest.raises(InvalidArtifactLifecycleError):
        await repo.quarantine_stale_file(art_id)
