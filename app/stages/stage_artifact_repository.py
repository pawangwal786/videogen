"""Phase 9 artifact repository with lifecycle enforcement and idempotency.

Provides durable artifact management on top of ArtifactStore:
- Lifecycle enforcement: validates all state transitions against the
  monotonic graph before mutation, serialized via PostgreSQL row locks (FOR UPDATE).
- Independent finalization & reconciliation: finalize() and reconcile_pending()
  read physical files through ArtifactStore.compute_checksum() in bounded memory
  WITHOUT holding database row locks during slow I/O, preventing transaction bottlenecks.
- Immutability & Concurrency Verification: Enforces service-owned immutability by
  verifying file modification timestamps (st_mtime_ns) and sizes (st_size) before and
  after hashing. Any concurrent mutation raises SecurityViolationError.
- Threat Model Assumption: Artifact directories are exclusively service-owned; OS
  permissions prohibit untrusted local writers, eliminating same-size/timestamp-restoring
  tampering threats outside the service perimeter.
- Global Database Lock Hierarchy: All multi-row transactions MUST acquire
  row locks strictly in hierarchical order:
  1. JobModel (FOR UPDATE)
  2. JobAttemptModel (FOR UPDATE)
  3. ArtifactModel (FOR UPDATE)
  This eliminates cross-repository deadlocks between worker finalization,
  orchestrator scheduling, and recovery reconciliation.
- Fencing Invariant: Stale workers whose leases have expired or been taken over
  cannot commit authoritative state in finalize() (rejected via LeaseConflictError
  and InvalidArtifactLifecycleError), nor overwrite physical data on disk
  (rejected via ArtifactAlreadyExistsError).
"""

import hashlib
import logging
import math
import re
import uuid
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utc_now
from app.db.models.artifact import ArtifactModel
from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration.errors import InvalidStateTransitionError, LeaseConflictError
from app.orchestration.state_machine import (
    ArtifactLifecycleStatus,
    AttemptStatus,
    validate_artifact_lifecycle_transition,
)
from app.stages.artifact_errors import (
    ArtifactUnavailableError,
    FingerprintConflictError,
    InvalidArtifactLifecycleError,
    SecurityViolationError,
)
from app.stages.artifact_store import ArtifactStore
from app.stages.dto import ArtifactKind, PipelineStage

logger = logging.getLogger(__name__)

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class StageArtifactRepository:
    """Phase 9 artifact repository with lifecycle enforcement,
    independently verified finalization, fingerprint-based
    idempotency, and deterministic artifact selection.
    """

    def __init__(self, session: AsyncSession, store: ArtifactStore) -> None:
        """Initialize with DB session and physical artifact store.

        The repository requires both so that finalize() can independently
        verify artifact files through the store.
        """
        self._session = session
        self._store = store

    async def create_pending(
        self,
        workflow_id: UUID,
        job_id: UUID,
        stage: PipelineStage,
        artifact_kind: ArtifactKind,
        stage_fingerprint: str,
        storage_path: str | Path | None = None,
        extension: str = "json",
        mime_type: str = "application/json",
        artifact_id: UUID | None = None,
    ) -> ArtifactModel:
        """Create an artifact record in PENDING status.

        Validates:
        - stage is a valid PipelineStage enum member
        - artifact_kind is a valid ArtifactKind enum member
        - stage_fingerprint is a 64-char lowercase hex string
        - storage_path matches canonical UUID-addressed layout and confinement

        Uses a transaction savepoint (begin_nested) so a concurrency conflict
        rolls back only the claim attempt rather than the outer transaction.
        Only the uq_artifacts_active_fingerprint violation is translated to
        FingerprintConflictError; unrelated IntegrityErrors are re-raised.

        Args:
            workflow_id: Workflow this artifact belongs to.
            job_id: Job that is producing this artifact.
            stage: Pipeline stage producing this artifact.
            artifact_kind: Typed classification of the artifact.
            stage_fingerprint: 64-char lowercase hex SHA-256 fingerprint.
            storage_path: Optional explicit storage path (must match canonical layout).
            extension: File extension for canonical path generation (default "json").
            mime_type: MIME type of the artifact content.
            artifact_id: Optional UUID (generated automatically if None).

        Returns:
            The created ArtifactModel in PENDING status.

        Raises:
            ValueError: If enum members or fingerprint format are invalid.
            SecurityViolationError: If storage_path is outside root or does not match layout.
            FingerprintConflictError: If another worker holds an active claim.
            IntegrityError: If unrelated database constraints fail.
        """
        if not isinstance(stage, PipelineStage):
            raise TypeError(
                f"stage must be a PipelineStage enum member, got: {type(stage).__name__}"
            )

        if not isinstance(artifact_kind, ArtifactKind):
            raise TypeError(
                f"artifact_kind must be an ArtifactKind enum member, got: {type(artifact_kind).__name__}"
            )

        if not _FINGERPRINT_PATTERN.match(stage_fingerprint):
            raise ValueError(
                f"stage_fingerprint must be a 64-char lowercase hex string, "
                f"got: '{stage_fingerprint[:20]}...'"
            )

        actual_art_id = artifact_id or uuid.uuid4()
        canonical_path = self._store.build_path(workflow_id, stage, actual_art_id, extension)

        if storage_path is not None:
            user_path = Path(storage_path)
            self._store._validate_confinement(user_path)
            if user_path.resolve() != canonical_path.resolve():
                raise SecurityViolationError(
                    f"Storage path '{storage_path}' does not match canonical layout '{canonical_path}'"
                )
            final_storage_path = str(user_path.resolve())
        else:
            final_storage_path = str(canonical_path)

        artifact = ArtifactModel(
            id=str(actual_art_id),
            workflow_id=str(workflow_id),
            job_id=str(job_id),
            artifact_type=artifact_kind.value,
            artifact_kind=artifact_kind.value,
            storage_path=final_storage_path,
            mime_type=mime_type,
            lifecycle_status=ArtifactLifecycleStatus.PENDING.value,
            stage=stage.value,
            stage_fingerprint=stage_fingerprint,
        )

        try:
            async with self._session.begin_nested():
                self._session.add(artifact)
                await self._session.flush()
        except IntegrityError as exc:
            exc_str = str(exc)
            orig = getattr(exc, "orig", None)
            constraint = getattr(orig, "constraint_name", None)
            if (
                constraint == "uq_artifacts_active_fingerprint"
                or "uq_artifacts_active_fingerprint" in exc_str
            ):
                raise FingerprintConflictError(
                    workflow_id=str(workflow_id),
                    stage=stage.value,
                    stage_fingerprint=stage_fingerprint,
                ) from exc
            raise

        return artifact

    async def finalize(
        self,
        artifact_id: UUID,
        *,
        worker_id: str | None = None,
        lease_token: str | None = None,
        lease_timeout_seconds: float = 60.0,
    ) -> ArtifactModel:
        """Transition artifact PENDING → AVAILABLE with independent verification and mandatory lease fencing.

        Separates slow streaming file verification from the short database row lock:
        1. Reads artifact without lock to load storage_path and verify PENDING status.
        2. Enforces mandatory worker credentials if the artifact is associated with a job.
           Omission of credentials on job-associated artifacts is strictly prohibited.
        3. If an artifact is in FAILED status and a physical file exists, detects and quarantines
           the stale orphan file to prevent unauthoritative data residency before raising.
        4. Independently reads physical file and computes SHA-256 + size via ArtifactStore.compute_checksum()
           WITHOUT holding any database row lock.
        5. Acquires database row locks in strict global hierarchy:
           JobModel (FOR UPDATE) → JobAttemptModel (FOR UPDATE) → ArtifactModel (FOR UPDATE).
        6. Enforces strict lease ownership preconditions:
           - Referenced job must exist.
           - Job must have an active attempt.
           - Attempt must match worker_id, lease_token, be in CLAIMED/RUNNING status, and have a fresh heartbeat.
        7. Revalidates under lock that the record is still in PENDING status and that
           the physical file was not modified concurrently during hashing.
        8. Persists verified checksum, size, and AVAILABLE status, then flushes promptly.
        """
        if (
            not isinstance(lease_timeout_seconds, (int, float))
            or lease_timeout_seconds <= 0
            or not math.isfinite(lease_timeout_seconds)
        ):
            raise ValueError(
                f"lease_timeout_seconds must be a positive finite number, got: {lease_timeout_seconds}"
            )

        # Step 1: Check initial status and get storage path without holding DB row lock
        initial_artifact = await self._get_artifact(artifact_id, for_update=False)

        # Stale worker recovery quarantine: if artifact was already marked FAILED, quarantine orphan file
        if initial_artifact.lifecycle_status == ArtifactLifecycleStatus.FAILED.value:
            file_path = Path(initial_artifact.storage_path)
            if file_path.exists():
                await self.quarantine_stale_file(artifact_id)
            self._validate_transition(initial_artifact, ArtifactLifecycleStatus.AVAILABLE)

        self._validate_transition(initial_artifact, ArtifactLifecycleStatus.AVAILABLE)

        # Mandatory worker credentials for job-associated artifacts
        if initial_artifact.job_id and (
            not worker_id
            or not str(worker_id).strip()
            or not lease_token
            or not str(lease_token).strip()
        ):
            raise ValueError(
                f"worker_id and lease_token are mandatory to finalize job-associated artifact '{artifact_id}'. "
                f"Use finalize_administrative() if privileged operator bypass is intended."
            )

        file_path = Path(initial_artifact.storage_path)
        stat_before = file_path.stat()

        # Step 2: Stream and verify physical file without holding DB row lock
        checksum, file_size = await self._store.compute_checksum(file_path)

        # Step 3: Acquire row locks in strict global hierarchy: Job -> Attempt -> Artifact
        if initial_artifact.job_id:
            job_stmt = (
                select(JobModel).where(JobModel.id == initial_artifact.job_id).with_for_update()
            )
            job = (await self._session.execute(job_stmt)).scalar_one_or_none()
            if job is None:
                raise LeaseConflictError(
                    str(initial_artifact.job_id),
                    worker_id or "",
                    lease_token or "",
                    message=f"Cannot finalize artifact '{artifact_id}': referenced job '{initial_artifact.job_id}' does not exist",
                )
            if not job.current_attempt_id:
                raise LeaseConflictError(
                    str(job.id),
                    worker_id or "",
                    lease_token or "",
                    message=f"Cannot finalize artifact '{artifact_id}': job '{job.id}' has no current attempt",
                )

            att_stmt = (
                select(JobAttemptModel)
                .where(JobAttemptModel.id == job.current_attempt_id)
                .with_for_update()
            )
            attempt = (await self._session.execute(att_stmt)).scalar_one_or_none()
            if attempt is None:
                raise LeaseConflictError(
                    str(job.id),
                    worker_id or "",
                    lease_token or "",
                    message=f"Cannot finalize artifact '{artifact_id}': current attempt '{job.current_attempt_id}' for job '{job.id}' does not exist",
                )

            if (
                attempt.worker_id != worker_id
                or attempt.lease_token != lease_token
                or attempt.status not in (AttemptStatus.CLAIMED.value, AttemptStatus.RUNNING.value)
            ):
                raise LeaseConflictError(
                    str(job.id),
                    worker_id or "",
                    lease_token or "",
                    message=(
                        f"Cannot finalize artifact '{artifact_id}': worker '{worker_id}' with token "
                        f"'{lease_token}' does not own active lease on job '{job.id}' "
                        f"(status={attempt.status}, worker={attempt.worker_id})"
                    ),
                )

            age = utc_now() - attempt.heartbeat_at
            if age >= timedelta(seconds=lease_timeout_seconds):
                raise LeaseConflictError(
                    str(job.id),
                    worker_id or "",
                    lease_token or "",
                    message=(
                        f"Cannot finalize artifact '{artifact_id}': worker '{worker_id}' lease on job '{job.id}' "
                        f"has expired (heartbeat {age.total_seconds():.1f}s ago, lease_timeout={lease_timeout_seconds}s)"
                    ),
                )

        locked_artifact = await self._get_artifact(artifact_id, for_update=True)
        if locked_artifact.lifecycle_status == ArtifactLifecycleStatus.FAILED.value:
            if file_path.exists():
                await self.quarantine_stale_file(artifact_id)
            self._validate_transition(locked_artifact, ArtifactLifecycleStatus.AVAILABLE)

        self._validate_transition(locked_artifact, ArtifactLifecycleStatus.AVAILABLE)

        # Step 4: Verify physical file immutability (not mutated during hashing/locking)
        stat_after = file_path.stat()
        if stat_after.st_mtime_ns != stat_before.st_mtime_ns or stat_after.st_size != file_size:
            raise SecurityViolationError(
                f"Artifact file '{file_path}' was modified concurrently during finalization"
            )

        # Step 5: Persist verified metadata and transition to AVAILABLE
        locked_artifact.checksum = checksum
        locked_artifact.file_size_bytes = file_size
        locked_artifact.lifecycle_status = ArtifactLifecycleStatus.AVAILABLE.value
        await self._session.flush()

        return locked_artifact

    async def finalize_administrative(
        self,
        artifact_id: UUID,
        *,
        admin_reason: str,
    ) -> ArtifactModel:
        """Administrative finalization bypassing worker lease verification.

        Used explicitly for operator recovery, migration repairs, or trusted admin tasks.
        Requires an explicit non-empty admin_reason which is recorded in the artifact metadata.
        Maintains the strict global lock hierarchy (Job -> Attempt -> Artifact) and independent
        physical file checksum and immutability verification.
        """
        if not admin_reason or not admin_reason.strip():
            raise ValueError(
                "admin_reason must be a non-empty string for administrative finalization"
            )

        logger.warning(
            "Privileged administrative finalization invoked for artifact '%s': %s",
            artifact_id,
            admin_reason,
        )

        initial_artifact = await self._get_artifact(artifact_id, for_update=False)
        self._validate_transition(initial_artifact, ArtifactLifecycleStatus.AVAILABLE)

        file_path = Path(initial_artifact.storage_path)
        stat_before = file_path.stat()

        checksum, file_size = await self._store.compute_checksum(file_path)

        # Global hierarchy: Job -> Attempt -> Artifact
        if initial_artifact.job_id:
            job_stmt = (
                select(JobModel).where(JobModel.id == initial_artifact.job_id).with_for_update()
            )
            job = (await self._session.execute(job_stmt)).scalar_one_or_none()
            if job is not None and job.current_attempt_id:
                att_stmt = (
                    select(JobAttemptModel)
                    .where(JobAttemptModel.id == job.current_attempt_id)
                    .with_for_update()
                )
                await self._session.execute(att_stmt)

        locked_artifact = await self._get_artifact(artifact_id, for_update=True)
        self._validate_transition(locked_artifact, ArtifactLifecycleStatus.AVAILABLE)

        stat_after = file_path.stat()
        if stat_after.st_mtime_ns != stat_before.st_mtime_ns or stat_after.st_size != file_size:
            raise SecurityViolationError(
                f"Artifact file '{file_path}' was modified concurrently during administrative finalization"
            )

        locked_artifact.checksum = checksum
        locked_artifact.file_size_bytes = file_size
        locked_artifact.lifecycle_status = ArtifactLifecycleStatus.AVAILABLE.value
        metadata = dict(locked_artifact.artifact_metadata or {})
        metadata["administrative_finalization"] = {
            "reason": admin_reason,
            "timestamp": utc_now().isoformat(),
        }
        locked_artifact.artifact_metadata = metadata
        await self._session.flush()

        return locked_artifact

    async def quarantine_stale_file(self, artifact_id: UUID) -> Path | None:
        """Quarantine orphan physical file for a FAILED or superseded artifact.

        If a stale worker published a file to disk after the artifact's database record
        was marked FAILED, this method renames the physical file to a timestamped
        quarantine path ({canonical_path}.stale.{timestamp_ns}) and records the quarantine
        in the artifact metadata.

        Returns the quarantine Path if a stale file was quarantined, or None if no file existed.
        """
        initial_artifact = await self._get_artifact(artifact_id, for_update=False)
        if initial_artifact.lifecycle_status not in (
            ArtifactLifecycleStatus.FAILED.value,
            ArtifactLifecycleStatus.SUPERSEDED.value,
        ):
            raise InvalidArtifactLifecycleError(
                artifact_id=initial_artifact.id,
                current_status=initial_artifact.lifecycle_status,
                target_status="QUARANTINED",
            )

        file_path = Path(initial_artifact.storage_path)
        if not file_path.exists():
            return None

        # Lock Job -> Attempt -> Artifact in order
        if initial_artifact.job_id:
            job_stmt = (
                select(JobModel).where(JobModel.id == initial_artifact.job_id).with_for_update()
            )
            job = (await self._session.execute(job_stmt)).scalar_one_or_none()
            if job is not None and job.current_attempt_id:
                att_stmt = (
                    select(JobAttemptModel)
                    .where(JobAttemptModel.id == job.current_attempt_id)
                    .with_for_update()
                )
                await self._session.execute(att_stmt)

        locked_artifact = await self._get_artifact(artifact_id, for_update=True)
        if not file_path.exists():
            return None

        timestamp_ns = int(utc_now().timestamp() * 1e9)
        quarantine_path = file_path.with_name(f"{file_path.name}.stale.{timestamp_ns}")
        try:
            file_path.rename(quarantine_path)
        except OSError as exc:
            logger.error(
                "Failed to rename stale file %s to quarantine %s: %s",
                file_path,
                quarantine_path,
                exc,
            )
            return None

        logger.warning(
            "Quarantined stale orphan artifact file %s to %s for failed artifact %s",
            file_path,
            quarantine_path,
            locked_artifact.id,
        )
        metadata = dict(locked_artifact.artifact_metadata or {})
        quarantined = list(metadata.get("quarantined_stale_files", []))
        quarantined.append(str(quarantine_path))
        metadata["quarantined_stale_files"] = quarantined
        locked_artifact.artifact_metadata = metadata
        await self._session.flush()
        return quarantine_path

    async def reconcile_pending(
        self,
        artifact_id: UUID,
        *,
        expected_checksum: str | None = None,
        worker_id: str | None = None,
        lease_token: str | None = None,
        lease_timeout_seconds: float = 60.0,
    ) -> ArtifactModel:
        """Reconcile an artifact in PENDING status after crash or ambiguous publication.

        Separates slow streaming file verification from the short database row lock,
        and enforces strict lease-aware recovery semantics serialized against competing workers:
        1. Reads artifact metadata without lock to verify record and obtain storage path.
        2. If record is already in AVAILABLE, returns immediately (idempotent retry).
        3. If record is in FAILED status, detects whether a stale worker published an orphan
           file and safely quarantines it.
        4. Inspects physical file and streams SHA-256 checksum WITHOUT holding any database row lock.
        5. Acquires short row lock (FOR UPDATE) to serialize the transition and lock the owning job/attempt.
        6. Enforces Lease Ownership & Recovery Eligibility:
           - If owning job has an active attempt (CLAIMED/RUNNING):
             - If caller is the active lease holder (worker_id & lease_token match): reconciliation is permitted.
             - If caller is a third-party reconciler: checks if attempt heartbeat is older than lease_timeout_seconds.
               If lease is still active (not expired), raises LeaseConflictError: active leases cannot be abandoned!
               If lease has expired, recovery is permitted.
             - Under FOR UPDATE, if another worker renewed or took over the lease, the fresh heartbeat is seen,
               preventing stale recovery decisions from invalidating renewed claims.
        7. Revalidates under lock:
           - If another worker/reconciler already transitioned the record, returns idempotently.
           - If physical file is missing: transitions PENDING -> FAILED.
           - If physical file was modified concurrently during hashing: raises SecurityViolationError.
           - If expected_checksum mismatches: transitions PENDING -> FAILED (without deleting physical data).
           - If physical file is intact and valid: transitions PENDING -> AVAILABLE.
        """
        if (
            not isinstance(lease_timeout_seconds, (int, float))
            or lease_timeout_seconds <= 0
            or not math.isfinite(lease_timeout_seconds)
        ):
            raise ValueError(
                f"lease_timeout_seconds must be a positive finite number, got: {lease_timeout_seconds}"
            )

        # Step 1: Read artifact without lock to check initial status and storage path
        initial_artifact = await self._get_artifact(artifact_id, for_update=False)
        if initial_artifact.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value:
            return initial_artifact
        if initial_artifact.lifecycle_status == ArtifactLifecycleStatus.FAILED.value:
            file_path = Path(initial_artifact.storage_path)
            if file_path.exists():
                await self.quarantine_stale_file(artifact_id)
            return initial_artifact
        if initial_artifact.lifecycle_status != ArtifactLifecycleStatus.PENDING.value:
            return initial_artifact

        file_path = Path(initial_artifact.storage_path)
        file_exists = file_path.exists()
        checksum: str | None = None
        file_size: int | None = None
        stat_before = None

        # Step 2: Stream checksum outside lock if file is present
        if file_exists:
            stat_before = file_path.stat()
            checksum, file_size = await self._store.compute_checksum(file_path)

        # Step 3: Acquire row locks in global hierarchy: Job -> Attempt -> Artifact
        if initial_artifact.job_id:
            job_stmt = (
                select(JobModel).where(JobModel.id == initial_artifact.job_id).with_for_update()
            )
            job = (await self._session.execute(job_stmt)).scalar_one_or_none()
            if job is not None and job.current_attempt_id:
                att_stmt = (
                    select(JobAttemptModel)
                    .where(JobAttemptModel.id == job.current_attempt_id)
                    .with_for_update()
                )
                attempt = (await self._session.execute(att_stmt)).scalar_one_or_none()
                if attempt is not None and attempt.status in (
                    AttemptStatus.CLAIMED.value,
                    AttemptStatus.RUNNING.value,
                ):
                    is_owner = (
                        worker_id is not None
                        and lease_token is not None
                        and attempt.worker_id == worker_id
                        and attempt.lease_token == lease_token
                    )
                    if not is_owner:
                        age = utc_now() - attempt.heartbeat_at
                        if age < timedelta(seconds=lease_timeout_seconds):
                            raise LeaseConflictError(
                                str(job.id),
                                worker_id,
                                lease_token,
                                message=(
                                    f"Cannot reconcile artifact '{artifact_id}': owning job '{job.id}' is "
                                    f"actively leased by worker '{attempt.worker_id}' (heartbeat {age.total_seconds():.1f}s ago, "
                                    f"lease_timeout={lease_timeout_seconds}s)"
                                ),
                            )

        locked_artifact = await self._get_artifact(artifact_id, for_update=True)
        if locked_artifact.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value:
            return locked_artifact
        if locked_artifact.lifecycle_status == ArtifactLifecycleStatus.FAILED.value:
            if file_path.exists():
                await self.quarantine_stale_file(artifact_id)
            return locked_artifact
        if locked_artifact.lifecycle_status != ArtifactLifecycleStatus.PENDING.value:
            return locked_artifact

        # Step 5: Apply resolution based on verified physical file status
        if not file_exists:
            locked_artifact.lifecycle_status = ArtifactLifecycleStatus.FAILED.value
            metadata = locked_artifact.artifact_metadata or {}
            metadata["error_message"] = (
                "Reconciliation failed: physical artifact file does not exist"
            )
            locked_artifact.artifact_metadata = metadata
            await self._session.flush()
            return locked_artifact

        # Verify physical file was not mutated concurrently during hashing or locking
        assert stat_before is not None
        stat_after = file_path.stat()
        if stat_after.st_mtime_ns != stat_before.st_mtime_ns or stat_after.st_size != file_size:
            raise SecurityViolationError(
                f"Artifact file '{file_path}' was modified concurrently during reconciliation"
            )

        if expected_checksum is not None and checksum != expected_checksum:
            locked_artifact.lifecycle_status = ArtifactLifecycleStatus.FAILED.value
            metadata = locked_artifact.artifact_metadata or {}
            metadata["error_message"] = (
                f"Reconciliation failed: checksum mismatch (expected {expected_checksum}, got {checksum})"
            )
            locked_artifact.artifact_metadata = metadata
            await self._session.flush()
            return locked_artifact

        locked_artifact.checksum = checksum
        locked_artifact.file_size_bytes = file_size
        locked_artifact.lifecycle_status = ArtifactLifecycleStatus.AVAILABLE.value
        await self._session.flush()
        return locked_artifact

    async def mark_failed(
        self, artifact_id: UUID, error_message: str | None = None
    ) -> ArtifactModel:
        """Transition artifact PENDING → FAILED with row locking."""
        artifact = await self._get_artifact(artifact_id, for_update=True)
        self._validate_transition(artifact, ArtifactLifecycleStatus.FAILED)

        artifact.lifecycle_status = ArtifactLifecycleStatus.FAILED.value
        if error_message:
            metadata = artifact.artifact_metadata or {}
            metadata["error_message"] = error_message
            artifact.artifact_metadata = metadata
        await self._session.flush()

        return artifact

    async def supersede(self, artifact_id: UUID) -> ArtifactModel:
        """Transition artifact AVAILABLE → SUPERSEDED with row locking.

        Used ONLY when a new artifact with a DIFFERENT fingerprint replaces
        the logical output of a stage. NOT used for ordinary retries with
        the same fingerprint (those reuse the existing AVAILABLE artifact).
        """
        artifact = await self._get_artifact(artifact_id, for_update=True)
        self._validate_transition(artifact, ArtifactLifecycleStatus.SUPERSEDED)

        artifact.lifecycle_status = ArtifactLifecycleStatus.SUPERSEDED.value
        await self._session.flush()

        return artifact

    async def find_by_fingerprint(
        self,
        workflow_id: UUID,
        stage: PipelineStage,
        stage_fingerprint: str,
        artifact_kind: ArtifactKind | None = None,
    ) -> ArtifactModel | None:
        """Find an AVAILABLE artifact matching the idempotency fingerprint."""
        if not isinstance(stage, PipelineStage):
            raise TypeError(
                f"stage must be a PipelineStage enum member, got: {type(stage).__name__}"
            )

        stmt = (
            select(ArtifactModel)
            .where(
                ArtifactModel.workflow_id == str(workflow_id),
                ArtifactModel.stage == stage.value,
                ArtifactModel.stage_fingerprint == stage_fingerprint,
                ArtifactModel.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE.value,
            )
            .limit(1)
        )
        if artifact_kind is not None:
            if not isinstance(artifact_kind, ArtifactKind):
                raise TypeError(
                    f"artifact_kind must be an ArtifactKind enum member, got: {type(artifact_kind).__name__}"
                )
            stmt = stmt.where(ArtifactModel.artifact_kind == artifact_kind.value)

        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def read_and_verify(self, artifact: ArtifactModel) -> tuple[ArtifactModel, bytes]:
        """Read bytes and verify SHA-256 and size atomically for an ArtifactModel.

        Validates lifecycle status is AVAILABLE, file exists on disk, reads the bytes,
        and independently calculates SHA-256 and size against stored metadata.
        Returns the verified model and verified in-memory bytes, eliminating any
        time-of-check / time-of-use (TOCTOU) gap between verification and consumption.
        Fails closed: requires stored checksum and file_size_bytes to be present.
        """
        if artifact.lifecycle_status != ArtifactLifecycleStatus.AVAILABLE.value:
            raise ArtifactUnavailableError(
                str(artifact.id),
                f"expected AVAILABLE, got {artifact.lifecycle_status}",
            )

        file_path = Path(artifact.storage_path)
        if not file_path.exists():
            raise ArtifactUnavailableError(
                str(artifact.id),
                f"physical artifact file does not exist: {file_path}",
            )

        # Fail closed: reject missing or malformed stored integrity metadata
        if not artifact.checksum or not artifact.checksum.strip():
            raise SecurityViolationError(
                f"Integrity failure: artifact '{artifact.id}' missing or empty stored checksum"
            )
        if artifact.file_size_bytes is None or artifact.file_size_bytes < 0:
            raise SecurityViolationError(
                f"Integrity failure: artifact '{artifact.id}' missing or invalid stored file_size_bytes"
            )

        raw_bytes = await self._store.read_artifact(file_path)
        computed_checksum = hashlib.sha256(raw_bytes).hexdigest()
        computed_size = len(raw_bytes)

        if computed_checksum != artifact.checksum.strip():
            raise SecurityViolationError(
                f"Integrity failure: artifact '{artifact.id}' checksum mismatch "
                f"(stored: {artifact.checksum}, disk: {computed_checksum})"
            )
        if computed_size != artifact.file_size_bytes:
            raise SecurityViolationError(
                f"Integrity failure: artifact '{artifact.id}' file size mismatch "
                f"(stored: {artifact.file_size_bytes}, disk: {computed_size})"
            )

        return artifact, raw_bytes

    async def read_verified_artifact(self, artifact_id: UUID) -> tuple[ArtifactModel, bytes]:
        """Load an AVAILABLE artifact by ID, reading its bytes and verifying integrity atomically."""
        artifact = await self._session.get(ArtifactModel, str(artifact_id))
        if artifact is None:
            raise ArtifactUnavailableError(str(artifact_id), "artifact record not found")
        return await self.read_and_verify(artifact)

    async def get_available(
        self, artifact_id: UUID, *, verify_integrity: bool = False
    ) -> ArtifactModel:
        """Load a specific AVAILABLE artifact by ID.

        Verifies that:
        1. The database record exists and is in AVAILABLE lifecycle status.
        2. The physical file exists at the canonical storage path.
        3. If verify_integrity is True, independently computes the SHA-256 checksum and size
           and asserts they match the stored metadata, preventing silent data corruption.
        """
        if verify_integrity:
            artifact, _ = await self.read_verified_artifact(artifact_id)
            return artifact

        artifact = await self._session.get(ArtifactModel, str(artifact_id))
        if artifact is None:
            raise ArtifactUnavailableError(str(artifact_id), "artifact record not found")

        if artifact.lifecycle_status != ArtifactLifecycleStatus.AVAILABLE.value:
            raise ArtifactUnavailableError(
                str(artifact_id),
                f"expected AVAILABLE, got {artifact.lifecycle_status}",
            )

        file_path = Path(artifact.storage_path)
        if not file_path.exists():
            raise ArtifactUnavailableError(
                str(artifact_id),
                f"physical artifact file does not exist: {file_path}",
            )

        return artifact

    # --- Internal helpers ---

    async def _get_artifact(self, artifact_id: UUID, *, for_update: bool = False) -> ArtifactModel:
        """Load artifact by ID, optionally acquiring a row lock."""
        stmt = select(ArtifactModel).where(ArtifactModel.id == str(artifact_id))
        if for_update:
            stmt = stmt.with_for_update().execution_options(populate_existing=True)
        result = await self._session.execute(stmt)
        artifact = result.scalar_one_or_none()
        if artifact is None:
            raise ValueError(f"Artifact '{artifact_id}' not found")
        return artifact

    @staticmethod
    def _validate_transition(artifact: ArtifactModel, target: ArtifactLifecycleStatus) -> None:
        """Validate lifecycle transition, raising domain error on failure."""
        current = ArtifactLifecycleStatus(artifact.lifecycle_status)
        if current == target:
            raise InvalidArtifactLifecycleError(
                artifact_id=artifact.id,
                current_status=current.value,
                target_status=target.value,
            )
        try:
            validate_artifact_lifecycle_transition(current, target)
        except InvalidStateTransitionError:
            raise InvalidArtifactLifecycleError(
                artifact_id=artifact.id,
                current_status=current.value,
                target_status=target.value,
            ) from None


__all__ = ["StageArtifactRepository"]
