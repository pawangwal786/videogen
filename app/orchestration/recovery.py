"""Recovery worker for detecting expired leases and reconciling ambiguous submissions."""

import asyncio
import uuid
from datetime import timedelta
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import utc_now
from app.db.session import get_session_factory
from app.logging import get_logger
from app.models.errors import ModelRateLimitError, ModelTimeoutError
from app.orchestration.models import ReconciliationOutcome, ReconciliationStatus
from app.orchestration.orchestrator import WorkflowOrchestrator
from app.orchestration.retry import compute_next_available_at
from app.orchestration.state_machine import AttemptStatus, JobStatus
from app.repositories.job import JobRepository

logger = get_logger(__name__)


class ProviderReconciler(Protocol):
    """Protocol for provider-specific operation reconciliation."""

    async def reconcile_submission(
        self,
        submission_token: str | None,
        provider_operation_id: str | None,
    ) -> ReconciliationOutcome:
        """Query provider to find whether an operation was created."""
        ...


class RecoveryReport(BaseModel):
    """Summary of a single recovery cycle."""

    model_config = ConfigDict(frozen=True)

    expired_attempts_detected: int = 0
    jobs_rescheduled: int = 0
    jobs_terminally_failed: int = 0
    ambiguous_reconciliations: int = 0
    recovered_job_ids: list[str] = []


class RecoveryWorker:
    """Detects expired worker leases and safely recovers or reschedules stalled jobs.

    In accordance with architectural principles:
    - Ambiguous SUBMISSION_PENDING states are reconciled if provider supports it.
    - No duplicate generation is blindly submitted if provider side-effect is ambiguous.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        lease_timeout_seconds: float = 60.0,
        reconcilers: dict[str, ProviderReconciler] | None = None,
        worker_id: str | None = None,
        orchestrator: WorkflowOrchestrator | None = None,
        heartbeat_interval_seconds: float | None = None,
    ) -> None:
        self._session_factory = session_factory or get_session_factory()
        self._lease_timeout_seconds = lease_timeout_seconds
        self._reconcilers = reconcilers or {}
        self.worker_id = worker_id or f"recovery-{uuid.uuid4().hex[:8]}"
        self._orchestrator = orchestrator or WorkflowOrchestrator(self._session_factory)
        self._heartbeat_interval = heartbeat_interval_seconds or min(
            5.0, max(0.05, self._lease_timeout_seconds / 3)
        )
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def register_reconciler(self, provider: str, reconciler: ProviderReconciler) -> None:
        """Register a provider reconciler for ambiguous attempt resolution."""
        self._reconcilers[provider] = reconciler

    async def run_once(self) -> RecoveryReport:
        """Execute a single scan for expired leases and perform recovery transitions.

        Separates execution into three distinct phases:
        - Phase A: Fast DB transaction using SELECT ... FOR UPDATE SKIP LOCKED.
          Ordinary expired attempts are marked EXPIRED and rescheduled immediately.
          Attempts exceeding max_attempts are marked FAILED and advance the workflow.
          SUBMISSION_PENDING attempts are atomically claimed by establishing durable
          recovery ownership (worker_id + new recovery_lease_token + heartbeat_at = now).
        - Phase B: External provider reconciliation executed completely outside any DB session,
          protected by an active background recovery heartbeat task to prevent lease expiration.
        - Phase C: Fenced DB transaction applying the reconciliation outcome conditionally.
          If ownership was lost (0 rows affected), the advisory outcome is discarded.
          Terminal failures (max attempts or UNRESOLVED) advance the workflow to FAILED.
        """
        cutoff = utc_now() - timedelta(seconds=self._lease_timeout_seconds)
        rescheduled = 0
        failed = 0
        ambiguous = 0
        recovered_ids: list[str] = []
        expired_count = 0

        reconciliation_candidates: list[dict[str, Any]] = []
        phase_a_failed_workflows: set[str] = set()

        # ------------------------------------------------------------------
        # Phase A: Fast DB transaction claiming expired work
        # ------------------------------------------------------------------
        async with self._session_factory() as session:
            job_repo = JobRepository(session)
            expired_attempts = await job_repo.find_expired_leases(
                self._lease_timeout_seconds, for_update=True
            )

            if not expired_attempts:
                return RecoveryReport()

            expired_count = len(expired_attempts)
            now = utc_now()

            for attempt in expired_attempts:
                job = await job_repo.get_job(attempt.job_id)
                if job is None or job.status in {
                    JobStatus.COMPLETED.value,
                    JobStatus.CANCELLED.value,
                }:
                    continue

                if attempt.status == AttemptStatus.SUBMISSION_PENDING.value:
                    # Atomically claim for recovery by establishing durable ownership fencing
                    recovery_lease_token = str(uuid.uuid4())
                    claimed = await job_repo.claim_expired_attempt_for_recovery(
                        attempt_id=attempt.id,
                        recovery_worker_id=self.worker_id,
                        recovery_lease_token=recovery_lease_token,
                        cutoff=cutoff,
                    )
                    if claimed is not None:
                        reconciliation_candidates.append(
                            {
                                "attempt_id": attempt.id,
                                "job_id": job.id,
                                "workflow_id": job.workflow_id,
                                "max_attempts": job.max_attempts,
                                "attempt_number": attempt.attempt_number,
                                "provider": attempt.provider,
                                "submission_token": attempt.submission_token,
                                "provider_operation_id": attempt.provider_operation_id,
                                "recovery_lease_token": recovery_lease_token,
                            }
                        )
                else:
                    # Ordinary worker crash or missing heartbeat (CLAIMED or RUNNING)
                    attempt.status = AttemptStatus.EXPIRED.value
                    attempt.completed_at = now

                    if attempt.attempt_number < job.max_attempts:
                        job.status = JobStatus.PENDING.value
                        job.available_at = compute_next_available_at(attempt.attempt_number)
                        job.version += 1
                        rescheduled += 1
                        recovered_ids.append(job.id)
                    else:
                        job.status = JobStatus.FAILED.value
                        job.completed_at = now
                        job.error_code = "LEASE_EXPIRED"
                        job.error_message = (
                            f"Worker lease expired for worker '{attempt.worker_id}' "
                            f"and exceeded max attempts ({attempt.attempt_number}/{job.max_attempts})"
                        )
                        job.version += 1
                        failed += 1
                        phase_a_failed_workflows.add(job.workflow_id)

            await session.commit()

        # Advance workflows for jobs that terminally failed in Phase A
        for wf_id in phase_a_failed_workflows:
            try:
                await self._orchestrator.advance_workflow(wf_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "failed_to_advance_workflow_after_phase_a_failure",
                    workflow_id=wf_id,
                    error=str(exc),
                )

        if not reconciliation_candidates:
            return RecoveryReport(
                expired_attempts_detected=expired_count,
                jobs_rescheduled=rescheduled,
                jobs_terminally_failed=failed,
                ambiguous_reconciliations=ambiguous,
                recovered_job_ids=recovered_ids,
            )

        # ------------------------------------------------------------------
        # Phase B & Phase C: Decoupled reconciliation and fenced application
        # ------------------------------------------------------------------
        active_claims: dict[str, tuple[str, str, str]] = {
            c["attempt_id"]: (c["attempt_id"], self.worker_id, c["recovery_lease_token"])
            for c in reconciliation_candidates
        }
        stop_heartbeat = asyncio.Event()

        async def _recovery_heartbeat_loop() -> None:
            while not stop_heartbeat.is_set():
                try:
                    await asyncio.sleep(self._heartbeat_interval)
                    if stop_heartbeat.is_set() or not active_claims:
                        break
                    async with self._session_factory() as s:
                        repo = JobRepository(s)
                        for att_id, rec_w_id, rec_l_tok in list(active_claims.values()):
                            await repo.heartbeat_recovery_claim(
                                attempt_id=att_id,
                                recovery_worker_id=rec_w_id,
                                recovery_lease_token=rec_l_tok,
                            )
                        await s.commit()
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning("recovery_heartbeat_loop_error", error=str(exc))

        heartbeat_task = asyncio.create_task(_recovery_heartbeat_loop())

        try:
            for candidate in reconciliation_candidates:
                provider = candidate["provider"]
                reconciler = self._reconcilers.get(provider) if provider else None
                outcome: ReconciliationOutcome | None = None
                transient_failure = False

                if reconciler is not None:
                    try:
                        outcome = await reconciler.reconcile_submission(
                            candidate["submission_token"],
                            candidate["provider_operation_id"],
                        )
                    except (ModelTimeoutError, ModelRateLimitError, TimeoutError) as exc:
                        logger.warning(
                            "reconciliation_transient_error",
                            attempt_id=candidate["attempt_id"],
                            job_id=candidate["job_id"],
                            error=str(exc),
                        )
                        transient_failure = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "reconciliation_failed_unresolved",
                            attempt_id=candidate["attempt_id"],
                            job_id=candidate["job_id"],
                            error=str(exc),
                        )
                        # UNRESOLVED indicates that the reconciler could not safely establish the
                        # external state. To protect against duplicate generation / billing,
                        # this transitions to terminal failure requiring operator intervention.
                        outcome = ReconciliationOutcome(
                            status=ReconciliationStatus.UNRESOLVED,
                            provider_operation_id=candidate["provider_operation_id"],
                            error_message=str(exc),
                        )
                else:
                    outcome = ReconciliationOutcome(
                        status=ReconciliationStatus.UNRESOLVED,
                        provider_operation_id=candidate["provider_operation_id"],
                        error_message=f"No reconciler registered for provider '{provider}'",
                    )

                # Remove from active heartbeat before executing Phase C transaction
                active_claims.pop(candidate["attempt_id"], None)

                if transient_failure:
                    # Deliberately relinquish recovery ownership by stopping heartbeats.
                    # The attempt remains SUBMISSION_PENDING and will naturally expire its lease
                    # so that a future recovery cycle (or another worker) can reclaim and retry it.
                    continue

                # Phase C: Conditionally apply outcome using recovery ownership fencing
                phase_c_failed_wf: str | None = None
                now = utc_now()
                async with self._session_factory() as session:
                    job_repo = JobRepository(session)
                    job = await job_repo.get_job(candidate["job_id"])
                    if job is None:
                        continue

                    if outcome is not None and outcome.status == ReconciliationStatus.RESOLVED:
                        applied = await job_repo.apply_recovery_outcome(
                            attempt_id=candidate["attempt_id"],
                            recovery_worker_id=self.worker_id,
                            recovery_lease_token=candidate["recovery_lease_token"],
                            new_attempt_status=AttemptStatus.EXPIRED.value,
                            completed_at=now,
                            provider_operation_id=outcome.provider_operation_id,
                        )
                        if applied is None:
                            logger.warning(
                                "recovery_outcome_discarded_ownership_lost",
                                attempt_id=candidate["attempt_id"],
                            )
                            continue

                        # Reschedule job to PENDING with immediate availability for Worker B to claim
                        if candidate["attempt_number"] < candidate["max_attempts"]:
                            job.status = JobStatus.PENDING.value
                            job.available_at = now
                            job.version += 1
                            rescheduled += 1
                            recovered_ids.append(job.id)
                        else:
                            job.status = JobStatus.FAILED.value
                            job.completed_at = now
                            job.error_code = "MAX_ATTEMPTS_EXCEEDED"
                            job.error_message = f"Exceeded max attempts ({candidate['attempt_number']}/{candidate['max_attempts']})"
                            job.version += 1
                            failed += 1
                            phase_c_failed_wf = job.workflow_id

                    elif (
                        outcome is not None
                        and outcome.status == ReconciliationStatus.CONFIRMED_ABSENT
                    ):
                        applied = await job_repo.apply_recovery_outcome(
                            attempt_id=candidate["attempt_id"],
                            recovery_worker_id=self.worker_id,
                            recovery_lease_token=candidate["recovery_lease_token"],
                            new_attempt_status=AttemptStatus.EXPIRED.value,
                            completed_at=now,
                        )
                        if applied is None:
                            logger.warning(
                                "recovery_outcome_discarded_ownership_lost",
                                attempt_id=candidate["attempt_id"],
                            )
                            continue

                        if candidate["attempt_number"] < candidate["max_attempts"]:
                            job.status = JobStatus.PENDING.value
                            job.available_at = compute_next_available_at(
                                candidate["attempt_number"]
                            )
                            job.version += 1
                            rescheduled += 1
                            recovered_ids.append(job.id)
                        else:
                            job.status = JobStatus.FAILED.value
                            job.completed_at = now
                            job.error_code = "MAX_ATTEMPTS_EXCEEDED"
                            job.error_message = f"Exceeded max attempts ({candidate['attempt_number']}/{candidate['max_attempts']})"
                            job.version += 1
                            failed += 1
                            phase_c_failed_wf = job.workflow_id

                    else:
                        # UNRESOLVED: terminal failure with zero automated retry
                        error_code = "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"
                        error_message = (
                            f"Worker crashed during ambiguous submission. "
                            f"Submission token: {candidate['submission_token']}. Requires manual/provider reconciliation."
                        )
                        applied = await job_repo.apply_recovery_outcome(
                            attempt_id=candidate["attempt_id"],
                            recovery_worker_id=self.worker_id,
                            recovery_lease_token=candidate["recovery_lease_token"],
                            new_attempt_status=AttemptStatus.FAILED.value,
                            completed_at=now,
                            error_code=error_code,
                            error_message=error_message,
                        )
                        if applied is None:
                            logger.warning(
                                "recovery_outcome_discarded_ownership_lost",
                                attempt_id=candidate["attempt_id"],
                            )
                            continue

                        job.status = JobStatus.FAILED.value
                        job.completed_at = now
                        job.error_code = error_code
                        job.error_message = error_message
                        job.version += 1

                        ambiguous += 1
                        failed += 1
                        phase_c_failed_wf = job.workflow_id

                    await session.commit()

                # Advance workflow outside Phase C transaction if job reached terminal FAILED
                if phase_c_failed_wf is not None:
                    try:
                        await self._orchestrator.advance_workflow(phase_c_failed_wf)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "failed_to_advance_workflow_after_phase_c_failure",
                            workflow_id=phase_c_failed_wf,
                            error=str(exc),
                        )

        finally:
            stop_heartbeat.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        return RecoveryReport(
            expired_attempts_detected=expired_count,
            jobs_rescheduled=rescheduled,
            jobs_terminally_failed=failed,
            ambiguous_reconciliations=ambiguous,
            recovered_job_ids=recovered_ids,
        )

    async def start(self, interval_seconds: float = 10.0) -> None:
        """Start background polling loop for expired leases."""
        if self._running:
            return
        self._running = True

        async def _loop():
            while self._running:
                try:
                    await self.run_once()
                except Exception as exc:  # noqa: BLE001
                    logger.error("recovery_worker_loop_error", error=str(exc))
                await asyncio.sleep(interval_seconds)

        self._task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        """Stop background recovery worker loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
