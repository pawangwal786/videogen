"""Recovery worker for detecting expired leases and reconciling ambiguous submissions."""

import asyncio
from typing import Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import utc_now
from app.db.session import get_session_factory
from app.logging import get_logger
from app.models.errors import ModelRateLimitError, ModelTimeoutError
from app.orchestration.models import ReconciliationOutcome, ReconciliationStatus
from app.orchestration.retry import compute_next_available_at
from app.orchestration.state_machine import AttemptStatus, JobStatus, WorkflowStatus
from app.repositories.job import JobRepository
from app.repositories.workflow import WorkflowRepository

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
    ) -> None:
        self._session_factory = session_factory or get_session_factory()
        self._lease_timeout_seconds = lease_timeout_seconds
        self._reconcilers = reconcilers or {}
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def register_reconciler(self, provider: str, reconciler: ProviderReconciler) -> None:
        """Register a provider reconciler for ambiguous attempt resolution."""
        self._reconcilers[provider] = reconciler

    async def run_once(self) -> RecoveryReport:
        """Execute a single scan for expired leases and perform recovery transitions."""
        async with self._session_factory() as session:
            job_repo = JobRepository(session)
            expired_attempts = await job_repo.find_expired_leases(self._lease_timeout_seconds)

            if not expired_attempts:
                return RecoveryReport()

            rescheduled = 0
            failed = 0
            ambiguous = 0
            recovered_ids: list[str] = []

            for attempt in expired_attempts:
                job = await job_repo.get_job(attempt.job_id)
                if job is None or job.status in {
                    JobStatus.COMPLETED.value,
                    JobStatus.CANCELLED.value,
                }:
                    continue

                now = utc_now()

                # Ambiguous external submission recovery
                if attempt.status == AttemptStatus.SUBMISSION_PENDING.value:
                    provider = attempt.provider
                    reconciler = self._reconcilers.get(provider) if provider else None

                    outcome: ReconciliationOutcome | None = None
                    transient_failure = False

                    if reconciler is not None:
                        try:
                            outcome = await reconciler.reconcile_submission(
                                attempt.submission_token,
                                attempt.provider_operation_id,
                            )
                        except (ModelTimeoutError, ModelRateLimitError, TimeoutError, asyncio.TimeoutError) as exc:
                            logger.warning(
                                "reconciliation_transient_error",
                                attempt_id=attempt.id,
                                job_id=job.id,
                                error=str(exc),
                            )
                            transient_failure = True
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "reconciliation_failed_unresolved",
                                attempt_id=attempt.id,
                                job_id=job.id,
                                error=str(exc),
                            )
                            outcome = ReconciliationOutcome(
                                status=ReconciliationStatus.UNRESOLVED,
                                provider_operation_id=attempt.provider_operation_id,
                                error_message=str(exc),
                            )

                    if transient_failure:
                        # Leave attempt as SUBMISSION_PENDING so subsequent recovery iteration can retry
                        continue

                    if outcome is not None and outcome.status == ReconciliationStatus.RESOLVED:
                        # Operation was found running on external provider; resume polling
                        attempt.provider_operation_id = outcome.provider_operation_id
                        attempt.status = AttemptStatus.RUNNING.value
                        attempt.heartbeat_at = now
                        job.status = JobStatus.RUNNING.value
                        job.version += 1
                        rescheduled += 1
                        recovered_ids.append(job.id)
                    elif outcome is not None and outcome.status == ReconciliationStatus.CONFIRMED_ABSENT:
                        # Confirmed that provider never received it; safe to mark attempt expired and reschedule
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
                            job.error_code = "MAX_ATTEMPTS_EXCEEDED"
                            job.error_message = f"Exceeded max attempts ({attempt.attempt_number}/{job.max_attempts})"
                            job.version += 1
                            failed += 1
                    else:
                        # Ambiguous / UNRESOLVED: do not blindly generate duplicate; zero automated retries!
                        error_code = "AMBIGUOUS_SUBMISSION_REQUIRES_MANUAL_RECONCILIATION"
                        error_message = (
                            f"Worker crashed during ambiguous submission. "
                            f"Submission token: {attempt.submission_token}. Requires manual/provider reconciliation."
                        )
                        attempt.status = AttemptStatus.FAILED.value
                        attempt.completed_at = now
                        attempt.error_code = error_code
                        attempt.error_message = error_message

                        job.status = JobStatus.FAILED.value
                        job.completed_at = now
                        job.error_code = error_code
                        job.error_message = error_message
                        job.version += 1

                        wf_repo = WorkflowRepository(session)
                        try:
                            await wf_repo.update_status(
                                job.workflow_id,
                                status=WorkflowStatus.FAILED.value,
                                error_code=error_code,
                                error_message=error_message,
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.error("failed_to_update_workflow_status_on_ambiguity", error=str(exc))

                        ambiguous += 1
                        failed += 1

                else:
                    # Normal worker crash or missing heartbeat
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

            await session.commit()
            return RecoveryReport(
                expired_attempts_detected=len(expired_attempts),
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
