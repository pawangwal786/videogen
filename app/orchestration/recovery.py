"""Recovery worker for detecting expired leases and reconciling ambiguous submissions."""

import asyncio
from typing import Protocol

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.base import utc_now
from app.db.session import get_session_factory
from app.logging import get_logger
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
    ) -> str | None:
        """Query provider to find whether an operation was created.

        Returns operation ID if active, None if definitively not created,
        or raises ProviderReconciliationRequiredError if ambiguous.
        """
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

                    reconciled_op_id = None
                    confirmed_not_created = False

                    if reconciler is not None:
                        try:
                            reconciled_op_id = await reconciler.reconcile_submission(
                                attempt.submission_token,
                                attempt.provider_operation_id,
                            )
                            if reconciled_op_id is None:
                                confirmed_not_created = True
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "reconciliation_failed_ambiguous",
                                attempt_id=attempt.id,
                                job_id=job.id,
                                error=str(exc),
                            )

                    if reconciled_op_id is not None:
                        # Operation was found running on external provider; resume polling
                        attempt.provider_operation_id = reconciled_op_id
                        attempt.status = AttemptStatus.RUNNING.value
                        attempt.heartbeat_at = now
                        job.status = JobStatus.RUNNING.value
                        job.version += 1
                        rescheduled += 1
                        recovered_ids.append(job.id)
                    elif confirmed_not_created:
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
                        # Ambiguous: do not blindly generate duplicate; fail job with reconciliation required
                        attempt.status = AttemptStatus.EXPIRED.value
                        attempt.completed_at = now
                        job.status = JobStatus.FAILED.value
                        job.completed_at = now
                        job.error_code = "PROVIDER_RECONCILIATION_REQUIRED"
                        job.error_message = (
                            f"Worker crashed during ambiguous submission. "
                            f"Submission token: {attempt.submission_token}. Requires manual/provider reconciliation."
                        )
                        job.version += 1
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
