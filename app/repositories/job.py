"""Job and JobAttempt repository handling atomic claims, leases, and lifecycle."""

import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.base import utc_now
from app.db.models.job import JobAttemptModel, JobModel
from app.orchestration.errors import (
    JobNotFoundError,
    LeaseConflictError,
)
from app.orchestration.state_machine import (
    AttemptStatus,
    JobStatus,
    validate_attempt_transition,
    validate_job_transition,
)


def is_unique_violation(exc: IntegrityError, constraint_hint: str | None = None) -> bool:
    """Check if IntegrityError is specifically a PostgreSQL unique_violation (code 23505)."""
    orig = getattr(exc, "orig", None)
    if orig is None:
        return False
    sqlstate = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
    if str(sqlstate) != "23505":
        return False
    if constraint_hint:
        diag = getattr(orig, "diag", None)
        constraint_name = getattr(diag, "constraint_name", None) or getattr(orig, "constraint_name", None)
        if constraint_name and constraint_hint not in constraint_name:
            return False
        if not constraint_name and constraint_hint not in str(orig):
            return False
    return True


class JobRepository:
    """Repository managing jobs and execution attempts with atomic locking and lease tokens."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_logical_key(
        self,
        workflow_id: str,
        logical_key: str,
    ) -> JobModel | None:
        """Look up a job by its workflow_id and logical_key."""
        stmt = select(JobModel).where(
            JobModel.workflow_id == workflow_id,
            JobModel.logical_key == logical_key,
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def create_job(
        self,
        workflow_id: str,
        logical_key: str,
        job_type: str,
        stage: str,
        input_payload: dict[str, Any],
        max_attempts: int = 3,
        available_at: datetime | None = None,
    ) -> JobModel:
        """Create a job idempotently. If (workflow_id, logical_key) exists, return existing."""
        existing = await self.get_by_logical_key(workflow_id, logical_key)
        if existing is not None:
            return existing

        job = JobModel(
            workflow_id=workflow_id,
            logical_key=logical_key,
            job_type=job_type,
            stage=stage,
            input_payload=input_payload,
            max_attempts=max_attempts,
            available_at=available_at or utc_now(),
            status=JobStatus.PENDING.value,
        )
        try:
            async with self._session.begin_nested():
                self._session.add(job)
                await self._session.flush()
        except IntegrityError as exc:
            if is_unique_violation(exc, "uq_jobs_workflow_logical_key"):
                existing = await self.get_by_logical_key(workflow_id, logical_key)
                if existing is not None:
                    return existing
            raise exc
        return job

    async def claim_next_job(
        self,
        worker_id: str,
        lease_token: str | None = None,
    ) -> tuple[JobModel, JobAttemptModel] | None:
        """Atomically claim the next eligible PENDING job using SELECT FOR UPDATE SKIP LOCKED."""
        now = utc_now()
        stmt = (
            select(JobModel)
            .where(
                JobModel.status == JobStatus.PENDING.value,
                JobModel.available_at <= now,
            )
            .order_by(JobModel.available_at.asc(), JobModel.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(stmt)
        job = result.scalar_one_or_none()
        if job is None:
            return None

        # Determine attempt number
        max_att_stmt = select(func.max(JobAttemptModel.attempt_number)).where(JobAttemptModel.job_id == job.id)
        current_max = (await self._session.execute(max_att_stmt)).scalar() or 0
        attempt_number = current_max + 1

        # Check if attempts exceeded max allowed
        if attempt_number > job.max_attempts:
            job.status = JobStatus.FAILED.value
            job.error_code = "MAX_ATTEMPTS_EXCEEDED"
            job.error_message = f"Exceeded max attempts ({current_max}/{job.max_attempts})"
            job.completed_at = now
            await self._session.flush()
            # Attempt next available job
            return await self.claim_next_job(worker_id, lease_token)

        token = lease_token or str(uuid.uuid4())
        submission_token = f"{job.workflow_id}:{job.logical_key}:{attempt_number}"

        attempt = JobAttemptModel(
            job_id=job.id,
            attempt_number=attempt_number,
            worker_id=worker_id,
            lease_token=token,
            status=AttemptStatus.CLAIMED.value,
            started_at=now,
            heartbeat_at=now,
            submission_token=submission_token,
            request_payload=job.input_payload,
        )
        self._session.add(attempt)
        await self._session.flush()

        job.status = JobStatus.CLAIMED.value
        job.current_attempt_id = attempt.id
        if job.started_at is None:
            job.started_at = now
        job.version += 1
        await self._session.flush()

        return job, attempt

    async def get_active_attempt(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
    ) -> JobAttemptModel:
        """Verify and return the active attempt matching job_id, worker_id, and lease_token."""
        stmt = select(JobAttemptModel).where(
            JobAttemptModel.job_id == job_id,
            JobAttemptModel.worker_id == worker_id,
            JobAttemptModel.lease_token == lease_token,
            JobAttemptModel.status.in_([
                AttemptStatus.CLAIMED.value,
                AttemptStatus.SUBMISSION_PENDING.value,
                AttemptStatus.RUNNING.value,
            ]),
        )
        attempt = (await self._session.execute(stmt)).scalar_one_or_none()
        if attempt is None:
            raise LeaseConflictError(job_id, worker_id, lease_token)
        return attempt

    async def heartbeat_attempt(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
    ) -> JobAttemptModel:
        """Update heartbeat timestamp for the worker's active lease."""
        attempt = await self.get_active_attempt(job_id, worker_id, lease_token)
        attempt.heartbeat_at = utc_now()
        await self._session.flush()
        return attempt

    async def record_submission_pending(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        provider: str,
        request_payload: dict[str, Any] | None = None,
    ) -> JobAttemptModel:
        """Mark attempt SUBMISSION_PENDING before dispatching an external side-effect."""
        attempt = await self.get_active_attempt(job_id, worker_id, lease_token)
        validate_attempt_transition(AttemptStatus(attempt.status), AttemptStatus.SUBMISSION_PENDING)
        attempt.status = AttemptStatus.SUBMISSION_PENDING.value
        attempt.provider = provider
        if request_payload is not None:
            attempt.request_payload = request_payload
        attempt.heartbeat_at = utc_now()
        await self._session.flush()
        return attempt

    async def record_submitted(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        provider_operation_id: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> JobAttemptModel:
        """Mark attempt and job RUNNING after provider has accepted the operation and returned an ID."""
        attempt = await self.get_active_attempt(job_id, worker_id, lease_token)
        attempt.provider_operation_id = provider_operation_id
        attempt.status = AttemptStatus.RUNNING.value
        if response_metadata is not None:
            attempt.response_metadata = response_metadata
        attempt.heartbeat_at = utc_now()

        job = await self.get_job(job_id)
        if job is not None:
            job.status = JobStatus.RUNNING.value

        await self._session.flush()
        return attempt

    async def complete_job(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        output_payload: dict[str, Any],
        response_metadata: dict[str, Any] | None = None,
    ) -> tuple[JobModel, JobAttemptModel]:
        """Mark attempt and job as COMPLETED and persist final output payload."""
        job = await self.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        attempt = await self.get_active_attempt(job_id, worker_id, lease_token)

        now = utc_now()
        validate_attempt_transition(AttemptStatus(attempt.status), AttemptStatus.COMPLETED)
        validate_job_transition(JobStatus(job.status), JobStatus.COMPLETED)

        attempt.status = AttemptStatus.COMPLETED.value
        attempt.completed_at = now
        if response_metadata is not None:
            attempt.response_metadata = response_metadata

        job.status = JobStatus.COMPLETED.value
        job.completed_at = now
        job.output_payload = output_payload
        job.version += 1

        await self._session.flush()
        return job, attempt

    async def fail_job(
        self,
        job_id: str,
        worker_id: str,
        lease_token: str,
        error_code: str,
        error_message: str,
        retryable: bool = False,
        next_available_at: datetime | None = None,
    ) -> tuple[JobModel, JobAttemptModel]:
        """Fail the current attempt, and either schedule the job for retry or mark it terminally FAILED."""
        job = await self.get_job(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        attempt = await self.get_active_attempt(job_id, worker_id, lease_token)

        now = utc_now()
        validate_attempt_transition(AttemptStatus(attempt.status), AttemptStatus.FAILED)
        attempt.status = AttemptStatus.FAILED.value
        attempt.completed_at = now
        attempt.error_code = error_code
        attempt.error_message = error_message

        if retryable and attempt.attempt_number < job.max_attempts:
            job.status = JobStatus.PENDING.value
            job.available_at = next_available_at or now
            job.version += 1
        else:
            validate_job_transition(JobStatus(job.status), JobStatus.FAILED)
            job.status = JobStatus.FAILED.value
            job.completed_at = now
            job.error_code = error_code
            job.error_message = error_message
            job.version += 1

        await self._session.flush()
        return job, attempt

    async def get_job(self, job_id: str, load_attempts: bool = False) -> JobModel | None:
        """Retrieve a job by ID."""
        stmt = select(JobModel).where(JobModel.id == job_id)
        if load_attempts:
            stmt = stmt.options(selectinload(JobModel.attempts))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_jobs_for_workflow(self, workflow_id: str) -> list[JobModel]:
        """List all jobs belonging to a workflow ordered by creation time."""
        stmt = (
            select(JobModel)
            .where(JobModel.workflow_id == workflow_id)
            .order_by(JobModel.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_provider_operation_id(self, job_id: str) -> str | None:
        """Find the most recent non-null provider_operation_id across attempts for this job."""
        stmt = (
            select(JobAttemptModel.provider_operation_id)
            .where(
                JobAttemptModel.job_id == job_id,
                JobAttemptModel.provider_operation_id.is_not(None),
            )
            .order_by(JobAttemptModel.attempt_number.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_expired_leases(self, lease_timeout_seconds: float) -> list[JobAttemptModel]:
        """Locate all active job attempts whose heartbeat has exceeded lease_timeout_seconds."""
        cutoff = utc_now() - timedelta(seconds=lease_timeout_seconds)
        stmt = (
            select(JobAttemptModel)
            .where(
                JobAttemptModel.status.in_([
                    AttemptStatus.CLAIMED.value,
                    AttemptStatus.SUBMISSION_PENDING.value,
                    AttemptStatus.RUNNING.value,
                ]),
                JobAttemptModel.heartbeat_at < cutoff,
            )
            .order_by(JobAttemptModel.heartbeat_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
