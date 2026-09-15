"""Domain exception hierarchy for workflow orchestration."""

from typing import Any


class OrchestrationError(Exception):
    """Base exception for all orchestration domain errors."""

    def __init__(
        self, message: str, code: str = "ORCHESTRATION_ERROR", details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class InvalidStateTransitionError(OrchestrationError):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, entity_type: str, current_state: str, target_state: str) -> None:
        message = (
            f"Invalid {entity_type} state transition from '{current_state}' to '{target_state}'."
        )
        super().__init__(
            message,
            code="INVALID_TRANSITION",
            details={
                "entity_type": entity_type,
                "current_state": current_state,
                "target_state": target_state,
            },
        )
        self.entity_type = entity_type
        self.current_state = current_state
        self.target_state = target_state


class WorkflowNotFoundError(OrchestrationError):
    """Raised when a requested workflow does not exist."""

    def __init__(self, workflow_id: str) -> None:
        super().__init__(
            f"Workflow with ID '{workflow_id}' was not found.",
            code="WORKFLOW_NOT_FOUND",
            details={"workflow_id": workflow_id},
        )
        self.workflow_id = workflow_id


class JobNotFoundError(OrchestrationError):
    """Raised when a requested job does not exist."""

    def __init__(self, job_id: str) -> None:
        super().__init__(
            f"Job with ID '{job_id}' was not found.",
            code="JOB_NOT_FOUND",
            details={"job_id": job_id},
        )
        self.job_id = job_id


class LeaseConflictError(OrchestrationError):
    """Raised when a lease heartbeat or completion check fails due to ownership conflict."""

    def __init__(self, job_id: str, worker_id: str | None, lease_token: str | None) -> None:
        super().__init__(
            f"Lease conflict for job '{job_id}': worker '{worker_id}' with lease token '{lease_token}' does not own the active lease.",
            code="LEASE_CONFLICT",
            details={"job_id": job_id, "worker_id": worker_id, "lease_token": lease_token},
        )
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_token = lease_token


class MaxAttemptsExceededError(OrchestrationError):
    """Raised when a job has reached or exceeded max execution attempts."""

    def __init__(self, job_id: str, attempts: int, max_attempts: int) -> None:
        super().__init__(
            f"Job '{job_id}' exceeded max attempts ({attempts}/{max_attempts}).",
            code="MAX_ATTEMPTS_EXCEEDED",
            details={"job_id": job_id, "attempts": attempts, "max_attempts": max_attempts},
        )
        self.job_id = job_id
        self.attempts = attempts
        self.max_attempts = max_attempts


class IdempotencyConflictError(OrchestrationError):
    """Raised when an idempotency key is reused with different parameters."""

    def __init__(self, idempotency_key: str, existing_topic: str, new_topic: str) -> None:
        super().__init__(
            f"Idempotency key '{idempotency_key}' conflict: existing topic '{existing_topic}' differs from requested '{new_topic}'.",
            code="IDEMPOTENCY_CONFLICT",
            details={
                "idempotency_key": idempotency_key,
                "existing_topic": existing_topic,
                "new_topic": new_topic,
            },
        )
        self.idempotency_key = idempotency_key


class ProviderReconciliationRequiredError(OrchestrationError):
    """Raised when an attempt is left in SUBMISSION_PENDING and requires external reconciliation."""

    def __init__(self, job_id: str, attempt_id: str, submission_token: str | None) -> None:
        super().__init__(
            f"Attempt '{attempt_id}' for job '{job_id}' is in SUBMISSION_PENDING. External provider reconciliation is required before retry.",
            code="PROVIDER_RECONCILIATION_REQUIRED",
            details={
                "job_id": job_id,
                "attempt_id": attempt_id,
                "submission_token": submission_token,
            },
        )
        self.job_id = job_id
        self.attempt_id = attempt_id
        self.submission_token = submission_token
