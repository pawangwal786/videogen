"""State machine definitions, enums, and transition validation."""

from enum import StrEnum

from app.orchestration.errors import InvalidStateTransitionError


class WorkflowStatus(StrEnum):
    """Lifecycle status of an end-to-end workflow."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStatus(StrEnum):
    """Lifecycle status of a logical pipeline job."""

    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AttemptStatus(StrEnum):
    """Lifecycle status of a single execution attempt of a job."""

    CLAIMED = "CLAIMED"
    SUBMISSION_PENDING = "SUBMISSION_PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class JobStage(StrEnum):
    """Stages in the video generation pipeline."""

    RESEARCH = "RESEARCH"
    SCRIPT = "SCRIPT"
    STORYBOARD = "STORYBOARD"
    VIDEO_GENERATION = "VIDEO_GENERATION"
    MEDIA_ASSEMBLY = "MEDIA_ASSEMBLY"
    COMPLETED = "COMPLETED"


class ArtifactLifecycleStatus(StrEnum):
    """Lifecycle status of generated artifacts."""

    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    PURGED = "PURGED"
    FAILED = "FAILED"


# Valid state transitions
_WORKFLOW_TRANSITIONS: dict[WorkflowStatus, set[WorkflowStatus]] = {
    WorkflowStatus.PENDING: {
        WorkflowStatus.RUNNING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.RUNNING: {
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.COMPLETED: set(),
    WorkflowStatus.FAILED: set(),
    WorkflowStatus.CANCELLED: set(),
}

_JOB_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.CLAIMED, JobStatus.CANCELLED},
    JobStatus.CLAIMED: {
        JobStatus.RUNNING,
        JobStatus.COMPLETED,
        JobStatus.PENDING,  # Reclaimed or lease expired
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.PENDING,  # Retryable failure resets to PENDING
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}

_ATTEMPT_TRANSITIONS: dict[AttemptStatus, set[AttemptStatus]] = {
    AttemptStatus.CLAIMED: {
        AttemptStatus.SUBMISSION_PENDING,
        AttemptStatus.RUNNING,
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
        AttemptStatus.EXPIRED,
        AttemptStatus.CANCELLED,
    },
    AttemptStatus.SUBMISSION_PENDING: {
        AttemptStatus.RUNNING,
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
        AttemptStatus.EXPIRED,
    },
    AttemptStatus.RUNNING: {
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
        AttemptStatus.EXPIRED,
        AttemptStatus.CANCELLED,
    },
    AttemptStatus.COMPLETED: set(),
    AttemptStatus.FAILED: set(),
    AttemptStatus.EXPIRED: set(),
    AttemptStatus.CANCELLED: set(),
}


def validate_workflow_transition(current: WorkflowStatus, next_status: WorkflowStatus) -> None:
    """Ensure the requested workflow state transition is allowed."""
    if current == next_status:
        return
    allowed = _WORKFLOW_TRANSITIONS.get(current, set())
    if next_status not in allowed:
        raise InvalidStateTransitionError("Workflow", current.value, next_status.value)


def validate_job_transition(current: JobStatus, next_status: JobStatus) -> None:
    """Ensure the requested job state transition is allowed."""
    if current == next_status:
        return
    allowed = _JOB_TRANSITIONS.get(current, set())
    if next_status not in allowed:
        raise InvalidStateTransitionError("Job", current.value, next_status.value)


def validate_attempt_transition(current: AttemptStatus, next_status: AttemptStatus) -> None:
    """Ensure the requested attempt state transition is allowed."""
    if current == next_status:
        return
    allowed = _ATTEMPT_TRANSITIONS.get(current, set())
    if next_status not in allowed:
        raise InvalidStateTransitionError("JobAttempt", current.value, next_status.value)


def is_terminal_workflow_status(status: WorkflowStatus) -> bool:
    """Return True if workflow is in a terminal state."""
    return status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}


def is_terminal_job_status(status: JobStatus) -> bool:
    """Return True if job is in a terminal state."""
    return status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}


def is_terminal_attempt_status(status: AttemptStatus) -> bool:
    """Return True if attempt is in a terminal state."""
    return status in {
        AttemptStatus.COMPLETED,
        AttemptStatus.FAILED,
        AttemptStatus.EXPIRED,
        AttemptStatus.CANCELLED,
    }
