"""Unit tests for orchestration state machines, domain models, and error hierarchy."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.models.job import JobModel
from app.db.models.workflow import WorkflowModel
from app.orchestration import (
    Artifact,
    ArtifactLifecycleStatus,
    AttemptStatus,
    ClaimedJob,
    IdempotencyConflictError,
    InvalidStateTransitionError,
    Job,
    JobAttempt,
    JobNotFoundError,
    JobStage,
    JobStatus,
    LeaseConflictError,
    MaxAttemptsExceededError,
    OrchestrationError,
    ProviderReconciliationRequiredError,
    Workflow,
    WorkflowNotFoundError,
    WorkflowStatus,
    is_terminal_attempt_status,
    is_terminal_job_status,
    is_terminal_workflow_status,
    validate_artifact_lifecycle_transition,
    validate_attempt_transition,
    validate_job_transition,
    validate_workflow_transition,
)
from app.orchestration.orchestrator import WorkflowOrchestrator


def test_workflow_state_transitions():
    # Valid transitions
    validate_workflow_transition(WorkflowStatus.PENDING, WorkflowStatus.RUNNING)
    validate_workflow_transition(WorkflowStatus.PENDING, WorkflowStatus.CANCELLED)
    validate_workflow_transition(WorkflowStatus.PENDING, WorkflowStatus.PENDING)  # no-op
    validate_workflow_transition(WorkflowStatus.RUNNING, WorkflowStatus.COMPLETED)
    validate_workflow_transition(WorkflowStatus.RUNNING, WorkflowStatus.FAILED)
    validate_workflow_transition(WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED)

    # Invalid transitions
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        validate_workflow_transition(WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING)
    assert exc_info.value.code == "INVALID_TRANSITION"
    assert exc_info.value.entity_type == "Workflow"

    with pytest.raises(InvalidStateTransitionError):
        validate_workflow_transition(WorkflowStatus.FAILED, WorkflowStatus.RUNNING)

    with pytest.raises(InvalidStateTransitionError):
        validate_workflow_transition(WorkflowStatus.PENDING, WorkflowStatus.COMPLETED)


def test_job_state_transitions():
    # Valid transitions
    validate_job_transition(JobStatus.PENDING, JobStatus.CLAIMED)
    validate_job_transition(JobStatus.PENDING, JobStatus.CANCELLED)
    validate_job_transition(JobStatus.CLAIMED, JobStatus.RUNNING)
    validate_job_transition(JobStatus.CLAIMED, JobStatus.PENDING)
    validate_job_transition(JobStatus.CLAIMED, JobStatus.FAILED)
    validate_job_transition(JobStatus.RUNNING, JobStatus.COMPLETED)
    validate_job_transition(JobStatus.RUNNING, JobStatus.PENDING)
    validate_job_transition(JobStatus.RUNNING, JobStatus.FAILED)

    # Invalid transitions
    with pytest.raises(InvalidStateTransitionError):
        validate_job_transition(JobStatus.COMPLETED, JobStatus.RUNNING)

    with pytest.raises(InvalidStateTransitionError):
        validate_job_transition(JobStatus.FAILED, JobStatus.CLAIMED)

    with pytest.raises(InvalidStateTransitionError):
        validate_job_transition(JobStatus.PENDING, JobStatus.COMPLETED)


def test_attempt_state_transitions():
    # Valid transitions
    validate_attempt_transition(AttemptStatus.CLAIMED, AttemptStatus.SUBMISSION_PENDING)
    validate_attempt_transition(AttemptStatus.CLAIMED, AttemptStatus.RUNNING)
    validate_attempt_transition(AttemptStatus.CLAIMED, AttemptStatus.FAILED)
    validate_attempt_transition(AttemptStatus.CLAIMED, AttemptStatus.EXPIRED)
    validate_attempt_transition(AttemptStatus.SUBMISSION_PENDING, AttemptStatus.RUNNING)
    validate_attempt_transition(AttemptStatus.SUBMISSION_PENDING, AttemptStatus.COMPLETED)
    validate_attempt_transition(AttemptStatus.SUBMISSION_PENDING, AttemptStatus.FAILED)
    validate_attempt_transition(AttemptStatus.SUBMISSION_PENDING, AttemptStatus.EXPIRED)
    validate_attempt_transition(AttemptStatus.RUNNING, AttemptStatus.COMPLETED)
    validate_attempt_transition(AttemptStatus.RUNNING, AttemptStatus.FAILED)
    validate_attempt_transition(AttemptStatus.RUNNING, AttemptStatus.EXPIRED)

    # Invalid transitions
    with pytest.raises(InvalidStateTransitionError):
        validate_attempt_transition(AttemptStatus.COMPLETED, AttemptStatus.RUNNING)

    with pytest.raises(InvalidStateTransitionError):
        validate_attempt_transition(AttemptStatus.FAILED, AttemptStatus.SUBMISSION_PENDING)


def test_terminal_predicates():
    assert is_terminal_workflow_status(WorkflowStatus.COMPLETED) is True
    assert is_terminal_workflow_status(WorkflowStatus.FAILED) is True
    assert is_terminal_workflow_status(WorkflowStatus.CANCELLED) is True
    assert is_terminal_workflow_status(WorkflowStatus.RUNNING) is False
    assert is_terminal_workflow_status(WorkflowStatus.PENDING) is False

    assert is_terminal_job_status(JobStatus.COMPLETED) is True
    assert is_terminal_job_status(JobStatus.FAILED) is True
    assert is_terminal_job_status(JobStatus.CANCELLED) is True
    assert is_terminal_job_status(JobStatus.RUNNING) is False

    assert is_terminal_attempt_status(AttemptStatus.COMPLETED) is True
    assert is_terminal_attempt_status(AttemptStatus.FAILED) is True
    assert is_terminal_attempt_status(AttemptStatus.EXPIRED) is True
    assert is_terminal_attempt_status(AttemptStatus.CANCELLED) is True
    assert is_terminal_attempt_status(AttemptStatus.RUNNING) is False
    assert is_terminal_attempt_status(AttemptStatus.SUBMISSION_PENDING) is False


def test_domain_errors():
    err = OrchestrationError("generic", code="CUSTOM_CODE", details={"k": "v"})
    assert err.code == "CUSTOM_CODE"
    assert err.details == {"k": "v"}
    assert str(err) == "generic"

    wf_err = WorkflowNotFoundError("wf-1")
    assert wf_err.code == "WORKFLOW_NOT_FOUND"
    assert wf_err.workflow_id == "wf-1"

    job_err = JobNotFoundError("job-1")
    assert job_err.code == "JOB_NOT_FOUND"
    assert job_err.job_id == "job-1"

    lease_err = LeaseConflictError("job-1", "worker-a", "token-a")
    assert lease_err.code == "LEASE_CONFLICT"
    assert lease_err.job_id == "job-1"
    assert lease_err.worker_id == "worker-a"
    assert lease_err.lease_token == "token-a"

    max_err = MaxAttemptsExceededError("job-1", 3, 3)
    assert max_err.code == "MAX_ATTEMPTS_EXCEEDED"
    assert max_err.attempts == 3
    assert max_err.max_attempts == 3

    idemp_err = IdempotencyConflictError("key-1", "topic-old", "topic-new")
    assert idemp_err.code == "IDEMPOTENCY_CONFLICT"
    assert idemp_err.idempotency_key == "key-1"

    recon_err = ProviderReconciliationRequiredError("job-1", "att-1", "sub-tok-1")
    assert recon_err.code == "PROVIDER_RECONCILIATION_REQUIRED"
    assert recon_err.job_id == "job-1"
    assert recon_err.attempt_id == "att-1"
    assert recon_err.submission_token == "sub-tok-1"


def test_domain_models():
    now = datetime.now(UTC)
    wf = Workflow(
        id=str(uuid.uuid4()),
        topic="Neural Networks",
        status=WorkflowStatus.RUNNING,
        current_stage=JobStage.RESEARCH,
        created_at=now,
        updated_at=now,
    )
    assert wf.topic == "Neural Networks"
    assert wf.status == WorkflowStatus.RUNNING

    att = JobAttempt(
        id=str(uuid.uuid4()),
        job_id=str(uuid.uuid4()),
        attempt_number=1,
        worker_id="w-1",
        lease_token="lease-1",
        status=AttemptStatus.RUNNING,
        started_at=now,
        heartbeat_at=now,
    )
    assert att.worker_id == "w-1"
    assert att.lease_token == "lease-1"

    job = Job(
        id=att.job_id,
        workflow_id=wf.id,
        logical_key="research",
        job_type="research",
        stage=JobStage.RESEARCH,
        status=JobStatus.RUNNING,
        available_at=now,
        created_at=now,
    )
    assert job.logical_key == "research"
    assert job.stage == JobStage.RESEARCH

    claimed = ClaimedJob(job=job, attempt=att)
    assert claimed.job.id == job.id
    assert claimed.attempt.id == att.id

    art = Artifact(
        id=str(uuid.uuid4()),
        workflow_id=wf.id,
        artifact_type="RESEARCH_REPORT",
        storage_path="/tmp/res.json",
        lifecycle_status=ArtifactLifecycleStatus.AVAILABLE,
        created_at=now,
        updated_at=now,
    )
    assert art.artifact_type == "RESEARCH_REPORT"
    assert art.lifecycle_status == ArtifactLifecycleStatus.AVAILABLE


def test_artifact_lifecycle_transitions():
    # Valid transitions
    validate_artifact_lifecycle_transition(
        ArtifactLifecycleStatus.PENDING, ArtifactLifecycleStatus.AVAILABLE
    )
    validate_artifact_lifecycle_transition(
        ArtifactLifecycleStatus.PENDING, ArtifactLifecycleStatus.FAILED
    )
    validate_artifact_lifecycle_transition(
        ArtifactLifecycleStatus.AVAILABLE, ArtifactLifecycleStatus.SUPERSEDED
    )
    validate_artifact_lifecycle_transition(
        ArtifactLifecycleStatus.PENDING, ArtifactLifecycleStatus.PENDING
    )  # no-op
    validate_artifact_lifecycle_transition(
        ArtifactLifecycleStatus.AVAILABLE, ArtifactLifecycleStatus.AVAILABLE
    )  # no-op

    # Invalid transitions
    with pytest.raises(InvalidStateTransitionError) as exc_info:
        validate_artifact_lifecycle_transition(
            ArtifactLifecycleStatus.AVAILABLE, ArtifactLifecycleStatus.PENDING
        )
    assert exc_info.value.code == "INVALID_TRANSITION"
    assert exc_info.value.entity_type == "ArtifactLifecycle"

    with pytest.raises(InvalidStateTransitionError):
        validate_artifact_lifecycle_transition(
            ArtifactLifecycleStatus.SUPERSEDED, ArtifactLifecycleStatus.AVAILABLE
        )

    with pytest.raises(InvalidStateTransitionError):
        validate_artifact_lifecycle_transition(
            ArtifactLifecycleStatus.FAILED, ArtifactLifecycleStatus.AVAILABLE
        )

    with pytest.raises(InvalidStateTransitionError):
        validate_artifact_lifecycle_transition(
            ArtifactLifecycleStatus.PURGED, ArtifactLifecycleStatus.AVAILABLE
        )

    with pytest.raises(InvalidStateTransitionError):
        validate_artifact_lifecycle_transition(
            ArtifactLifecycleStatus.AVAILABLE, ArtifactLifecycleStatus.FAILED
        )


@pytest.mark.asyncio
async def test_advance_workflow_creates_artifact_driven_script_job(monkeypatch):
    """Verify advance_workflow creates script jobs referencing durable artifacts, not embedded payloads."""
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock, MagicMock

    from app.db.models.job import JobModel
    from app.db.models.workflow import WorkflowModel
    from app.orchestration.orchestrator import WorkflowOrchestrator

    wf_id = str(uuid.uuid4())
    art_id = str(uuid.uuid4())
    checksum = "a" * 64
    now = datetime.now(UTC)

    wf_model = WorkflowModel(
        id=wf_id,
        topic="Quantum Computing",
        current_stage="RESEARCH",
        status="RUNNING",
        created_at=now,
        updated_at=now,
    )
    r_job = JobModel(
        id=str(uuid.uuid4()),
        workflow_id=wf_id,
        logical_key="research",
        job_type="research",
        stage="RESEARCH",
        status=JobStatus.COMPLETED.value,
        output_payload={"_artifact_id": art_id, "_checksum": checksum, "other": "ignored"},
    )

    mock_wf_repo = AsyncMock()
    mock_wf_repo.get_workflow_for_update.return_value = wf_model
    mock_wf_repo.get_workflow.return_value = wf_model

    mock_job_repo = AsyncMock()
    mock_job_repo.list_jobs_for_workflow.return_value = [r_job]

    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = WorkflowOrchestrator(session_factory=session_factory)
    monkeypatch.setattr("app.orchestration.orchestrator.WorkflowRepository", lambda s: mock_wf_repo)
    monkeypatch.setattr("app.orchestration.orchestrator.JobRepository", lambda s: mock_job_repo)

    await orchestrator.advance_workflow(wf_id)

    mock_job_repo.create_job.assert_awaited_once()
    kwargs = mock_job_repo.create_job.await_args.kwargs
    assert kwargs["workflow_id"] == wf_id
    assert kwargs["logical_key"] == "script"
    assert kwargs["stage"] == "SCRIPT"
    input_payload = kwargs["input_payload"]
    assert input_payload["research_artifact_id"] == art_id
    assert input_payload["research_artifact_checksum"] == checksum
    assert "research_output" not in input_payload


@pytest.mark.asyncio
async def test_advance_workflow_missing_artifact_metadata_raises_and_halts(monkeypatch):
    """If completed research job lacks _artifact_id or _checksum, raise OrchestrationError and halt."""
    from app.orchestration.errors import OrchestrationError

    wf_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    wf_model = WorkflowModel(
        id=wf_id,
        topic="Quantum Computing",
        current_stage="RESEARCH",
        status="RUNNING",
        created_at=now,
        updated_at=now,
    )
    r_job = JobModel(
        id=str(uuid.uuid4()),
        workflow_id=wf_id,
        logical_key="research",
        job_type="research",
        stage="RESEARCH",
        status=JobStatus.COMPLETED.value,
        output_payload={
            "_artifact_id": "",
            "_checksum": "bad",
        },  # Missing/malformed _artifact_id and _checksum
    )

    mock_wf_repo = AsyncMock()
    mock_wf_repo.get_workflow_for_update.return_value = wf_model
    mock_job_repo = AsyncMock()
    mock_job_repo.list_jobs_for_workflow.return_value = [r_job]

    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = WorkflowOrchestrator(session_factory=session_factory)
    monkeypatch.setattr("app.orchestration.orchestrator.WorkflowRepository", lambda s: mock_wf_repo)
    monkeypatch.setattr("app.orchestration.orchestrator.JobRepository", lambda s: mock_job_repo)

    with pytest.raises(OrchestrationError, match="missing valid '_artifact_id'"):
        await orchestrator.advance_workflow(wf_id)

    # 2. Malformed UUID
    r_job.output_payload = {"_artifact_id": "not-a-valid-uuid", "_checksum": "a" * 64}
    with pytest.raises(OrchestrationError, match="malformed UUID"):
        await orchestrator.advance_workflow(wf_id)

    # 3. Malformed non-hex checksum
    r_job.output_payload = {"_artifact_id": str(uuid.uuid4()), "_checksum": "z" * 64}
    with pytest.raises(OrchestrationError, match="missing valid 64-char lowercase hex SHA-256"):
        await orchestrator.advance_workflow(wf_id)

    # Workflow status must NOT be updated to SCRIPT
    mock_wf_repo.update_status.assert_not_awaited()
    mock_job_repo.create_job.assert_not_awaited()
