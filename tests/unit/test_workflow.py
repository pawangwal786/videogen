"""Unit tests for orchestration domain models."""

from datetime import datetime, timezone
import uuid

from app.orchestration.models import (
    Artifact,
    ClaimedJob,
    Job,
    JobAttempt,
    ReconciliationOutcome,
    ReconciliationStatus,
    Workflow,
)
from app.orchestration.state_machine import (
    ArtifactLifecycleStatus,
    AttemptStatus,
    JobStage,
    JobStatus,
    WorkflowStatus,
)


def test_workflow_domain_model_defaults():
    now = datetime.now(timezone.utc)
    wf = Workflow(
        id=str(uuid.uuid4()),
        topic="AI infrastructure",
        created_at=now,
        updated_at=now,
    )
    assert wf.topic == "AI infrastructure"
    assert wf.status == WorkflowStatus.PENDING
    assert wf.current_stage == JobStage.RESEARCH
    assert wf.idempotency_key is None


def test_job_domain_model_defaults():
    now = datetime.now(timezone.utc)
    job = Job(
        id=str(uuid.uuid4()),
        workflow_id=str(uuid.uuid4()),
        logical_key="research",
        job_type="research",
        stage=JobStage.RESEARCH,
        available_at=now,
        created_at=now,
    )
    assert job.status == JobStatus.PENDING
    assert job.max_attempts == 3
    assert job.version == 1
    assert job.input_payload == {}


def test_job_attempt_domain_model():
    now = datetime.now(timezone.utc)
    attempt = JobAttempt(
        id=str(uuid.uuid4()),
        job_id=str(uuid.uuid4()),
        attempt_number=1,
        started_at=now,
        heartbeat_at=now,
    )
    assert attempt.status == AttemptStatus.CLAIMED
    assert attempt.worker_id is None
    assert attempt.provider_operation_id is None


def test_reconciliation_outcome_model():
    outcome = ReconciliationOutcome(
        status=ReconciliationStatus.RESOLVED,
        provider_operation_id="operations/test-123",
        metadata={"status": "done"},
    )
    assert outcome.status == ReconciliationStatus.RESOLVED
    assert outcome.provider_operation_id == "operations/test-123"
