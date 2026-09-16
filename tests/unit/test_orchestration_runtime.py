"""Unit tests for WorkflowOrchestrator and JobWorker edge cases and error paths."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.base import utc_now
from app.db.models.job import JobAttemptModel, JobModel
from app.db.models.workflow import WorkflowModel
from app.orchestration.errors import LeaseConflictError, WorkflowNotFoundError
from app.orchestration.orchestrator import WorkflowOrchestrator
from app.orchestration.state_machine import AttemptStatus, JobStage, JobStatus, WorkflowStatus
from app.orchestration.worker import JobWorker


class FakeHandler:
    def __init__(self, output: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self.output = output or {}
        self.exc = exc
        self.called = False

    async def execute(
        self, job: JobModel, attempt: JobAttemptModel, worker: JobWorker
    ) -> dict[str, Any]:
        self.called = True
        if self.exc:
            raise self.exc
        return self.output


@pytest.mark.asyncio
async def test_orchestrator_not_found_errors():
    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = WorkflowOrchestrator(session_factory)

    with patch("app.orchestration.orchestrator.WorkflowRepository") as mock_wf_repo_cls:
        wf_repo = AsyncMock()
        mock_wf_repo_cls.return_value = wf_repo
        wf_repo.get_workflow.return_value = None
        wf_repo.get_workflow_for_update.return_value = None

        assert await orchestrator.get_workflow("missing-id") is None

        with pytest.raises(WorkflowNotFoundError):
            await orchestrator.cancel_workflow("missing-id")

        with pytest.raises(WorkflowNotFoundError):
            await orchestrator.advance_workflow("missing-id")


@pytest.mark.asyncio
async def test_orchestrator_advance_terminal_status_early_return():
    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = WorkflowOrchestrator(session_factory)

    now = utc_now()
    wf_model = WorkflowModel(
        id="wf-1",
        topic="Test",
        status=WorkflowStatus.COMPLETED.value,
        current_stage=JobStage.COMPLETED.value,
        created_at=now,
        updated_at=now,
    )

    with patch("app.orchestration.orchestrator.WorkflowRepository") as mock_wf_repo_cls:
        wf_repo = AsyncMock()
        mock_wf_repo_cls.return_value = wf_repo
        wf_repo.get_workflow.return_value = wf_model
        wf_repo.get_workflow_for_update.return_value = wf_model

        result = await orchestrator.advance_workflow("wf-1")
        assert result.status == WorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_orchestrator_advance_stage_failures():
    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = WorkflowOrchestrator(session_factory)
    now = utc_now()

    for stage, job_key in [
        (JobStage.RESEARCH, "research"),
        (JobStage.SCRIPT, "script"),
        (JobStage.STORYBOARD, "storyboard"),
        (JobStage.MEDIA_ASSEMBLY, "media_assembly"),
    ]:
        wf_model = WorkflowModel(
            id=f"wf-{stage.value}",
            topic="Test",
            status=WorkflowStatus.RUNNING.value,
            current_stage=stage.value,
            created_at=now,
            updated_at=now,
        )
        failed_job = JobModel(
            id=f"job-{stage.value}",
            workflow_id=wf_model.id,
            logical_key=job_key,
            job_type=job_key,
            stage=stage.value,
            status=JobStatus.FAILED.value,
            error_code="TEST_FAILURE",
            error_message="Stage failed",
            created_at=now,
        )

        with (
            patch("app.orchestration.orchestrator.WorkflowRepository") as mock_wf_repo_cls,
            patch("app.orchestration.orchestrator.JobRepository") as mock_job_repo_cls,
        ):
            wf_repo = AsyncMock()
            job_repo = AsyncMock()
            mock_wf_repo_cls.return_value = wf_repo
            mock_job_repo_cls.return_value = job_repo

            wf_repo.get_workflow.return_value = wf_model
            wf_repo.get_workflow_for_update.return_value = wf_model
            job_repo.list_jobs_for_workflow.return_value = [failed_job]

            await orchestrator.advance_workflow(wf_model.id)
            wf_repo.update_status.assert_called_with(
                wf_model.id,
                status=WorkflowStatus.FAILED.value,
                error_code="TEST_FAILURE",
                error_message="Stage failed",
            )


@pytest.mark.asyncio
async def test_orchestrator_advance_video_generation_failure():
    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = WorkflowOrchestrator(session_factory)
    now = utc_now()

    wf_model = WorkflowModel(
        id="wf-video-fail",
        topic="Test",
        status=WorkflowStatus.RUNNING.value,
        current_stage=JobStage.VIDEO_GENERATION.value,
        created_at=now,
        updated_at=now,
    )
    failed_video_job = JobModel(
        id="job-video-1",
        workflow_id=wf_model.id,
        logical_key="video:shot:1",
        job_type="video_generation",
        stage=JobStage.VIDEO_GENERATION.value,
        status=JobStatus.FAILED.value,
        error_code="VEO_FAILED",
        error_message="Provider failed",
        created_at=now,
    )

    with (
        patch("app.orchestration.orchestrator.WorkflowRepository") as mock_wf_repo_cls,
        patch("app.orchestration.orchestrator.JobRepository") as mock_job_repo_cls,
    ):
        wf_repo = AsyncMock()
        job_repo = AsyncMock()
        mock_wf_repo_cls.return_value = wf_repo
        mock_job_repo_cls.return_value = job_repo

        wf_repo.get_workflow.return_value = wf_model
        wf_repo.get_workflow_for_update.return_value = wf_model
        job_repo.list_jobs_for_workflow.return_value = [failed_video_job]

        await orchestrator.advance_workflow(wf_model.id)
        wf_repo.update_status.assert_called_with(
            wf_model.id,
            status=WorkflowStatus.FAILED.value,
            error_code="VEO_FAILED",
            error_message="Provider failed",
        )


@pytest.mark.asyncio
async def test_worker_submission_no_active_job_is_noop():
    orchestrator = AsyncMock()
    worker = JobWorker(session_factory=MagicMock(), orchestrator=orchestrator)
    # Neither active job id nor lease token set
    await worker.record_submission_pending(provider="veo")
    await worker.record_submitted("op-123")
    assert worker._active_job_id is None


@pytest.mark.asyncio
async def test_worker_heartbeat_loop_error_handling():
    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = AsyncMock()
    worker = JobWorker(
        session_factory=session_factory, orchestrator=orchestrator, heartbeat_interval_seconds=0.01
    )
    worker._active_job_id = "job-1"

    # Case 1: LeaseConflictError terminates loop
    with patch("app.orchestration.worker.JobRepository") as mock_job_repo_cls:
        repo = AsyncMock()
        mock_job_repo_cls.return_value = repo
        repo.heartbeat_attempt.side_effect = LeaseConflictError("job-1", "worker-1", "token-1")

        await worker._heartbeat_loop("job-1", "token-1")
        repo.heartbeat_attempt.assert_called_once()

    # Case 2: Generic Exception logs and continues until active job changes
    with patch("app.orchestration.worker.JobRepository") as mock_job_repo_cls:
        repo = AsyncMock()
        mock_job_repo_cls.return_value = repo
        call_count = 0

        async def raise_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            worker._active_job_id = None  # Exit loop on next iteration
            raise RuntimeError("Temporary DB disconnect")

        repo.heartbeat_attempt.side_effect = raise_once
        worker._active_job_id = "job-2"
        await worker._heartbeat_loop("job-2", "token-2")
        assert call_count == 1


@pytest.mark.asyncio
async def test_worker_execution_failure_handling():
    session = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    session_factory.return_value.__aexit__.return_value = None

    orchestrator = AsyncMock()
    worker = JobWorker(
        session_factory=session_factory,
        orchestrator=orchestrator,
        handlers={"failing_type": FakeHandler(exc=ValueError("Computation error"))},
    )
    now = utc_now()

    job = JobModel(
        id="job-fail-1",
        workflow_id="wf-1",
        logical_key="failing_key",
        job_type="failing_type",
        stage=JobStage.RESEARCH.value,
        status=JobStatus.CLAIMED.value,
        max_attempts=3,
        created_at=now,
    )
    attempt = JobAttemptModel(
        id="att-1",
        job_id=job.id,
        attempt_number=1,
        lease_token="lease-1",
        status=AttemptStatus.CLAIMED.value,
        started_at=now,
        heartbeat_at=now,
    )

    with patch("app.orchestration.worker.JobRepository") as mock_job_repo_cls:
        repo = AsyncMock()
        mock_job_repo_cls.return_value = repo
        repo.claim_next_job.return_value = (job, attempt)

        worked = await worker.run_once()
        assert worked is True
        repo.fail_job.assert_called_once()
        orchestrator.advance_workflow.assert_called_once_with("wf-1")


@pytest.mark.asyncio
async def test_worker_run_once_when_shutdown_requested():
    orchestrator = AsyncMock()
    worker = JobWorker(session_factory=MagicMock(), orchestrator=orchestrator)
    await worker.stop()
    assert await worker.run_once() is False
