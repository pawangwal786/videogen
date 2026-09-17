"""Workflow orchestrator driving state transitions and pipeline stages."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.session import get_session_factory
from app.logging import get_logger
from app.orchestration.errors import WorkflowNotFoundError
from app.orchestration.models import Artifact, Workflow
from app.orchestration.state_machine import JobStage, JobStatus, WorkflowStatus
from app.repositories.artifact import ArtifactRepository
from app.repositories.job import JobRepository
from app.repositories.workflow import WorkflowRepository

logger = get_logger(__name__)


class WorkflowOrchestrator:
    """Orchestrates end-to-end video generation pipeline stages through PostgreSQL persistence."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._session_factory = session_factory or get_session_factory()

    async def create_artifact(
        self,
        workflow_id: str,
        artifact_type: str,
        storage_path: str,
        job_id: str | None = None,
        gdrive_file_id: str | None = None,
        file_size_bytes: int | None = None,
        checksum: str | None = None,
        mime_type: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        """Persist a new artifact and link to workflow and optional producing job."""
        async with self._session_factory() as session:
            repo = ArtifactRepository(session)
            model = await repo.create_artifact(
                workflow_id=workflow_id,
                artifact_type=artifact_type,
                storage_path=storage_path,
                job_id=job_id,
                gdrive_file_id=gdrive_file_id,
                file_size_bytes=file_size_bytes,
                checksum=checksum,
                mime_type=mime_type,
                artifact_metadata=artifact_metadata,
            )
            await session.commit()
            return Artifact.model_validate(model)

    async def list_artifacts_for_workflow(
        self,
        workflow_id: str,
        artifact_type: str | None = None,
    ) -> list[Artifact]:
        """List all artifacts for a given workflow."""
        async with self._session_factory() as session:
            repo = ArtifactRepository(session)
            models = await repo.list_artifacts_for_workflow(
                workflow_id, artifact_type=artifact_type
            )
            return [Artifact.model_validate(m) for m in models]

    async def create_workflow(
        self,
        topic: str,
        idempotency_key: str | None = None,
    ) -> Workflow:
        """Create a workflow and initialize its first stage job (RESEARCH)."""
        async with self._session_factory() as session:
            wf_repo = WorkflowRepository(session)
            job_repo = JobRepository(session)

            wf_model = await wf_repo.create_workflow(topic=topic, idempotency_key=idempotency_key)

            # Ensure initial RESEARCH job exists
            await job_repo.create_job(
                workflow_id=wf_model.id,
                logical_key="research",
                job_type="research",
                stage=JobStage.RESEARCH.value,
                input_payload={"topic": topic},
            )
            await session.commit()
            return Workflow.model_validate(wf_model)

    async def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Retrieve workflow by ID."""
        async with self._session_factory() as session:
            wf_repo = WorkflowRepository(session)
            wf_model = await wf_repo.get_workflow(workflow_id)
            if wf_model is None:
                return None
            return Workflow.model_validate(wf_model)

    async def cancel_workflow(self, workflow_id: str) -> Workflow:
        """Cancel an in-progress workflow and all its active jobs."""
        async with self._session_factory() as session:
            wf_repo = WorkflowRepository(session)
            job_repo = JobRepository(session)

            wf_model = await wf_repo.get_workflow_for_update(workflow_id)
            if wf_model is None:
                raise WorkflowNotFoundError(workflow_id)

            if wf_model.status in {
                WorkflowStatus.COMPLETED.value,
                WorkflowStatus.FAILED.value,
                WorkflowStatus.CANCELLED.value,
            }:
                return Workflow.model_validate(wf_model)

            await wf_repo.update_status(workflow_id, status=WorkflowStatus.CANCELLED.value)

            # Cancel all non-completed jobs
            jobs = await job_repo.list_jobs_for_workflow(workflow_id)
            for j in jobs:
                if j.status not in {
                    JobStatus.COMPLETED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                }:
                    j.status = JobStatus.CANCELLED.value

            await session.commit()
            return Workflow.model_validate(wf_model)

    async def advance_workflow(self, workflow_id: str) -> Workflow:
        """Evaluate workflow jobs and transition to subsequent stages or mark completed/failed."""
        async with self._session_factory() as session:
            wf_repo = WorkflowRepository(session)
            job_repo = JobRepository(session)

            wf = await wf_repo.get_workflow_for_update(workflow_id)
            if wf is None:
                raise WorkflowNotFoundError(workflow_id)

            if wf.status in {
                WorkflowStatus.COMPLETED.value,
                WorkflowStatus.FAILED.value,
                WorkflowStatus.CANCELLED.value,
            }:
                return Workflow.model_validate(wf)

            jobs = await job_repo.list_jobs_for_workflow(workflow_id)
            jobs_by_key = {j.logical_key: j for j in jobs}

            failed_job = next((j for j in jobs if j.status == JobStatus.FAILED.value), None)
            if failed_job is not None:
                await wf_repo.update_status(
                    workflow_id,
                    status=WorkflowStatus.FAILED.value,
                    error_code=failed_job.error_code,
                    error_message=failed_job.error_message,
                )
                await session.commit()
                updated_wf = await wf_repo.get_workflow(workflow_id)
                return Workflow.model_validate(updated_wf)

            current_stage = JobStage(wf.current_stage)

            if current_stage == JobStage.RESEARCH:
                r_job = jobs_by_key.get("research")
                if r_job and r_job.status == JobStatus.FAILED.value:
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.FAILED.value,
                        error_code=r_job.error_code,
                        error_message=r_job.error_message,
                    )
                elif r_job and r_job.status == JobStatus.COMPLETED.value:
                    if "script" not in jobs_by_key:
                        await job_repo.create_job(
                            workflow_id=workflow_id,
                            logical_key="script",
                            job_type="script",
                            stage=JobStage.SCRIPT.value,
                            input_payload={
                                "topic": wf.topic,
                                "research_output": r_job.output_payload,
                            },
                        )
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.RUNNING.value,
                        current_stage=JobStage.SCRIPT.value,
                    )

            elif current_stage == JobStage.SCRIPT:
                s_job = jobs_by_key.get("script")
                if s_job and s_job.status == JobStatus.FAILED.value:
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.FAILED.value,
                        error_code=s_job.error_code,
                        error_message=s_job.error_message,
                    )
                elif s_job and s_job.status == JobStatus.COMPLETED.value:
                    if "storyboard" not in jobs_by_key:
                        await job_repo.create_job(
                            workflow_id=workflow_id,
                            logical_key="storyboard",
                            job_type="storyboard",
                            stage=JobStage.STORYBOARD.value,
                            input_payload={
                                "script_output": s_job.output_payload,
                            },
                        )
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.RUNNING.value,
                        current_stage=JobStage.STORYBOARD.value,
                    )

            elif current_stage == JobStage.STORYBOARD:
                sb_job = jobs_by_key.get("storyboard")
                if sb_job and sb_job.status == JobStatus.FAILED.value:
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.FAILED.value,
                        error_code=sb_job.error_code,
                        error_message=sb_job.error_message,
                    )
                elif sb_job and sb_job.status == JobStatus.COMPLETED.value:
                    shots = sb_job.output_payload.get("shots", []) if sb_job.output_payload else []
                    aspect_ratio = (
                        sb_job.output_payload.get("aspect_ratio", "9:16")
                        if sb_job.output_payload
                        else "9:16"
                    )

                    for shot in shots:
                        shot_num = shot.get("shot_number")
                        key = f"video:shot:{shot_num}"
                        if key not in jobs_by_key:
                            await job_repo.create_job(
                                workflow_id=workflow_id,
                                logical_key=key,
                                job_type="video_generation",
                                stage=JobStage.VIDEO_GENERATION.value,
                                input_payload={
                                    "workflow_id": workflow_id,
                                    "shot": shot,
                                    "aspect_ratio": aspect_ratio,
                                },
                            )

                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.RUNNING.value,
                        current_stage=JobStage.VIDEO_GENERATION.value,
                    )

            elif current_stage == JobStage.VIDEO_GENERATION:
                video_jobs = [j for k, j in jobs_by_key.items() if k.startswith("video:shot:")]
                any_failed = any(j.status == JobStatus.FAILED.value for j in video_jobs)
                all_completed = len(video_jobs) > 0 and all(
                    j.status == JobStatus.COMPLETED.value for j in video_jobs
                )

                if any_failed:
                    first_fail = next(j for j in video_jobs if j.status == JobStatus.FAILED.value)
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.FAILED.value,
                        error_code=first_fail.error_code,
                        error_message=first_fail.error_message,
                    )
                elif all_completed:
                    if "media_assembly" not in jobs_by_key:
                        shot_outputs = [
                            j.output_payload
                            for j in sorted(video_jobs, key=lambda x: x.logical_key)
                        ]
                        await job_repo.create_job(
                            workflow_id=workflow_id,
                            logical_key="media_assembly",
                            job_type="media_assembly",
                            stage=JobStage.MEDIA_ASSEMBLY.value,
                            input_payload={
                                "workflow_id": workflow_id,
                                "shots": shot_outputs,
                            },
                        )
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.RUNNING.value,
                        current_stage=JobStage.MEDIA_ASSEMBLY.value,
                    )

            elif current_stage == JobStage.MEDIA_ASSEMBLY:
                assembly_job = jobs_by_key.get("media_assembly")
                if assembly_job and assembly_job.status == JobStatus.FAILED.value:
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.FAILED.value,
                        error_code=assembly_job.error_code,
                        error_message=assembly_job.error_message,
                    )
                elif assembly_job and assembly_job.status == JobStatus.COMPLETED.value:
                    await wf_repo.update_status(
                        workflow_id,
                        status=WorkflowStatus.COMPLETED.value,
                        current_stage=JobStage.COMPLETED.value,
                    )

            await session.commit()
            updated_wf = await wf_repo.get_workflow(workflow_id)
            return Workflow.model_validate(updated_wf)
