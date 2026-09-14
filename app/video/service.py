import asyncio
import time
from pathlib import Path
from uuid import UUID

from app.agents.storyboard.models import StoryboardResult, StoryboardShot
from app.artifacts.models import ArtifactRef
from app.logging import get_logger
from app.models.errors import ModelResponseError, ModelTimeoutError
from app.models.video import (
    ACTIVE_VIDEO_STATUSES,
    VideoGenerationRequest,
    VideoJobRecord,
    VideoModel,
    VideoOperation,
)
from app.storage.base import ArtifactStorage

logger = get_logger(__name__)


class VideoGenerationService:
    """Orchestrates video generation with stable logical-shot idempotency and storage staging."""

    def __init__(
        self,
        video_model: VideoModel,
        *,
        storage: ArtifactStorage | None = None,
        staging_dir: Path | None = None,
        poll_interval_seconds: float = 5.0,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.video_model = video_model
        self.storage = storage
        self.staging_dir = staging_dir or Path("tmp/staging")
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds

        # In-memory registry tracking jobs by stable logical key (workflow_id:shot:shot_number)
        self._jobs: dict[str, VideoJobRecord] = {}
        self._artifact_refs: dict[str, ArtifactRef] = {}

    @staticmethod
    def resolve_provider_duration(estimated_duration_seconds: float) -> int:
        """Resolve a storyboard shot's creative timing target into a discrete provider duration.

        Policy: Veo generation uses provider-supported discrete clip durations (5s or 10s).
        Normalization from storyboard target duration to provider duration is an explicit
        generation-layer decision and must not mutate the storyboard contract.
        """
        if estimated_duration_seconds <= 7.5:
            return 5
        return 10

    def get_job(self, logical_key: str) -> VideoJobRecord | None:
        """Retrieve existing job record for a logical shot if present."""
        return self._jobs.get(logical_key)

    def get_artifact(self, logical_key: str) -> ArtifactRef | None:
        """Retrieve stored artifact reference for a logical shot if present."""
        return self._artifact_refs.get(logical_key)

    async def generate_shot(
        self,
        shot: StoryboardShot,
        workflow_id: UUID,
        *,
        aspect_ratio: str = "9:16",
        attempt: int = 1,
    ) -> VideoJobRecord:
        """Generate a video clip for a storyboard shot with stable logical-shot idempotency."""
        logical_key = f"{workflow_id}:shot:{shot.shot_number}"
        existing_job = self.get_job(logical_key)

        # 1. Idempotency Check: Same logical generation request handling
        if existing_job is not None:
            if existing_job.status == "completed":
                logger.info(
                    "service.video.idempotent_reuse",
                    logical_key=logical_key,
                    job_id=str(existing_job.job_id),
                    artifact_id=str(existing_job.output_artifact_id),
                )
                return existing_job

            if existing_job.status in ACTIVE_VIDEO_STATUSES and existing_job.operation_id:
                logger.info(
                    "service.video.idempotent_resume",
                    logical_key=logical_key,
                    operation_id=existing_job.operation_id,
                )
                record = existing_job
                operation_id = existing_job.operation_id
            else:
                # Previous attempt failed or cancelled -> initiate new attempt
                logger.info(
                    "service.video.retry_attempt",
                    logical_key=logical_key,
                    previous_status=existing_job.status,
                    attempt=attempt,
                )
                record = self._create_new_job_record(
                    shot, workflow_id, logical_key, aspect_ratio, attempt
                )
                operation_id = None
        else:
            record = self._create_new_job_record(
                shot, workflow_id, logical_key, aspect_ratio, attempt
            )
            operation_id = None

        # 2. Submit to provider if no active operation ID exists
        if operation_id is None:
            req = VideoGenerationRequest(
                workflow_id=workflow_id,
                shot_number=shot.shot_number,
                prompt=shot.video_prompt,
                negative_prompt=shot.negative_prompt,
                duration_seconds=record.duration_seconds,
                aspect_ratio=aspect_ratio,
                attempt=attempt,
            )

            try:
                op = await self.video_model.submit_generation(req)
                record.operation_id = op.operation_id
                record.transition_to("submitted")
                operation_id = op.operation_id
            except Exception as e:
                record.transition_to("failed", error_message=str(e))
                raise

        # 3. Poll operation until complete
        op = await self._poll_operation_until_done(record, operation_id)

        if op.status == "failed" or op.error_message:
            error_msg = op.error_message or "Unknown provider failure"
            record.transition_to("failed", error_message=error_msg)
            raise ModelResponseError(
                f"Video generation failed for shot {shot.shot_number}: {error_msg}",
                provider=record.provider,
            )

        # 4. Stage locally and persist to storage
        await self._stage_and_persist_video(record, op, workflow_id, shot.shot_number)

        record.transition_to("completed")
        logger.info(
            "service.video.job_completed",
            logical_key=logical_key,
            job_id=str(record.job_id),
            duration=record.duration_seconds,
            storage_path=record.output_storage_path,
        )
        return record

    async def generate_storyboard(
        self,
        storyboard: StoryboardResult,
        *,
        aspect_ratio: str | None = None,
        max_concurrency: int = 2,
    ) -> list[VideoJobRecord]:
        """Generate video clips for all shots in a storyboard with concurrency throttling."""
        target_aspect_ratio = aspect_ratio or storyboard.aspect_ratio
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _gen_with_limit(shot: StoryboardShot) -> VideoJobRecord:
            async with semaphore:
                return await self.generate_shot(
                    shot,
                    storyboard.workflow_id,
                    aspect_ratio=target_aspect_ratio,
                )

        tasks = [_gen_with_limit(shot) for shot in storyboard.shots]
        return await asyncio.gather(*tasks)

    def _create_new_job_record(
        self,
        shot: StoryboardShot,
        workflow_id: UUID,
        logical_key: str,
        aspect_ratio: str,
        attempt: int,
    ) -> VideoJobRecord:
        duration_seconds = self.resolve_provider_duration(shot.estimated_duration_seconds)
        provider = getattr(self.video_model, "provider", "veo")
        model_name = getattr(self.video_model, "model_name", "veo-2.0-generate-001")

        record = VideoJobRecord(
            workflow_id=workflow_id,
            shot_number=shot.shot_number,
            attempt=attempt,
            logical_key=logical_key,
            provider=provider,
            model_name=model_name,
            prompt=shot.video_prompt,
            negative_prompt=shot.negative_prompt,
            duration_seconds=duration_seconds,
            aspect_ratio=aspect_ratio,
        )
        self._jobs[logical_key] = record
        return record

    async def _poll_operation_until_done(
        self,
        record: VideoJobRecord,
        operation_id: str,
    ) -> VideoOperation:
        start_time = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout_seconds:
                timeout_msg = f"Polling video operation {operation_id} timed out after {elapsed:.1f}s (timeout: {self.timeout_seconds}s)."
                record.transition_to("failed", error_message=timeout_msg)
                raise ModelTimeoutError(timeout_msg, provider=record.provider)

            op = await self.video_model.get_operation_status(operation_id)

            if op.done:
                return op

            if record.status == "submitted" and op.status == "processing":
                record.transition_to("processing")

            await asyncio.sleep(self.poll_interval_seconds)

    async def _stage_and_persist_video(
        self,
        record: VideoJobRecord,
        op: VideoOperation,
        workflow_id: UUID,
        shot_number: int,
    ) -> None:
        """Write raw video bytes to local staging and upload to cloud storage if configured."""
        shot_filename = f"shot_{shot_number}.mp4"
        local_dir = self.staging_dir / str(workflow_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / shot_filename

        video_bytes = op.video_bytes or b""
        local_file.write_bytes(video_bytes)

        if self.storage is not None:
            artifact_ref = await self.storage.upload(
                local_file,
                destination=f"shots/{shot_filename}",
                workflow_id=workflow_id,
                artifact_type="video_shot",
                mime_type="video/mp4",
                overwrite=True,
            )
            record.output_artifact_id = artifact_ref.id
            record.output_storage_path = artifact_ref.drive_file_id
            self._artifact_refs[record.logical_key] = artifact_ref
        else:
            record.output_storage_path = str(local_file)
