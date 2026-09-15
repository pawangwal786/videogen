from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from uuid import UUID

from app.logging import get_logger
from app.mpt.adapter import MediaProcessor
from app.mpt.errors import MediaConfigurationError, MediaValidationError
from app.mpt.models import (
    MediaAssemblyRequest,
    MediaAssemblyResult,
    MediaClip,
)
from app.storage.base import ArtifactStorage

logger = get_logger(__name__)


class MediaAssemblyService:
    """Orchestrates clip resolution, normalization, concatenation, and audio muxing into a final video artifact."""

    def __init__(
        self,
        media_processor: MediaProcessor,
        *,
        storage: ArtifactStorage | None = None,
        staging_dir: Path | None = None,
        max_concurrency: int = 2,
        duration_tolerance_seconds: float = 0.5,
    ) -> None:
        if max_concurrency < 1:
            raise MediaConfigurationError("max_concurrency must be >= 1", operation="init")

        self.media_processor = media_processor
        self.storage = storage
        self.staging_dir = staging_dir or Path("tmp/staging")
        self.max_concurrency = max_concurrency
        self.duration_tolerance_seconds = duration_tolerance_seconds

    async def assemble(
        self,
        request: MediaAssemblyRequest,
    ) -> MediaAssemblyResult:
        """Assemble video clips and optional audio into an authoritative final video artifact."""
        workflow_id = request.workflow_id
        logger.info(
            "mpt.assembly.started",
            workflow_id=str(workflow_id),
            num_clips=len(request.clips),
            has_audio=request.audio_track is not None,
        )

        work_dir = self.staging_dir / str(workflow_id) / "assembly"
        input_dir = work_dir / "input"
        normalized_dir = work_dir / "normalized"
        concat_dir = work_dir / "concat"
        final_dir = work_dir / "final"

        work_dir.mkdir(parents=True, exist_ok=True)
        normalized_dir.mkdir(parents=True, exist_ok=True)
        concat_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        downloaded_inputs: list[Path] = []
        normalized_clips: list[Path] = []

        try:
            # 1. Resolve every clip to a verified, local readable path
            resolved_paths: dict[int, Path] = {}
            for clip in request.clips:
                resolved_path = await self._resolve_clip_source(
                    clip, workflow_id, input_dir, downloaded_inputs
                )
                resolved_paths[clip.shot_number] = resolved_path

            # 2. Normalize all clips in parallel bounded by semaphore
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _normalize_single(clip: MediaClip) -> Path:
                async with semaphore:
                    src = resolved_paths[clip.shot_number]
                    dest = normalized_dir / f"clip_{clip.shot_number}.mp4"
                    logger.debug(
                        "mpt.assembly.normalize_clip",
                        workflow_id=str(workflow_id),
                        shot_number=clip.shot_number,
                    )
                    await self.media_processor.normalize_clip(
                        source=src,
                        output=dest,
                        profile=request.profile,
                        target_duration=clip.target_duration_seconds,
                        tolerance_seconds=request.duration_tolerance_seconds,
                    )
                    return dest

            tasks = [_normalize_single(clip) for clip in request.clips]
            normalized_clips = await asyncio.gather(*tasks)

            # 3. Concatenate normalized clips
            concat_output = concat_dir / "concat.mp4"
            logger.info("mpt.assembly.concat_started", workflow_id=str(workflow_id))
            concat_info = await self.media_processor.concatenate(normalized_clips, concat_output)

            # 4. Integrate optional audio
            final_output = final_dir / "final_video.mp4"
            if request.audio_track is not None:
                logger.info("mpt.assembly.audio_mux_started", workflow_id=str(workflow_id))
                final_info = await self.media_processor.add_audio(
                    video=concat_output,
                    audio=request.audio_track,
                    output=final_output,
                    video_duration=concat_info.duration_seconds,
                )
            else:
                shutil.copy2(concat_output, final_output)
                final_info = await self.media_processor.probe(final_output)

            # 5. Persist to ArtifactStorage if configured
            artifact_ref = None
            if self.storage is not None:
                logger.info("mpt.assembly.upload_started", workflow_id=str(workflow_id))
                artifact_ref = await self.storage.upload(
                    final_output,
                    destination=f"final/{workflow_id}/final_video.mp4",
                    workflow_id=workflow_id,
                    artifact_type="final_video",
                    mime_type="video/mp4",
                    overwrite=True,
                )

            logger.info(
                "mpt.assembly.completed",
                workflow_id=str(workflow_id),
                final_duration=final_info.duration_seconds,
                output_path=str(final_output),
                artifact_id=str(artifact_ref.id) if artifact_ref else None,
            )

            return MediaAssemblyResult(
                workflow_id=workflow_id,
                output_path=final_output,
                duration_seconds=final_info.duration_seconds,
                media_info=final_info,
                artifact_ref=artifact_ref,
            )

        finally:
            # Cleanup intermediate normalized clips, concat output, and downloaded inputs
            self._cleanup_intermediates(normalized_dir, concat_dir, downloaded_inputs)

    async def _resolve_clip_source(
        self,
        clip: MediaClip,
        workflow_id: UUID,
        input_dir: Path,
        downloaded_inputs: list[Path],
    ) -> Path:
        """Ensure the source clip exists locally; download via storage if needed."""
        local_path: Path
        if clip.source_path.exists() and clip.source_path.is_file():
            local_path = clip.source_path
        elif clip.artifact_ref is not None and self.storage is not None:
            input_dir.mkdir(parents=True, exist_ok=True)
            local_path = input_dir / f"shot_{clip.shot_number}.mp4"
            logger.info(
                "mpt.assembly.download_artifact",
                workflow_id=str(workflow_id),
                shot_number=clip.shot_number,
                artifact_id=str(clip.artifact_ref.id),
            )
            await self.storage.download(clip.artifact_ref, local_path)
            downloaded_inputs.append(local_path)
        else:
            raise MediaValidationError(
                f"Clip file does not exist locally and cannot be resolved: {clip.source_path}",
                workflow_id=workflow_id,
                operation="resolve_source",
            )

        # Enforce universal local media validation invariant
        await self._validate_local_video(local_path, clip.shot_number, workflow_id)
        return local_path

    async def _validate_local_video(
        self,
        path: Path,
        shot_number: int,
        workflow_id: UUID,
    ) -> None:
        """Enforce strict invariant that a local video file exists, is non-empty, and has a probeable video stream."""
        if not path.exists() or not path.is_file():
            raise MediaValidationError(
                f"Source media file for shot {shot_number} does not exist: {path}",
                workflow_id=workflow_id,
                operation="resolve_source",
            )
        try:
            if path.stat().st_size == 0:
                raise MediaValidationError(
                    f"Source media file for shot {shot_number} is empty (0 bytes): {path}",
                    workflow_id=workflow_id,
                    operation="resolve_source",
                )
        except OSError as e:
            raise MediaValidationError(
                f"Cannot inspect media file for shot {shot_number}: {e}",
                workflow_id=workflow_id,
                operation="resolve_source",
            ) from e

        try:
            probe_info = await self.media_processor.probe(path)
        except Exception as e:
            raise MediaValidationError(
                f"Source media file for shot {shot_number} failed media probe validation: {e}",
                workflow_id=workflow_id,
                operation="resolve_source",
            ) from e

        if not probe_info.has_video:
            raise MediaValidationError(
                f"Source media file for shot {shot_number} contains no video stream: {path}",
                workflow_id=workflow_id,
                operation="resolve_source",
            )

    @staticmethod
    def _cleanup_intermediates(
        normalized_dir: Path,
        concat_dir: Path,
        downloaded_inputs: list[Path],
    ) -> None:
        """Best-effort cleanup of temporary media files."""
        try:
            if normalized_dir.exists():
                shutil.rmtree(normalized_dir, ignore_errors=True)
            if concat_dir.exists():
                shutil.rmtree(concat_dir, ignore_errors=True)
            for path in downloaded_inputs:
                if path.exists():
                    path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("mpt.assembly.cleanup_error", error=str(e))
