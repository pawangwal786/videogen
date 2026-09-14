import asyncio
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.logging import get_logger
from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)
from app.models.video import (
    VideoGenerationRequest,
    VideoModel,
    VideoOperation,
)

logger = get_logger(__name__)

SUPPORTED_VEO_DURATIONS = {5, 10}


class VeoVideoModel(VideoModel):
    """Google Veo implementation of the VideoModel protocol using the official google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "veo-2.0-generate-001",
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 5.0,
        client: genai.Client | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ModelConfigurationError(
                "Veo/Gemini API key must not be empty.",
                provider="veo",
            )
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._client = client or genai.Client(api_key=api_key.strip())

    def _normalize_error(self, error: Exception, operation_name: str) -> ModelError:
        """Map provider exceptions to normalized ModelError types."""
        if isinstance(error, genai_errors.APIError):
            code = getattr(error, "code", None)
            message = str(error)
            if code in {401, 403} or "API_KEY_INVALID" in message or "PERMISSION_DENIED" in message:
                return ModelAuthenticationError(
                    f"Veo authentication failed ({operation_name}): {message}",
                    provider="veo",
                    cause=error,
                )
            if code == 429 or "RESOURCE_EXHAUSTED" in message or "Quota exceeded" in message:
                return ModelRateLimitError(
                    f"Veo rate limit exceeded ({operation_name}): {message}",
                    provider="veo",
                    cause=error,
                )
            if code in {408, 504} or "DEADLINE_EXCEEDED" in message:
                return ModelTimeoutError(
                    f"Veo request timed out ({operation_name}): {message}",
                    provider="veo",
                    cause=error,
                )
            return ModelResponseError(
                f"Veo API error ({operation_name}): {message}",
                provider="veo",
                cause=error,
            )
        if isinstance(error, TimeoutError | asyncio.TimeoutError):
            return ModelTimeoutError(
                f"Veo operation timed out ({operation_name}): {error}",
                provider="veo",
                cause=error,
            )
        return ModelResponseError(
            f"Veo unexpected error during {operation_name}: {error}",
            provider="veo",
            cause=error,
        )

    async def submit_generation(
        self,
        request: VideoGenerationRequest,
    ) -> VideoOperation:
        """Submit a video generation job to Google Veo."""
        if request.duration_seconds not in SUPPORTED_VEO_DURATIONS:
            raise ModelConfigurationError(
                f"Veo supports clip durations of {sorted(SUPPORTED_VEO_DURATIONS)} seconds; got {request.duration_seconds}s.",
                provider="veo",
            )

        logger.info(
            "model.veo.submit_start",
            model=self.model_name,
            workflow_id=str(request.workflow_id),
            shot_number=request.shot_number,
            duration=request.duration_seconds,
            aspect_ratio=request.aspect_ratio,
        )

        source = types.GenerateVideosSource(prompt=request.prompt)
        config = types.GenerateVideosConfig(
            aspect_ratio=request.aspect_ratio,
            duration_seconds=request.duration_seconds,
            negative_prompt=request.negative_prompt,
        )

        try:
            op = await self._client.aio.models.generate_videos(
                model=self.model_name,
                source=source,
                config=config,
            )
        except Exception as e:
            normalized = self._normalize_error(e, "generate_videos")
            logger.error(
                "model.veo.submit_failed",
                model=self.model_name,
                workflow_id=str(request.workflow_id),
                shot_number=request.shot_number,
                error=str(normalized),
            )
            raise normalized from e

        operation_id = op.name or f"veo-op-{request.logical_key}"
        status = "completed" if op.done else "submitted"

        logger.info(
            "model.veo.submitted",
            model=self.model_name,
            operation_id=operation_id,
            done=op.done,
        )

        return VideoOperation(
            operation_id=operation_id,
            provider="veo",
            status=status,
            done=bool(op.done),
            metadata={"model": self.model_name},
        )

    async def get_operation_status(
        self,
        operation_id: str,
    ) -> VideoOperation:
        """Poll the status of an ongoing video generation operation."""
        try:
            raw_op = types.GenerateVideosOperation(name=operation_id)
            op = await self._client.aio.operations.get(raw_op)
        except Exception as e:
            normalized = self._normalize_error(e, f"get_operation({operation_id})")
            logger.error(
                "model.veo.poll_failed",
                operation_id=operation_id,
                error=str(normalized),
            )
            raise normalized from e

        # Check for provider error
        if getattr(op, "error", None):
            error_msg = str(op.error)
            logger.error("model.veo.operation_failed", operation_id=operation_id, error=error_msg)
            return VideoOperation(
                operation_id=operation_id,
                provider="veo",
                status="failed",
                done=True,
                error_message=error_msg,
            )

        if not op.done:
            return VideoOperation(
                operation_id=operation_id,
                provider="veo",
                status="processing",
                done=False,
            )

        # Operation is completed: extract generated video
        result = getattr(op, "response", None) or getattr(op, "result", None)
        if not result:
            return VideoOperation(
                operation_id=operation_id,
                provider="veo",
                status="failed",
                done=True,
                error_message="Operation marked done but returned no response or result payload.",
            )

        # Check safety / content filtering
        filtered_reasons = getattr(result, "rai_media_filtered_reasons", None)
        if filtered_reasons:
            reasons_str = ", ".join(filtered_reasons)
            logger.warning(
                "model.veo.content_filtered",
                operation_id=operation_id,
                reasons=reasons_str,
            )
            return VideoOperation(
                operation_id=operation_id,
                provider="veo",
                status="failed",
                done=True,
                error_message=f"Video content was filtered by safety guidelines: {reasons_str}",
            )

        generated_videos = getattr(result, "generated_videos", None)
        if not generated_videos or not generated_videos[0].video:
            return VideoOperation(
                operation_id=operation_id,
                provider="veo",
                status="failed",
                done=True,
                error_message="Operation finished without generated video content.",
            )

        video_obj = generated_videos[0].video
        video_bytes = getattr(video_obj, "video_bytes", None)
        video_uri = getattr(video_obj, "uri", None)

        logger.info(
            "model.veo.completed",
            operation_id=operation_id,
            has_bytes=bool(video_bytes),
            uri=video_uri,
        )

        return VideoOperation(
            operation_id=operation_id,
            provider="veo",
            status="completed",
            done=True,
            video_bytes=video_bytes,
            video_uri=video_uri,
        )

    async def cancel_generation(
        self,
        operation_id: str,
    ) -> None:
        """Attempt to cancel an active video generation operation."""
        logger.info("model.veo.cancel_requested", operation_id=operation_id)
        # google-genai does not currently expose an explicit cancel endpoint on video operations;
        # log notice for operational visibility.

    async def poll_until_complete(
        self,
        operation_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> VideoOperation:
        """Poll the operation until it reaches a terminal state (completed or failed), or times out."""
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        interval = (
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self.poll_interval_seconds
        )
        start_time = time.perf_counter()

        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed > timeout:
                raise ModelTimeoutError(
                    f"Polling Veo operation {operation_id} timed out after {elapsed:.1f}s (timeout: {timeout}s).",
                    provider="veo",
                )

            op = await self.get_operation_status(operation_id)
            if op.done:
                return op

            await asyncio.sleep(interval)
