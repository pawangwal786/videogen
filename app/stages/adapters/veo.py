"""Google Veo capability adapter implementing VideoGenerationProvider.

Conforms to the typed VideoGenerationProvider capability interface in
`app.stages.providers`. Zero raw dicts and zero `Any` are permitted across
this boundary.
"""

import asyncio
import time
from typing import Literal

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
from app.stages.providers import (
    VideoGenerationProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReconciliationResult,
    VideoSubmission,
)

logger = get_logger(__name__)

SUPPORTED_VEO_DURATIONS: frozenset[int] = frozenset({5, 10})


class VeoCapabilityAdapter(VideoGenerationProvider):
    """Google Veo video generation capability adapter using the official google-genai SDK.

    Provides:
    - Asynchronous generation submission via `submit()`.
    - Non-blocking single status inspection via `poll()`.
    - Bounded polling with timeout handling via `poll_until_complete()`.
    - Idempotent crash recovery and operation reconciliation via `reconcile()`.
    - Normalized typed exception hierarchy.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "veo-2.0-generate-001",
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 5.0,
        max_poll_attempts: int = 60,
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
        self.max_poll_attempts = max_poll_attempts
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

    async def submit(self, request: VideoGenerationRequest) -> VideoSubmission:
        """Asynchronously submit a video generation request to Google Veo."""
        if request.duration_seconds not in SUPPORTED_VEO_DURATIONS:
            raise ModelConfigurationError(
                f"Veo supports clip durations of {sorted(SUPPORTED_VEO_DURATIONS)} seconds; "
                f"got {request.duration_seconds}s.",
                provider="veo",
            )

        logger.info(
            "model.veo.submit_start",
            model=self.model_name,
            client_request_id=request.client_request_id,
            duration=request.duration_seconds,
            aspect_ratio=request.aspect_ratio,
        )

        source = types.GenerateVideosSource(prompt=request.prompt)
        config = types.GenerateVideosConfig(
            aspect_ratio=request.aspect_ratio,
            duration_seconds=request.duration_seconds,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
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
                client_request_id=request.client_request_id,
                error=str(normalized),
            )
            raise normalized from e

        operation_id = op.name or f"veo-op-{request.client_request_id}"
        submission_status: Literal["PENDING", "PROCESSING"] = (
            "PROCESSING" if getattr(op, "done", False) else "PENDING"
        )

        logger.info(
            "model.veo.submitted",
            model=self.model_name,
            client_request_id=request.client_request_id,
            operation_id=operation_id,
            done=bool(getattr(op, "done", False)),
        )

        return VideoSubmission(
            provider_operation_id=operation_id,
            status=submission_status,
            estimated_duration_seconds=request.duration_seconds * 10,
        )

    async def poll(self, operation_id: str) -> VideoGenerationResult:
        """Inspect the current state of an in-flight video generation operation."""
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

        # Check for provider failure
        if getattr(op, "error", None):
            error_msg = str(op.error)
            logger.error(
                "model.veo.operation_failed",
                operation_id=operation_id,
                error=error_msg,
            )
            return VideoGenerationResult(
                provider_operation_id=operation_id,
                status="FAILED",
                error_message=error_msg,
            )

        if not getattr(op, "done", False):
            return VideoGenerationResult(
                provider_operation_id=operation_id,
                status="PROCESSING",
            )

        # Operation is completed: extract generated video
        result = getattr(op, "response", None) or getattr(op, "result", None)
        if not result:
            return VideoGenerationResult(
                provider_operation_id=operation_id,
                status="FAILED",
                error_message="Operation marked done but returned no response or result payload.",
            )

        # Check safety / RAI filtering
        filtered_reasons = getattr(result, "rai_media_filtered_reasons", None)
        if filtered_reasons:
            reasons_str = ", ".join(filtered_reasons)
            logger.warning(
                "model.veo.content_filtered",
                operation_id=operation_id,
                reasons=reasons_str,
            )
            return VideoGenerationResult(
                provider_operation_id=operation_id,
                status="FAILED",
                error_message=f"Video content was filtered by safety guidelines: {reasons_str}",
            )

        generated_videos = getattr(result, "generated_videos", None)
        if not generated_videos or not generated_videos[0].video:
            return VideoGenerationResult(
                provider_operation_id=operation_id,
                status="FAILED",
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

        return VideoGenerationResult(
            provider_operation_id=operation_id,
            video_bytes=video_bytes,
            download_url=video_uri,
            status="COMPLETED",
        )

    async def poll_until_complete(
        self,
        operation_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
    ) -> VideoGenerationResult:
        """Bounded polling until the operation reaches a terminal state (COMPLETED or FAILED).

        Raises ModelTimeoutError if polling exceeds timeout_seconds.
        """
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
                    f"Polling Veo operation {operation_id} timed out after {elapsed:.1f}s "
                    f"(timeout: {timeout}s).",
                    provider="veo",
                )

            res = await self.poll(operation_id)
            if res.status in {"COMPLETED", "FAILED"}:
                return res

            await asyncio.sleep(interval)

    async def reconcile(
        self,
        client_request_id: str,
        provider_operation_id: str | None = None,
    ) -> VideoReconciliationResult:
        """Reconcile in-flight operation for crash recovery and deduplication.

        Adheres strictly to the Class C recovery taxonomy:
        - If provider_operation_id is available, verifies operation existence with the provider.
        - If the provider reports NOT_FOUND (404), returns CONFIRMED_ABSENT.
        - If only client_request_id is known without an operation ID, returns UNRESOLVED
          because Veo does not provide client request token reverse lookup.
        - Transient errors (timeout, rate limit) and auth errors are re-raised.
        """
        target_op_id = provider_operation_id
        if not target_op_id and client_request_id.startswith("operations/"):
            target_op_id = client_request_id

        if target_op_id:
            try:
                poll_result = await self.poll(target_op_id)
                logger.info(
                    "veo.reconciliation.operation_verified",
                    provider_operation_id=target_op_id,
                    status=poll_result.status,
                )
                return VideoReconciliationResult(
                    status="RESOLVED",
                    provider_operation_id=target_op_id,
                )
            except (ModelTimeoutError, ModelRateLimitError, TimeoutError) as exc:
                logger.warning(
                    "veo.reconciliation.transient_error",
                    provider_operation_id=target_op_id,
                    error=str(exc),
                )
                raise
            except (ModelAuthenticationError, ModelConfigurationError) as exc:
                logger.error(
                    "veo.reconciliation.auth_config_error",
                    provider_operation_id=target_op_id,
                    error=str(exc),
                )
                raise
            except ModelResponseError as exc:
                err_str = str(exc)
                if "404" in err_str or "NOT_FOUND" in err_str:
                    logger.info(
                        "veo.reconciliation.confirmed_absent",
                        provider_operation_id=target_op_id,
                    )
                    return VideoReconciliationResult(
                        status="CONFIRMED_ABSENT",
                        provider_operation_id=target_op_id,
                        error_message=err_str,
                    )
                logger.warning(
                    "veo.reconciliation.unresolved_response_error",
                    provider_operation_id=target_op_id,
                    error=err_str,
                )
                return VideoReconciliationResult(
                    status="UNRESOLVED",
                    provider_operation_id=target_op_id,
                    error_message=err_str,
                )

        logger.warning(
            "veo.reconciliation.ambiguous_no_operation_id",
            client_request_id=client_request_id,
        )
        return VideoReconciliationResult(
            status="UNRESOLVED",
            error_message="Veo provider does not support lookup by client_request_id without provider_operation_id",
        )


__all__ = [
    "SUPPORTED_VEO_DURATIONS",
    "VeoCapabilityAdapter",
]
