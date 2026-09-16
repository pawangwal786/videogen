"""Veo provider operation reconciler."""

from app.logging import get_logger
from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)
from app.models.video import VideoModel
from app.orchestration.models import ReconciliationOutcome, ReconciliationStatus

logger = get_logger(__name__)


class VeoProviderReconciler:
    """Reconciles in-flight or crashed Veo operations with Google's API.

    Adheres strictly to the architectural directive:
    - Never invent provider guarantees.
    - If provider_operation_id exists, polls provider to verify operation validity.
    - If only submission_token exists without an operation_id, Veo API has no lookup token
      mechanism; returns UNRESOLVED to avoid duplicate billing. Veo never manufactures CONFIRMED_ABSENT.
    - Transient provider failures (ModelTimeoutError, ModelRateLimitError) and configuration/auth errors
      are propagated so callers can retry or surface system defects rather than converting transient issues
      into permanent manual-reconciliation failures.
    - Provider response errors (e.g. unknown operation ID) yield UNRESOLVED.
    """

    def __init__(self, video_model: VideoModel) -> None:
        self._video_model = video_model

    async def reconcile_submission(
        self,
        submission_token: str | None,
        provider_operation_id: str | None,
    ) -> ReconciliationOutcome:
        """Query provider to confirm whether the operation is active."""
        if provider_operation_id:
            try:
                op = await self._video_model.get_operation_status(provider_operation_id)
                logger.info(
                    "veo.reconciliation.operation_verified",
                    provider_operation_id=provider_operation_id,
                    status=op.status,
                    done=op.done,
                )
                return ReconciliationOutcome(
                    status=ReconciliationStatus.RESOLVED,
                    provider_operation_id=op.operation_id,
                    metadata={"status": op.status, "done": op.done},
                )
            except (ModelTimeoutError, ModelRateLimitError, TimeoutError) as exc:
                logger.warning(
                    "veo.reconciliation.transient_error",
                    provider_operation_id=provider_operation_id,
                    error=str(exc),
                )
                raise
            except (ModelAuthenticationError, ModelConfigurationError) as exc:
                logger.error(
                    "veo.reconciliation.auth_config_error",
                    provider_operation_id=provider_operation_id,
                    error=str(exc),
                )
                raise
            except ModelResponseError as exc:
                logger.warning(
                    "veo.reconciliation.unresolved_response_error",
                    provider_operation_id=provider_operation_id,
                    error=str(exc),
                )
                return ReconciliationOutcome(
                    status=ReconciliationStatus.UNRESOLVED,
                    provider_operation_id=provider_operation_id,
                    error_message=str(exc),
                )

        # Veo API does not provide client request token search; cannot authoritatively confirm absence
        logger.warning(
            "veo.reconciliation.ambiguous_no_operation_id",
            submission_token=submission_token,
        )
        return ReconciliationOutcome(
            status=ReconciliationStatus.UNRESOLVED,
            error_message="Veo provider does not support lookup by submission_token without provider_operation_id",
        )
