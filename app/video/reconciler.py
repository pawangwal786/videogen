"""Veo provider operation reconciler."""

from app.logging import get_logger
from app.models.video import VideoModel

logger = get_logger(__name__)


class VeoProviderReconciler:
    """Reconciles in-flight or crashed Veo operations with Google's API.

    Adheres strictly to the architectural directive:
    - Never invent provider guarantees.
    - If provider_operation_id exists, polls provider to verify operation validity.
    - If only submission_token exists without an operation_id, Veo API has no lookup token
      mechanism; returns None to indicate ambiguous submission requiring manual review.
    """

    def __init__(self, video_model: VideoModel) -> None:
        self._video_model = video_model

    async def reconcile_submission(
        self,
        submission_token: str | None,
        provider_operation_id: str | None,
    ) -> str | None:
        """Query provider to confirm whether the operation is active."""
        if provider_operation_id:
            try:
                op = await self._video_model.get_operation_status(provider_operation_id)
                logger.info(
                    "veo.reconciliation.operation_verified",
                    provider_operation_id=provider_operation_id,
                    status=op.status,
                )
                return op.operation_id
            except Exception as exc:
                logger.warning(
                    "veo.reconciliation.operation_check_failed",
                    provider_operation_id=provider_operation_id,
                    error=str(exc),
                )
                return None

        # Veo API does not provide client request token search
        logger.warning(
            "veo.reconciliation.ambiguous_no_operation_id",
            submission_token=submission_token,
        )
        return None
