"""Unit tests for VeoProviderReconciler."""

from unittest.mock import AsyncMock

import pytest

from app.models.video import VideoModel, VideoOperation
from app.video.reconciler import VeoProviderReconciler


@pytest.mark.asyncio
async def test_veo_reconciler_success_with_operation_id():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.return_value = VideoOperation(
        operation_id="operations/veo-12345",
        provider="veo",
        status="processing",
    )

    reconciler = VeoProviderReconciler(mock_model)
    result = await reconciler.reconcile_submission(
        submission_token="wf-1:video:shot:1:1",
        provider_operation_id="operations/veo-12345",
    )
    assert result == "operations/veo-12345"
    mock_model.get_operation_status.assert_awaited_once_with("operations/veo-12345")


@pytest.mark.asyncio
async def test_veo_reconciler_operation_lookup_failure():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.side_effect = RuntimeError("Operation not found")

    reconciler = VeoProviderReconciler(mock_model)
    result = await reconciler.reconcile_submission(
        submission_token="wf-1:video:shot:1:1",
        provider_operation_id="operations/veo-expired",
    )
    assert result is None


@pytest.mark.asyncio
async def test_veo_reconciler_ambiguous_without_operation_id():
    mock_model = AsyncMock(spec=VideoModel)
    reconciler = VeoProviderReconciler(mock_model)

    result = await reconciler.reconcile_submission(
        submission_token="wf-1:video:shot:1:1",
        provider_operation_id=None,
    )
    assert result is None
    mock_model.get_operation_status.assert_not_awaited()
