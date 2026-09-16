"""Unit tests for VeoProviderReconciler."""

from unittest.mock import AsyncMock

import pytest

from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)
from app.models.video import VideoModel, VideoOperation
from app.orchestration.models import ReconciliationStatus
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
    outcome = await reconciler.reconcile_submission(
        submission_token="wf-1:video:shot:1:1",
        provider_operation_id="operations/veo-12345",
    )
    assert outcome.status == ReconciliationStatus.RESOLVED
    assert outcome.provider_operation_id == "operations/veo-12345"
    assert outcome.metadata["status"] == "processing"
    mock_model.get_operation_status.assert_awaited_once_with("operations/veo-12345")


@pytest.mark.asyncio
async def test_veo_reconciler_operation_lookup_failure_returns_unresolved():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.side_effect = ModelResponseError("Operation not found")

    reconciler = VeoProviderReconciler(mock_model)
    outcome = await reconciler.reconcile_submission(
        submission_token="wf-1:video:shot:1:1",
        provider_operation_id="operations/veo-expired",
    )
    assert outcome.status == ReconciliationStatus.UNRESOLVED
    assert outcome.provider_operation_id == "operations/veo-expired"
    assert outcome.status != ReconciliationStatus.CONFIRMED_ABSENT


@pytest.mark.asyncio
async def test_veo_reconciler_ambiguous_without_operation_id():
    mock_model = AsyncMock(spec=VideoModel)
    reconciler = VeoProviderReconciler(mock_model)

    outcome = await reconciler.reconcile_submission(
        submission_token="wf-1:video:shot:1:1",
        provider_operation_id=None,
    )
    assert outcome.status == ReconciliationStatus.UNRESOLVED
    assert outcome.status != ReconciliationStatus.CONFIRMED_ABSENT
    mock_model.get_operation_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_veo_reconciler_transient_timeout_propagates():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.side_effect = ModelTimeoutError("Timeout polling provider")

    reconciler = VeoProviderReconciler(mock_model)
    with pytest.raises(ModelTimeoutError):
        await reconciler.reconcile_submission(
            submission_token="wf-1:video:shot:1:1",
            provider_operation_id="operations/veo-12345",
        )


@pytest.mark.asyncio
async def test_veo_reconciler_transient_ratelimit_propagates():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.side_effect = ModelRateLimitError("Rate limit exceeded")

    reconciler = VeoProviderReconciler(mock_model)
    with pytest.raises(ModelRateLimitError):
        await reconciler.reconcile_submission(
            submission_token="wf-1:video:shot:1:1",
            provider_operation_id="operations/veo-12345",
        )


@pytest.mark.asyncio
async def test_veo_reconciler_auth_error_propagates():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.side_effect = ModelAuthenticationError("Invalid API key")

    reconciler = VeoProviderReconciler(mock_model)
    with pytest.raises(ModelAuthenticationError):
        await reconciler.reconcile_submission(
            submission_token="wf-1:video:shot:1:1",
            provider_operation_id="operations/veo-12345",
        )


@pytest.mark.asyncio
async def test_veo_reconciler_config_error_propagates():
    mock_model = AsyncMock(spec=VideoModel)
    mock_model.get_operation_status.side_effect = ModelConfigurationError("Invalid configuration")

    reconciler = VeoProviderReconciler(mock_model)
    with pytest.raises(ModelConfigurationError):
        await reconciler.reconcile_submission(
            submission_token="wf-1:video:shot:1:1",
            provider_operation_id="operations/veo-12345",
        )
