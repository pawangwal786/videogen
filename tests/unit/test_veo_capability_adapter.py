"""Unit tests for VeoCapabilityAdapter conforming to VideoGenerationProvider."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors

from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from app.stages.adapters.veo import VeoCapabilityAdapter
from app.stages.providers import (
    VideoGenerationProvider,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoReconciliationResult,
    VideoSubmission,
)


@pytest.fixture
def mock_genai_client():
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_videos = AsyncMock()
    client.aio.operations = MagicMock()
    client.aio.operations.get = AsyncMock()
    return client


def test_veo_capability_adapter_protocol_conformance(mock_genai_client):
    """Verify that VeoCapabilityAdapter satisfies the VideoGenerationProvider protocol."""
    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    assert isinstance(adapter, VideoGenerationProvider)


def test_veo_capability_adapter_init_empty_key():
    """Verify that empty or whitespace API key raises ModelConfigurationError."""
    with pytest.raises(ModelConfigurationError) as exc_info:
        VeoCapabilityAdapter(api_key="")
    assert "API key must not be empty" in str(exc_info.value)

    with pytest.raises(ModelConfigurationError) as exc_info_ws:
        VeoCapabilityAdapter(api_key="   ")
    assert "API key must not be empty" in str(exc_info_ws.value)


@pytest.mark.asyncio
async def test_veo_capability_adapter_submit_success(mock_genai_client):
    """Verify successful video generation submission returning VideoSubmission."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = False
    mock_genai_client.aio.models.generate_videos.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        client_request_id="shot-01-req",
        prompt="Cinematic drone shot over misty pine forest at sunrise.",
        negative_prompt="blurry, low quality",
        duration_seconds=5,
        aspect_ratio="16:9",
        seed=42,
    )

    submission = await adapter.submit(req)

    assert isinstance(submission, VideoSubmission)
    assert submission.provider_operation_id == "operations/veo-test-12345"
    assert submission.status == "PENDING"
    assert submission.estimated_duration_seconds == 50

    mock_genai_client.aio.models.generate_videos.assert_awaited_once()
    _, kwargs = mock_genai_client.aio.models.generate_videos.call_args
    assert kwargs["model"] == "veo-2.0-generate-001"
    assert kwargs["config"].duration_seconds == 5
    assert kwargs["config"].aspect_ratio == "16:9"
    assert kwargs["config"].negative_prompt == "blurry, low quality"
    assert kwargs["config"].seed == 42


@pytest.mark.asyncio
async def test_veo_capability_adapter_submit_10_seconds(mock_genai_client):
    """Verify Veo supports 10-second duration."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-10s"
    mock_op.done = True
    mock_genai_client.aio.models.generate_videos.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        client_request_id="shot-10s-req",
        prompt="Cinematic drone shot over misty pine forest at sunrise.",
        duration_seconds=10,
        aspect_ratio="9:16",
    )

    submission = await adapter.submit(req)
    assert submission.provider_operation_id == "operations/veo-test-10s"
    assert submission.status == "PROCESSING"


@pytest.mark.asyncio
async def test_veo_capability_adapter_submit_unsupported_duration(mock_genai_client):
    """Verify that unsupported duration raises ModelConfigurationError."""
    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        client_request_id="shot-invalid-dur",
        prompt="Cinematic drone shot over misty pine forest at sunrise.",
        duration_seconds=7,  # Veo only supports 5 or 10
        aspect_ratio="16:9",
    )

    with pytest.raises(ModelConfigurationError) as exc_info:
        await adapter.submit(req)
    assert "supports clip durations of [5, 10]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_capability_adapter_submit_auth_error(mock_genai_client):
    """Verify 401/403 maps to ModelAuthenticationError."""
    api_error = genai_errors.APIError(401, "API_KEY_INVALID: The provided API key is invalid.")
    mock_genai_client.aio.models.generate_videos.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="bad-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        client_request_id="shot-auth-fail",
        prompt="Cinematic drone shot over misty pine forest at sunrise.",
        duration_seconds=5,
    )

    with pytest.raises(ModelAuthenticationError) as exc_info:
        await adapter.submit(req)
    assert "authentication failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_capability_adapter_submit_rate_limit(mock_genai_client):
    """Verify 429 maps to ModelRateLimitError."""
    api_error = genai_errors.APIError(429, "RESOURCE_EXHAUSTED: Quota exceeded.")
    mock_genai_client.aio.models.generate_videos.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        client_request_id="shot-rate-limit",
        prompt="Cinematic drone shot over misty pine forest at sunrise.",
        duration_seconds=5,
    )

    with pytest.raises(ModelRateLimitError) as exc_info:
        await adapter.submit(req)
    assert "rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_capability_adapter_submit_timeout(mock_genai_client):
    """Verify 504 / timeout maps to ModelTimeoutError."""
    api_error = genai_errors.APIError(504, "DEADLINE_EXCEEDED: timed out.")
    mock_genai_client.aio.models.generate_videos.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        client_request_id="shot-timeout",
        prompt="Cinematic drone shot over misty pine forest at sunrise.",
        duration_seconds=5,
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        await adapter.submit(req)
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_in_progress(mock_genai_client):
    """Verify poll returning PROCESSING status when operation is still in progress."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = False
    mock_op.error = None
    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    res = await adapter.poll("operations/veo-test-12345")

    assert isinstance(res, VideoGenerationResult)
    assert res.provider_operation_id == "operations/veo-test-12345"
    assert res.status == "PROCESSING"
    assert res.video_bytes is None


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_completed_with_bytes(mock_genai_client):
    """Verify poll returning COMPLETED status with video bytes and URL."""
    mock_video = MagicMock()
    mock_video.video_bytes = b"\x00\x00\x00 ftypmp42"
    mock_video.uri = "https://storage.googleapis.com/veo-outputs/video.mp4"

    mock_gen_video = MagicMock()
    mock_gen_video.video = mock_video

    mock_response = MagicMock()
    mock_response.rai_media_filtered_reasons = None
    mock_response.generated_videos = [mock_gen_video]

    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = True
    mock_op.error = None
    mock_op.response = mock_response

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    res = await adapter.poll("operations/veo-test-12345")

    assert res.status == "COMPLETED"
    assert res.video_bytes == b"\x00\x00\x00 ftypmp42"
    assert res.download_url == "https://storage.googleapis.com/veo-outputs/video.mp4"
    assert res.error_message is None


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_safety_filtered(mock_genai_client):
    """Verify poll detecting RAI safety filtration returns FAILED with reasons."""
    mock_response = MagicMock()
    mock_response.rai_media_filtered_reasons = ["Violence or gore detected"]
    mock_response.generated_videos = []

    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = True
    mock_op.error = None
    mock_op.response = mock_response

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    res = await adapter.poll("operations/veo-test-12345")

    assert res.status == "FAILED"
    assert "filtered by safety guidelines" in res.error_message


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_op_error(mock_genai_client):
    """Verify provider operation error returns FAILED."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = True
    mock_op.error = "GPU device failure"

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    res = await adapter.poll("operations/veo-test-12345")

    assert res.status == "FAILED"
    assert "GPU device failure" in res.error_message


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_missing_response(mock_genai_client):
    """Verify done operation with no response payload returns FAILED."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = True
    mock_op.error = None
    mock_op.response = None
    mock_op.result = None

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    res = await adapter.poll("operations/veo-test-12345")

    assert res.status == "FAILED"
    assert "returned no response or result payload" in res.error_message


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_missing_videos(mock_genai_client):
    """Verify done operation with empty generated_videos returns FAILED."""
    mock_response = MagicMock()
    mock_response.rai_media_filtered_reasons = []
    mock_response.generated_videos = []

    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = True
    mock_op.error = None
    mock_op.response = mock_response

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)
    res = await adapter.poll("operations/veo-test-12345")

    assert res.status == "FAILED"
    assert "without generated video content" in res.error_message


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_until_complete_success(mock_genai_client):
    """Verify bounded polling until operation completes."""
    op_processing = MagicMock()
    op_processing.name = "operations/veo-test-12345"
    op_processing.done = False
    op_processing.error = None

    mock_response = MagicMock()
    mock_response.rai_media_filtered_reasons = None
    mock_video = MagicMock()
    mock_video.video_bytes = b"complete_bytes"
    mock_video.uri = "https://example.com/video.mp4"
    mock_video_item = MagicMock()
    mock_video_item.video = mock_video
    mock_response.generated_videos = [mock_video_item]

    op_completed = MagicMock()
    op_completed.name = "operations/veo-test-12345"
    op_completed.done = True
    op_completed.error = None
    op_completed.response = mock_response

    mock_genai_client.aio.operations.get.side_effect = [op_processing, op_completed]

    adapter = VeoCapabilityAdapter(
        api_key="valid-key",
        client=mock_genai_client,
        poll_interval_seconds=0.01,
        timeout_seconds=5.0,
    )

    result = await adapter.poll_until_complete("operations/veo-test-12345")

    assert result.status == "COMPLETED"
    assert result.video_bytes == b"complete_bytes"
    assert mock_genai_client.aio.operations.get.await_count == 2


@pytest.mark.asyncio
async def test_veo_capability_adapter_poll_until_complete_timeout(mock_genai_client):
    """Verify bounded polling raises ModelTimeoutError when timeout exceeded."""
    op_processing = MagicMock()
    op_processing.name = "operations/veo-test-12345"
    op_processing.done = False
    op_processing.error = None
    mock_genai_client.aio.operations.get.return_value = op_processing

    adapter = VeoCapabilityAdapter(
        api_key="valid-key",
        client=mock_genai_client,
        poll_interval_seconds=0.01,
        timeout_seconds=0.03,
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        await adapter.poll_until_complete("operations/veo-test-12345")
    assert "timed out after" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_resolved(mock_genai_client):
    """Verify reconcile with existing provider_operation_id returns RESOLVED."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = False
    mock_op.error = None
    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    recon = await adapter.reconcile(
        client_request_id="shot-01-req",
        provider_operation_id="operations/veo-test-12345",
    )

    assert isinstance(recon, VideoReconciliationResult)
    assert recon.status == "RESOLVED"
    assert recon.provider_operation_id == "operations/veo-test-12345"


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_confirmed_absent(mock_genai_client):
    """Verify reconcile with 404/NOT_FOUND returns CONFIRMED_ABSENT."""
    api_error = genai_errors.APIError(404, "NOT_FOUND: Operation does not exist.")
    mock_genai_client.aio.operations.get.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    recon = await adapter.reconcile(
        client_request_id="shot-01-req",
        provider_operation_id="operations/veo-nonexistent",
    )

    assert recon.status == "CONFIRMED_ABSENT"
    assert recon.provider_operation_id == "operations/veo-nonexistent"


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_transient_error_propagates(mock_genai_client):
    """Verify transient error during reconcile re-raises rather than marking absent/unresolved."""
    api_error = genai_errors.APIError(504, "DEADLINE_EXCEEDED: request timed out.")
    mock_genai_client.aio.operations.get.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelTimeoutError):
        await adapter.reconcile(
            client_request_id="shot-01-req",
            provider_operation_id="operations/veo-test-12345",
        )


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_auth_error_propagates(mock_genai_client):
    """Verify auth error during reconcile re-raises."""
    api_error = genai_errors.APIError(401, "API_KEY_INVALID")
    mock_genai_client.aio.operations.get.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelAuthenticationError):
        await adapter.reconcile(
            client_request_id="shot-01-req",
            provider_operation_id="operations/veo-test-12345",
        )


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_ambiguous_no_operation_id(mock_genai_client):
    """Verify Class C taxonomy: no operation ID returns UNRESOLVED to prevent double billing."""
    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    recon = await adapter.reconcile(client_request_id="shot-01-idempotency-key")

    assert recon.status == "UNRESOLVED"
    assert "does not support lookup by client_request_id" in recon.error_message
    mock_genai_client.aio.operations.get.assert_not_called()


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_with_operation_id_in_client_request_id(
    mock_genai_client,
):
    """Verify that client_request_id starting with operations/ is recognized as operation ID."""
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-embedded"
    mock_op.done = True
    mock_op.error = None
    mock_response = MagicMock()
    mock_response.rai_media_filtered_reasons = None
    mock_video = MagicMock()
    mock_video.video_bytes = b"bytes"
    mock_video.uri = None
    mock_gen_video = MagicMock()
    mock_gen_video.video = mock_video
    mock_response.generated_videos = [mock_gen_video]
    mock_op.response = mock_response

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    recon = await adapter.reconcile(client_request_id="operations/veo-test-embedded")

    assert recon.status == "RESOLVED"
    assert recon.provider_operation_id == "operations/veo-test-embedded"


@pytest.mark.asyncio
async def test_veo_capability_adapter_reconcile_generic_response_error(mock_genai_client):
    """Verify that unexpected provider error returns UNRESOLVED with error details."""
    api_error = genai_errors.APIError(500, "INTERNAL: Internal server failure.")
    mock_genai_client.aio.operations.get.side_effect = api_error

    adapter = VeoCapabilityAdapter(api_key="valid-key", client=mock_genai_client)

    recon = await adapter.reconcile(
        client_request_id="shot-01-req",
        provider_operation_id="operations/veo-test-12345",
    )

    assert isinstance(recon, VideoReconciliationResult)
    assert recon.status == "UNRESOLVED"
    assert "Internal server failure" in recon.error_message
