from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from google.genai import errors as genai_errors

from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from app.models.veo import VeoVideoModel
from app.models.video import VideoGenerationRequest


@pytest.fixture
def mock_genai_client():
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_videos = AsyncMock()
    client.aio.operations = MagicMock()
    client.aio.operations.get = AsyncMock()
    return client


def test_veo_adapter_initialization_empty_key():
    with pytest.raises(ModelConfigurationError) as exc_info:
        VeoVideoModel(api_key="")
    assert "API key must not be empty" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_adapter_submit_generation_success(mock_genai_client):
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = False
    mock_genai_client.aio.models.generate_videos.return_value = mock_op

    adapter = VeoVideoModel(api_key="test-api-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        workflow_id=uuid4(),
        shot_number=1,
        prompt="Cinematic drone aerial over foggy mountain ridge at sunrise.",
        duration_seconds=5,
        aspect_ratio="16:9",
    )

    op = await adapter.submit_generation(req)

    assert op.operation_id == "operations/veo-test-12345"
    assert op.status == "submitted"
    assert op.done is False
    assert op.provider == "veo"

    mock_genai_client.aio.models.generate_videos.assert_awaited_once()


@pytest.mark.asyncio
async def test_veo_adapter_submit_unsupported_duration(mock_genai_client):
    adapter = VeoVideoModel(api_key="test-api-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        workflow_id=uuid4(),
        shot_number=1,
        prompt="Cinematic drone aerial over foggy mountain ridge at sunrise.",
        duration_seconds=7,  # Veo only supports 5 or 10
        aspect_ratio="16:9",
    )

    with pytest.raises(ModelConfigurationError) as exc_info:
        await adapter.submit_generation(req)
    assert "Veo supports clip durations of [5, 10]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_adapter_submit_auth_error(mock_genai_client):
    api_error = genai_errors.APIError(401, "API_KEY_INVALID: The provided API key is invalid.")
    mock_genai_client.aio.models.generate_videos.side_effect = api_error

    adapter = VeoVideoModel(api_key="bad-api-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        workflow_id=uuid4(),
        shot_number=1,
        prompt="Cinematic drone aerial over foggy mountain ridge at sunrise.",
        duration_seconds=5,
    )

    with pytest.raises(ModelAuthenticationError) as exc_info:
        await adapter.submit_generation(req)
    assert "authentication failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_adapter_submit_rate_limit(mock_genai_client):
    api_error = genai_errors.APIError(429, "RESOURCE_EXHAUSTED: Quota exceeded.")
    mock_genai_client.aio.models.generate_videos.side_effect = api_error

    adapter = VeoVideoModel(api_key="test-api-key", client=mock_genai_client)

    req = VideoGenerationRequest(
        workflow_id=uuid4(),
        shot_number=1,
        prompt="Cinematic drone aerial over foggy mountain ridge at sunrise.",
        duration_seconds=5,
    )

    with pytest.raises(ModelRateLimitError) as exc_info:
        await adapter.submit_generation(req)
    assert "rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_veo_adapter_poll_processing(mock_genai_client):
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = False
    mock_op.error = None
    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoVideoModel(api_key="test-api-key", client=mock_genai_client)
    op = await adapter.get_operation_status("operations/veo-test-12345")

    assert op.status == "processing"
    assert op.done is False


@pytest.mark.asyncio
async def test_veo_adapter_poll_completed_with_bytes(mock_genai_client):
    mock_video = MagicMock()
    mock_video.video_bytes = b"\x00\x00\x00 ftypmp42"
    mock_video.uri = "https://storage.googleapis.com/test-bucket/video.mp4"

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

    adapter = VeoVideoModel(api_key="test-api-key", client=mock_genai_client)
    op = await adapter.get_operation_status("operations/veo-test-12345")

    assert op.status == "completed"
    assert op.done is True
    assert op.video_bytes == b"\x00\x00\x00 ftypmp42"
    assert op.video_uri == "https://storage.googleapis.com/test-bucket/video.mp4"


@pytest.mark.asyncio
async def test_veo_adapter_poll_safety_filter(mock_genai_client):
    mock_response = MagicMock()
    mock_response.rai_media_filtered_reasons = ["Violence or gore detected"]
    mock_response.generated_videos = []

    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = True
    mock_op.error = None
    mock_op.response = mock_response

    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoVideoModel(api_key="test-api-key", client=mock_genai_client)
    op = await adapter.get_operation_status("operations/veo-test-12345")

    assert op.status == "failed"
    assert op.done is True
    assert "filtered by safety guidelines" in op.error_message


@pytest.mark.asyncio
async def test_veo_adapter_poll_until_complete_timeout(mock_genai_client):
    mock_op = MagicMock()
    mock_op.name = "operations/veo-test-12345"
    mock_op.done = False
    mock_op.error = None
    mock_genai_client.aio.operations.get.return_value = mock_op

    adapter = VeoVideoModel(
        api_key="test-api-key",
        client=mock_genai_client,
        poll_interval_seconds=0.01,
        timeout_seconds=0.05,
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        await adapter.poll_until_complete("operations/veo-test-12345")
    assert "timed out" in str(exc_info.value)
