from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors

from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)
from app.models.gemini import GeminiTextModel


@pytest.fixture
def mock_genai_client():
    client = MagicMock()
    client.aio = MagicMock()
    client.aio.models = MagicMock()
    client.aio.models.generate_content = AsyncMock()
    return client


def test_gemini_empty_key_raises():
    with pytest.raises(ModelConfigurationError) as exc_info:
        GeminiTextModel(api_key="")
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_generate_success(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "Generated script content"
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    model = GeminiTextModel(
        api_key="valid-key",
        model_name="gemini-2.5-flash",
        client=mock_genai_client,
    )

    result = await model.generate("Create a video script about AI")
    assert result == "Generated script content"

    mock_genai_client.aio.models.generate_content.assert_awaited_once()
    call_kwargs = mock_genai_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["model"] == "gemini-2.5-flash"
    assert call_kwargs["contents"] == "Create a video script about AI"
    assert call_kwargs["config"] is None


@pytest.mark.asyncio
async def test_gemini_generate_with_system_prompt(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "Script with system instruction"
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    model = GeminiTextModel(
        api_key="valid-key",
        client=mock_genai_client,
    )

    result = await model.generate(
        "Generate script",
        system_prompt="You are a professional video producer.",
    )
    assert result == "Script with system instruction"

    call_kwargs = mock_genai_client.aio.models.generate_content.await_args.kwargs
    assert call_kwargs["config"] is not None
    assert call_kwargs["config"].system_instruction == "You are a professional video producer."


@pytest.mark.asyncio
async def test_gemini_empty_response_raises(mock_genai_client):
    mock_response = MagicMock()
    mock_response.text = "   "
    mock_genai_client.aio.models.generate_content.return_value = mock_response

    model = GeminiTextModel(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"


@pytest.mark.asyncio
async def test_gemini_authentication_error(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = genai_errors.ClientError(
        403, "API key not valid. Please pass a valid API key."
    )

    model = GeminiTextModel(api_key="invalid-key", client=mock_genai_client)

    with pytest.raises(ModelAuthenticationError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_gemini_rate_limit_error(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = genai_errors.ClientError(
        429, "Resource has been exhausted (quota)."
    )

    model = GeminiTextModel(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelRateLimitError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_gemini_timeout_error(mock_genai_client):
    async def slow_generate(*args, **kwargs):
        import asyncio

        await asyncio.sleep(0.5)
        return MagicMock(text="late")

    mock_genai_client.aio.models.generate_content.side_effect = slow_generate

    model = GeminiTextModel(
        api_key="valid-key",
        timeout_seconds=0.05,
        client=mock_genai_client,
    )

    with pytest.raises(ModelTimeoutError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_gemini_bad_request_400_maps_to_configuration_error(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = genai_errors.ClientError(
        400, "Invalid argument: model parameter out of range."
    )
    model = GeminiTextModel(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelConfigurationError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"
    assert "Invalid argument" in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_server_error_500_maps_to_retryable_response_error(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = genai_errors.APIError(
        500, "Internal backend failure."
    )
    model = GeminiTextModel(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_gemini_unmapped_api_error_maps_to_non_retryable_response_error(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = genai_errors.APIError(
        404, "Not Found: requested resource does not exist."
    )
    model = GeminiTextModel(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_gemini_unexpected_exception_maps_to_response_error(mock_genai_client):
    mock_genai_client.aio.models.generate_content.side_effect = RuntimeError("Broken pipe")
    model = GeminiTextModel(api_key="valid-key", client=mock_genai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert "Broken pipe" in str(exc_info.value)
