from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)
from app.models.openrouter import OpenRouterTextModel


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def test_openrouter_empty_key_raises():
    with pytest.raises(ModelConfigurationError) as exc_info:
        OpenRouterTextModel(api_key="")
    assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
async def test_openrouter_generate_success(mock_openai_client):
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenRouter response text"
    mock_response = MagicMock(choices=[mock_choice])
    mock_openai_client.chat.completions.create.return_value = mock_response

    model = OpenRouterTextModel(
        api_key="valid-key",
        model_name="google/gemini-2.5-flash",
        client=mock_openai_client,
    )

    result = await model.generate(
        "Generate a summary",
        system_prompt="Be concise.",
    )
    assert result == "OpenRouter response text"

    call_kwargs = mock_openai_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["model"] == "google/gemini-2.5-flash"
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Generate a summary"},
    ]


@pytest.mark.asyncio
async def test_openrouter_empty_response_raises(mock_openai_client):
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_response = MagicMock(choices=[mock_choice])
    mock_openai_client.chat.completions.create.return_value = mock_response

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"


@pytest.mark.asyncio
async def test_openrouter_auth_error(mock_openai_client):
    fake_request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    fake_response = httpx.Response(401, request=fake_request)
    mock_openai_client.chat.completions.create.side_effect = AuthenticationError(
        "Invalid API key", response=fake_response, body=None
    )

    model = OpenRouterTextModel(api_key="invalid-key", client=mock_openai_client)

    with pytest.raises(ModelAuthenticationError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_openrouter_rate_limit_error(mock_openai_client):
    fake_request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    fake_response = httpx.Response(429, request=fake_request)
    mock_openai_client.chat.completions.create.side_effect = RateLimitError(
        "Rate limit exceeded", response=fake_response, body=None
    )

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelRateLimitError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_openrouter_timeout_error(mock_openai_client):
    fake_request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    mock_openai_client.chat.completions.create.side_effect = APITimeoutError(request=fake_request)

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelTimeoutError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_openrouter_bad_request_error(mock_openai_client):
    fake_request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    fake_response = httpx.Response(400, request=fake_request)
    mock_openai_client.chat.completions.create.side_effect = BadRequestError(
        "Invalid model parameters", response=fake_response, body=None
    )

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelConfigurationError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_openrouter_server_error_retryable(mock_openai_client):
    fake_request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    fake_response = httpx.Response(503, request=fake_request)
    mock_openai_client.chat.completions.create.side_effect = InternalServerError(
        "Service Unavailable", response=fake_response, body=None
    )

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_openrouter_empty_choices_raises(mock_openai_client):
    mock_response = MagicMock(choices=[])
    mock_openai_client.chat.completions.create.return_value = mock_response

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert "no choices" in str(exc_info.value)
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_openrouter_generic_api_error_non_retryable(mock_openai_client):
    from openai import APIError

    fake_request = httpx.Request("POST", "https://openrouter.ai/api/v1")
    mock_openai_client.chat.completions.create.side_effect = APIError(
        "Unprocessable entity", request=fake_request, body=None
    )

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert exc_info.value.provider == "openrouter"
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_openrouter_unexpected_exception(mock_openai_client):
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("Kernel panic")

    model = OpenRouterTextModel(api_key="valid-key", client=mock_openai_client)

    with pytest.raises(ModelResponseError) as exc_info:
        await model.generate("Hello")
    assert "Kernel panic" in str(exc_info.value)
    assert exc_info.value.provider == "openrouter"
