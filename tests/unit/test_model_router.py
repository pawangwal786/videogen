from unittest.mock import AsyncMock

import pytest

from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)
from app.models.router import ModelProvider, ModelRouter


@pytest.fixture
def mock_gemini():
    mock = AsyncMock()
    mock.generate = AsyncMock()
    return mock


@pytest.fixture
def mock_openrouter():
    mock = AsyncMock()
    mock.generate = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_router_primary_success(mock_gemini, mock_openrouter):
    mock_gemini.generate.return_value = "Gemini response"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    result = await router.generate("Test prompt")
    assert result == "Gemini response"
    mock_gemini.generate.assert_awaited_once_with("Test prompt", system_prompt=None)
    mock_openrouter.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_transient_failure_falls_back_to_openrouter(mock_gemini, mock_openrouter):
    # Gemini times out (transient)
    mock_gemini.generate.side_effect = ModelTimeoutError("Gemini timeout", provider="gemini")
    mock_openrouter.generate.return_value = "OpenRouter fallback response"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    result = await router.generate("Test prompt", system_prompt="System instructions")
    assert result == "OpenRouter fallback response"
    mock_gemini.generate.assert_awaited_once()
    mock_openrouter.generate.assert_awaited_once_with(
        "Test prompt", system_prompt="System instructions"
    )


@pytest.mark.asyncio
async def test_router_rate_limit_falls_back_to_openrouter(mock_gemini, mock_openrouter):
    # Gemini rate limited (transient)
    mock_gemini.generate.side_effect = ModelRateLimitError("Gemini rate limit", provider="gemini")
    mock_openrouter.generate.return_value = "OpenRouter fallback response"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    result = await router.generate("Test prompt")
    assert result == "OpenRouter fallback response"
    mock_openrouter.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_router_auth_failure_does_not_fallback(mock_gemini, mock_openrouter):
    # Gemini auth failure (permanent)
    mock_gemini.generate.side_effect = ModelAuthenticationError(
        "Invalid Gemini API key", provider="gemini"
    )

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(ModelAuthenticationError):
        await router.generate("Test prompt")

    mock_gemini.generate.assert_awaited_once()
    # MUST NOT call fallback on auth errors
    mock_openrouter.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_bad_config_does_not_fallback(mock_gemini, mock_openrouter):
    # Gemini configuration/bad request failure (permanent)
    mock_gemini.generate.side_effect = ModelConfigurationError(
        "Invalid model name", provider="gemini"
    )

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(ModelConfigurationError):
        await router.generate("Test prompt")

    mock_openrouter.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_router_fallback_also_fails_raises_error(mock_gemini, mock_openrouter):
    mock_gemini.generate.side_effect = ModelTimeoutError("Gemini timeout", provider="gemini")
    mock_openrouter.generate.side_effect = ModelResponseError(
        "OpenRouter 500 error", provider="openrouter", retryable=True
    )

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(ModelResponseError) as exc_info:
        await router.generate("Test prompt")
    assert exc_info.value.provider == "openrouter"


def test_router_missing_primary_provider():
    with pytest.raises(ModelConfigurationError) as exc_info:
        ModelRouter(
            providers={},
            primary=ModelProvider.GEMINI,
        )
    assert "Primary model provider" in str(exc_info.value)


def test_router_missing_fallback_provider(mock_gemini):
    with pytest.raises(ModelConfigurationError) as exc_info:
        ModelRouter(
            providers={ModelProvider.GEMINI: mock_gemini},
            primary=ModelProvider.GEMINI,
            fallback=ModelProvider.OPENROUTER,
        )
    assert "Fallback model provider" in str(exc_info.value)


def test_router_from_settings_dual_providers():
    from pydantic import SecretStr

    from app.config.settings import Settings

    settings = Settings(
        gemini_api_key=SecretStr("fake-gemini-key"),
        openrouter_api_key=SecretStr("fake-openrouter-key"),
        videogen_primary_text_provider=ModelProvider.GEMINI,
        videogen_fallback_text_provider=ModelProvider.OPENROUTER,
    )

    router = ModelRouter.from_settings(settings)
    assert router.primary == ModelProvider.GEMINI
    assert router.fallback == ModelProvider.OPENROUTER
    assert ModelProvider.GEMINI in router._providers
    assert ModelProvider.OPENROUTER in router._providers


def test_router_from_settings_single_provider():
    from pydantic import SecretStr

    from app.config.settings import Settings

    settings = Settings(
        gemini_api_key=SecretStr("fake-gemini-key"),
        videogen_primary_text_provider=ModelProvider.GEMINI,
        videogen_fallback_text_provider=None,
    )

    router = ModelRouter.from_settings(settings)
    assert router.primary == ModelProvider.GEMINI
    assert router.fallback is None
    assert ModelProvider.GEMINI in router._providers
    assert ModelProvider.OPENROUTER not in router._providers
