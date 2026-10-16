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


def test_router_rejects_identical_primary_and_fallback(mock_gemini):
    """Verify router construction fails when primary and fallback are identical."""
    with pytest.raises(ModelConfigurationError) as exc_info:
        ModelRouter(
            providers={ModelProvider.GEMINI: mock_gemini},
            primary=ModelProvider.GEMINI,
            fallback=ModelProvider.GEMINI,
        )
    assert "Primary and fallback model providers cannot be identical" in str(exc_info.value)
    assert exc_info.value.provider == "gemini"


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


def test_router_model_name_and_model_identity_separation(mock_gemini, mock_openrouter):
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_openrouter.model_name = "google/gemini-2.5-flash"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    # API model name must be a clean provider identifier, NOT the composite identity
    assert router.model_name == "gemini-2.5-flash"
    # Canonical composite identity includes provider prefixes for fingerprinting and logging
    assert (
        router.model_identity
        == "gemini:gemini-2.5-flash:fallback=openrouter:google/gemini-2.5-flash"
    )


@pytest.mark.asyncio
async def test_router_generate_default_does_not_pass_composite_to_providers(
    mock_gemini, mock_openrouter
):
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_openrouter.model_name = "google/gemini-2.5-flash"
    mock_gemini.generate.side_effect = ModelTimeoutError("Gemini timeout", provider="gemini")
    mock_openrouter.generate.return_value = "Fallback OK"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    result = await router.generate("Test prompt")
    assert result == "Fallback OK"

    # Primary called without model_name override
    mock_gemini.generate.assert_awaited_once_with("Test prompt", system_prompt=None)
    # Fallback called without model_name override (uses its own configured default)
    mock_openrouter.generate.assert_awaited_once_with("Test prompt", system_prompt=None)


@pytest.mark.asyncio
async def test_router_explicit_override_routes_and_fallback_reverts_to_default(
    mock_gemini, mock_openrouter
):
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_openrouter.model_name = "google/gemini-2.5-flash"
    mock_gemini.generate.side_effect = ModelRateLimitError("Rate limited", provider="gemini")
    mock_openrouter.generate.return_value = "Fallback OK"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    result = await router.generate("Test prompt", model_name="gemini-2.5-pro")
    assert result == "Fallback OK"

    # Primary received the clean requested override
    mock_gemini.generate.assert_awaited_once_with(
        "Test prompt", system_prompt=None, model_name="gemini-2.5-pro"
    )
    # Fallback engaged with its own valid default model, NOT the Gemini-specific override
    mock_openrouter.generate.assert_awaited_once_with("Test prompt", system_prompt=None)


@pytest.mark.asyncio
async def test_router_provider_prefix_and_slashed_model_routing(mock_gemini, mock_openrouter):
    mock_gemini.generate.return_value = "Gemini OK"
    mock_openrouter.generate.return_value = "OpenRouter OK"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    # 1. Slashed model routes to OpenRouter
    res1 = await router.generate("Test", model_name="anthropic/claude-3-5-sonnet")
    assert res1 == "OpenRouter OK"
    mock_openrouter.generate.assert_awaited_with(
        "Test", system_prompt=None, model_name="anthropic/claude-3-5-sonnet"
    )

    # 2. Prefixed model routes to specified provider and strips prefix
    res2 = await router.generate("Test", model_name="gemini:gemini-2.5-flash")
    assert res2 == "Gemini OK"
    mock_gemini.generate.assert_awaited_with(
        "Test", system_prompt=None, model_name="gemini-2.5-flash"
    )


def test_router_effective_model_identity_default_and_overrides(mock_gemini, mock_openrouter):
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_openrouter.model_name = "google/gemini-2.5-flash"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    # 1. Default (None or 'default') returns composite identity with providers
    assert (
        router.effective_model_identity(None)
        == "gemini:gemini-2.5-flash:fallback=openrouter:google/gemini-2.5-flash"
    )
    assert (
        router.effective_model_identity("default")
        == "gemini:gemini-2.5-flash:fallback=openrouter:google/gemini-2.5-flash"
    )

    # 2. Explicit primary override includes target provider and fallback provider/model
    eff_override = router.effective_model_identity("gemini-2.5-pro")
    assert eff_override == "gemini:gemini-2.5-pro:fallback=openrouter:google/gemini-2.5-flash"

    # 3. Explicit prefixed override normalizes to the same target identity
    eff_prefixed = router.effective_model_identity("gemini:gemini-2.5-pro")
    assert eff_prefixed == "gemini:gemini-2.5-pro:fallback=openrouter:google/gemini-2.5-flash"

    # 4. Explicit fallback override sets primary as fallback
    eff_fallback_target = router.effective_model_identity("openrouter:anthropic/claude-3-opus")
    assert (
        eff_fallback_target == "openrouter:anthropic/claude-3-opus:fallback=gemini:gemini-2.5-flash"
    )


def test_router_rejects_unknown_provider_prefix(mock_gemini, mock_openrouter):
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(ModelConfigurationError, match="Unsupported provider prefix 'anthropic'"):
        router._resolve_provider_for_override("anthropic:claude-3-opus")


def test_router_rejects_empty_provider_prefix(mock_gemini, mock_openrouter):
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(ModelConfigurationError, match="Empty provider prefix"):
        router._resolve_provider_for_override(":gemini-2.5-flash")


def test_router_rejects_empty_model_name_after_prefix(mock_gemini, mock_openrouter):
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(
        ModelConfigurationError, match="Empty model name after provider prefix 'gemini'"
    ):
        router._resolve_provider_for_override("gemini:")

    with pytest.raises(
        ModelConfigurationError, match="Empty model name after provider prefix 'openrouter'"
    ):
        router._resolve_provider_for_override("openrouter:   ")


def test_router_rejects_empty_or_whitespace_override(mock_gemini, mock_openrouter):
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    with pytest.raises(ModelConfigurationError, match="Model override cannot be empty"):
        router._resolve_provider_for_override("")

    with pytest.raises(ModelConfigurationError, match="Model override cannot be empty"):
        router._resolve_provider_for_override("   ")


def test_router_rejects_unconfigured_provider_prefix(mock_gemini):
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
        },
        primary=ModelProvider.GEMINI,
        fallback=None,
    )

    with pytest.raises(
        ModelConfigurationError,
        match="Provider 'openrouter' requested in model override 'openrouter:foo' is not configured",
    ):
        router._resolve_provider_for_override("openrouter:foo")


def test_router_identity_changes_when_primary_provider_changes_with_identical_model_identifier(
    mock_gemini, mock_openrouter
):
    """Verify that changing the primary provider changes canonical identity even with identical model identifiers."""
    mock_gemini.model_name = "common-model-v1"
    mock_openrouter.model_name = "common-model-v1"

    router_gemini = ModelRouter(
        providers={ModelProvider.GEMINI: mock_gemini},
        primary=ModelProvider.GEMINI,
        fallback=None,
    )
    router_openrouter = ModelRouter(
        providers={ModelProvider.OPENROUTER: mock_openrouter},
        primary=ModelProvider.OPENROUTER,
        fallback=None,
    )

    assert router_gemini.model_name == "common-model-v1"
    assert router_openrouter.model_name == "common-model-v1"

    assert router_gemini.model_identity == "gemini:common-model-v1"
    assert router_openrouter.model_identity == "openrouter:common-model-v1"
    assert router_gemini.model_identity != router_openrouter.model_identity


def test_router_single_provider_configuration_encodes_no_fallback(mock_gemini):
    """Verify that a single-provider router encodes no fallback in its identity or overrides."""
    mock_gemini.model_name = "gemini-2.5-flash"
    router = ModelRouter(
        providers={ModelProvider.GEMINI: mock_gemini},
        primary=ModelProvider.GEMINI,
        fallback=None,
    )

    # 1. Default identity has no fallback clause
    assert router.model_identity == "gemini:gemini-2.5-flash"
    assert ":fallback=" not in router.model_identity

    # 2. Effective identity for explicit override has no fallback clause
    eff_override = router.effective_model_identity("gemini-2.5-pro")
    assert eff_override == "gemini:gemini-2.5-pro"
    assert ":fallback=" not in eff_override


@pytest.mark.asyncio
async def test_router_fingerprint_and_execution_fallback_routing_alignment(
    mock_gemini, mock_openrouter
):
    """Verify that fingerprint identity derivation and execution routing resolve identical fallback providers."""
    mock_gemini.model_name = "gemini-2.5-flash"
    mock_openrouter.model_name = "google/gemini-2.5-flash"

    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: mock_gemini,
            ModelProvider.OPENROUTER: mock_openrouter,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    # 1. Default call: primary is GEMINI, fallback is OPENROUTER
    assert router._resolve_fallback_provider(router.primary) == ModelProvider.OPENROUTER
    mock_gemini.generate.side_effect = ModelTimeoutError("Timeout", provider="gemini")
    mock_openrouter.generate.return_value = "Recovered default"
    res1 = await router.generate("Prompt")
    assert res1 == "Recovered default"
    mock_openrouter.generate.assert_awaited_with("Prompt", system_prompt=None)

    # 2. Override targeting fallback provider: fallback flips to primary (GEMINI)
    assert router._resolve_fallback_provider(ModelProvider.OPENROUTER) == ModelProvider.GEMINI
    mock_openrouter.generate.side_effect = ModelTimeoutError("Timeout", provider="openrouter")
    mock_gemini.generate.side_effect = None
    mock_gemini.generate.return_value = "Recovered by primary"
    res2 = await router.generate("Prompt", model_name="openrouter:anthropic/claude-3-opus")
    assert res2 == "Recovered by primary"
    mock_gemini.generate.assert_awaited_with("Prompt", system_prompt=None)
