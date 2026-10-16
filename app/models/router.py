from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.settings import Settings
from app.logging import get_logger
from app.models.base import TextModel
from app.models.errors import (
    ModelConfigurationError,
    ModelError,
)
from app.models.gemini import GeminiTextModel
from app.models.openrouter import OpenRouterTextModel

logger = get_logger(__name__)


class ModelProvider(StrEnum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class ModelRouter:
    """Task-oriented model router with conservative fallback policy.

    The router does NOT perform loops or retries; it evaluates the primary
    provider's normalized error and routes to fallback only if the failure
    is transient and retryable.
    """

    def __init__(
        self,
        providers: dict[ModelProvider, TextModel],
        primary: ModelProvider = ModelProvider.GEMINI,
        fallback: ModelProvider | None = ModelProvider.OPENROUTER,
    ) -> None:
        if primary not in providers:
            raise ModelConfigurationError(
                f"Primary model provider '{primary}' is not registered in router.",
                provider=str(primary),
            )
        if fallback and fallback not in providers:
            raise ModelConfigurationError(
                f"Fallback model provider '{fallback}' is not registered in router.",
                provider=str(fallback),
            )
        if fallback is not None and primary == fallback:
            raise ModelConfigurationError(
                f"Primary and fallback model providers cannot be identical ('{primary}'). "
                f"Use fallback=None for a single-provider configuration.",
                provider=str(primary),
            )

        self._providers = providers
        self.primary = primary
        self.fallback = fallback

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelRouter:
        """Factory method to construct ModelRouter based on application Settings."""
        providers: dict[ModelProvider, TextModel] = {}

        # Initialize Gemini if configured as primary or fallback
        if settings.videogen_primary_text_provider == ModelProvider.GEMINI or (
            settings.videogen_fallback_text_provider == ModelProvider.GEMINI
        ):
            api_key = settings.require_gemini()
            providers[ModelProvider.GEMINI] = GeminiTextModel(
                api_key=api_key,
                model_name=settings.gemini_model,
            )

        # Initialize OpenRouter if configured as primary or fallback
        if settings.videogen_primary_text_provider == ModelProvider.OPENROUTER or (
            settings.videogen_fallback_text_provider == ModelProvider.OPENROUTER
        ):
            api_key, base_url, model = settings.require_openrouter()
            providers[ModelProvider.OPENROUTER] = OpenRouterTextModel(
                api_key=api_key,
                base_url=base_url,
                model_name=model,
            )

        return cls(
            providers=providers,
            primary=settings.videogen_primary_text_provider,
            fallback=settings.videogen_fallback_text_provider,
        )

    def _resolve_fallback_provider(self, target_provider: ModelProvider) -> ModelProvider | None:
        """Resolve the active fallback provider for a given target provider.

        Ensures execution routing and fingerprint identity derivation share
        an identical fallback resolution policy.
        """
        if target_provider == self.primary and self.fallback:
            return self.fallback
        if target_provider == self.fallback and self.primary:
            return self.primary
        return None

    def _model_identity_for_provider(self, provider: ModelProvider) -> str:
        """Derive '{provider}:{model_name}' canonical identity for a configured provider."""
        model = self._providers.get(provider)
        m_id = getattr(model, "model_identity", None)
        if isinstance(m_id, str) and m_id.strip():
            raw_id = m_id.strip()
            if raw_id.startswith(f"{provider.value}:"):
                return raw_id
            return f"{provider.value}:{raw_id}"

        m_name = getattr(model, "model_name", str(provider))
        raw_name = str(m_name).strip()
        if raw_name.startswith(f"{provider.value}:"):
            return raw_name
        return f"{provider.value}:{raw_name}"

    @property
    def model_identity(self) -> str:
        """Canonical composite model configuration identifier for fingerprinting and logging."""
        primary_id = self._model_identity_for_provider(self.primary)
        if self.fallback:
            fb_id = self._model_identity_for_provider(self.fallback)
            return f"{primary_id}:fallback={fb_id}"
        return primary_id

    @property
    def model_name(self) -> str:
        """API-compatible model identifier of the primary provider."""
        primary_m = self._providers.get(self.primary)
        return str(getattr(primary_m, "model_name", str(self.primary)))

    def _resolve_provider_for_override(self, model_override: str) -> tuple[ModelProvider, str]:
        """Resolve the target provider and clean model name for an explicit override.

        Routing convention:
        - Explicit provider prefixes ('gemini:<model>', 'openrouter:<model>'):
          Routed directly to the specified provider. Both the prefix and model name must be non-empty.
          Unsupported or unconfigured provider prefixes raise ModelConfigurationError.
        - Slashed model identifiers ('<org>/<model>', e.g. 'google/gemini-2.5-flash'):
          Routed to OpenRouter if configured, otherwise falls back to primary.
        - Unprefixed model identifiers (e.g. 'gemini-2.5-pro'):
          Routed to the primary provider.
        """
        clean = model_override.strip()
        if not clean:
            raise ModelConfigurationError("Model override cannot be empty or whitespace-only")

        if ":" in clean:
            prefix, _, rest = clean.partition(":")
            prefix_clean = prefix.strip()
            model_clean = rest.strip()

            if not prefix_clean:
                raise ModelConfigurationError(
                    f"Empty provider prefix in model override '{model_override}'"
                )
            if not model_clean:
                raise ModelConfigurationError(
                    f"Empty model name after provider prefix '{prefix_clean}' in model override '{model_override}'"
                )

            prefix_lower = prefix_clean.lower()
            valid_providers = {p.value: p for p in ModelProvider}
            if prefix_lower not in valid_providers:
                supported = sorted(p.value for p in self._providers)
                raise ModelConfigurationError(
                    f"Unsupported provider prefix '{prefix_clean}' in model override '{model_override}'. "
                    f"Supported providers: {supported}"
                )

            target_provider = valid_providers[prefix_lower]
            if target_provider not in self._providers:
                supported = sorted(p.value for p in self._providers)
                raise ModelConfigurationError(
                    f"Provider '{prefix_clean}' requested in model override '{model_override}' is not configured. "
                    f"Configured providers: {supported}"
                )

            return target_provider, model_clean

        if "/" in clean and ModelProvider.OPENROUTER in self._providers:
            return ModelProvider.OPENROUTER, clean

        return self.primary, clean

    def effective_model_identity(self, model_name: str | None = None) -> str:
        """Resolve canonical effective configuration identity for execution and fingerprinting.

        Builds the identity from a normalized effective execution configuration that includes
        the requested model, resolved provider, and relevant fallback configuration.
        """
        if model_name is None or model_name.strip() in {"", "default"}:
            return self.model_identity

        target_provider, clean_override = self._resolve_provider_for_override(model_name)
        target_identity = f"{target_provider.value}:{clean_override}"

        fallback_provider = self._resolve_fallback_provider(target_provider)
        if fallback_provider:
            fb_id = self._model_identity_for_provider(fallback_provider)
            return f"{target_identity}:fallback={fb_id}"

        return target_identity

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> str:
        """Route prompt generation to primary provider, falling back on transient failures.

        When model_name is None, providers use their respective configured default models.
        When an explicit model_name override is supplied, it is routed to the appropriate
        provider without leaking composite cache identities into provider API calls.
        If a transient failure occurs, the fallback provider engages using its own
        known-compatible default model identifier.
        """
        if model_name is not None and model_name.strip() not in {"", "default"}:
            target_provider, clean_override = self._resolve_provider_for_override(model_name)
            target_kwargs: dict[str, str | None] = {
                "system_prompt": system_prompt,
                "model_name": clean_override,
            }
            fallback_provider = self._resolve_fallback_provider(target_provider)
        else:
            target_provider = self.primary
            target_kwargs = {"system_prompt": system_prompt}
            fallback_provider = self.fallback

        primary_model = self._providers[target_provider]

        try:
            return await primary_model.generate(
                prompt,
                **target_kwargs,
            )
        except ModelError as e:
            if not e.retryable or not fallback_provider:
                logger.warning(
                    "router.generate.permanent_failure",
                    primary=str(target_provider),
                    error_type=type(e).__name__,
                    retryable=e.retryable,
                    fallback_available=bool(fallback_provider),
                    action="fail_without_fallback",
                )
                raise

            logger.warning(
                "router.generate.fallback_triggered",
                primary=str(target_provider),
                fallback=str(fallback_provider),
                error_type=type(e).__name__,
                action="falling_back",
            )

            fallback_model = self._providers[fallback_provider]
            # Fallback uses its configured default model to ensure valid provider identifier
            fallback_kwargs: dict[str, str | None] = {"system_prompt": system_prompt}
            try:
                return await fallback_model.generate(
                    prompt,
                    **fallback_kwargs,
                )
            except Exception as fb_err:
                logger.error(
                    "router.generate.fallback_failed",
                    fallback=str(fallback_provider),
                    error_type=type(fb_err).__name__,
                )
                raise
