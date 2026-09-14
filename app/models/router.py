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

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Route prompt generation to primary provider, falling back on transient failures."""
        primary_model = self._providers[self.primary]

        try:
            return await primary_model.generate(prompt, system_prompt=system_prompt)
        except ModelError as e:
            # Check if failure is transient/retryable and fallback is available
            if not e.retryable or not self.fallback:
                logger.warning(
                    "router.generate.permanent_failure",
                    primary=str(self.primary),
                    error_type=type(e).__name__,
                    retryable=e.retryable,
                    fallback_available=bool(self.fallback),
                    action="fail_without_fallback",
                )
                raise

            logger.warning(
                "router.generate.fallback_triggered",
                primary=str(self.primary),
                fallback=str(self.fallback),
                error_type=type(e).__name__,
                action="falling_back",
            )

            fallback_model = self._providers[self.fallback]
            try:
                return await fallback_model.generate(prompt, system_prompt=system_prompt)
            except Exception as fb_err:
                logger.error(
                    "router.generate.fallback_failed",
                    fallback=str(self.fallback),
                    error_type=type(fb_err).__name__,
                )
                raise
