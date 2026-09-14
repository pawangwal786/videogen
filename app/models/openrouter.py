import asyncio
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from app.logging import get_logger
from app.models.errors import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelError,
    ModelRateLimitError,
    ModelResponseError,
    ModelTimeoutError,
)

logger = get_logger(__name__)


class OpenRouterTextModel:
    """Production OpenRouter implementation of the TextModel protocol using the OpenAI client."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "google/gemini-2.5-flash",
        *,
        timeout_seconds: float = 60.0,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ModelConfigurationError(
                "OpenRouter API key must not be empty.",
                provider="openrouter",
            )
        self.base_url = base_url.strip()
        self.model_name = model_name.strip()
        self.timeout_seconds = timeout_seconds
        self._client = client or AsyncOpenAI(
            api_key=api_key.strip(),
            base_url=self.base_url,
            timeout=timeout_seconds,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from OpenRouter with structured logging, timeout, and normalized errors."""
        start_time = time.perf_counter()
        messages: list[dict[str, Any]] = []

        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,  # type: ignore[arg-type]
                ),
                timeout=self.timeout_seconds,
            )

            if not response.choices or not response.choices[0].message:
                raise ModelResponseError(
                    "OpenRouter returned a response with no choices.",
                    provider="openrouter",
                    retryable=False,
                )

            text = response.choices[0].message.content
            if not text or not text.strip():
                raise ModelResponseError(
                    "OpenRouter returned an empty response.",
                    provider="openrouter",
                    retryable=False,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                prompt_length=len(prompt),
                duration_ms=round(duration_ms, 2),
                success=True,
            )
            return text.strip()

        except (TimeoutError, APITimeoutError) as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="timeout",
            )
            raise ModelTimeoutError(
                f"OpenRouter generation timed out after {self.timeout_seconds}s.",
                provider="openrouter",
                cause=e,
            ) from e

        except (AuthenticationError, PermissionDeniedError) as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="authentication",
            )
            raise ModelAuthenticationError(
                f"OpenRouter authentication failed: {e}",
                provider="openrouter",
                cause=e,
            ) from e

        except RateLimitError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="rate_limit",
            )
            raise ModelRateLimitError(
                f"OpenRouter rate limit exceeded: {e}",
                provider="openrouter",
                cause=e,
            ) from e

        except BadRequestError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="bad_request",
            )
            raise ModelConfigurationError(
                f"OpenRouter bad request: {e}",
                provider="openrouter",
                cause=e,
            ) from e

        except (InternalServerError, APIConnectionError) as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="server_error",
            )
            raise ModelResponseError(
                f"OpenRouter server or connection error: {e}",
                provider="openrouter",
                cause=e,
                retryable=True,
            ) from e

        except APIError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="api_error",
            )
            raise ModelResponseError(
                f"OpenRouter API error: {e}",
                provider="openrouter",
                cause=e,
                retryable=False,
            ) from e

        except (ModelError, ModelResponseError):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="ModelResponseError",
            )
            raise

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="openrouter",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=type(e).__name__,
            )
            raise ModelResponseError(
                f"Unexpected error communicating with OpenRouter: {e}",
                provider="openrouter",
                cause=e,
                retryable=False,
            ) from e
