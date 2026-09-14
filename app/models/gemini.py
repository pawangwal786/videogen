import asyncio
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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


class GeminiTextModel:
    """Production Gemini implementation of the TextModel protocol."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        *,
        timeout_seconds: float = 60.0,
        client: genai.Client | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ModelConfigurationError(
                "Gemini API key must not be empty.",
                provider="gemini",
            )
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._client = client or genai.Client(api_key=api_key.strip())

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from Gemini with structured logging, timeout, and normalized errors."""
        start_time = time.perf_counter()
        config = (
            types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
            if system_prompt
            else None
        )

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                ),
                timeout=self.timeout_seconds,
            )

            text = response.text
            if not text or not text.strip():
                raise ModelResponseError(
                    "Gemini returned an empty response.",
                    provider="gemini",
                    retryable=False,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "model.generate",
                provider="gemini",
                model=self.model_name,
                prompt_length=len(prompt),
                duration_ms=round(duration_ms, 2),
                success=True,
            )
            return text.strip()

        except TimeoutError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "model.generate",
                provider="gemini",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type="timeout",
            )
            raise ModelTimeoutError(
                f"Gemini generation timed out after {self.timeout_seconds}s.",
                provider="gemini",
                cause=e,
            ) from e

        except genai_errors.APIError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            code = getattr(e, "code", None)
            message = str(e)

            if (
                code in (401, 403)
                or "unauthorized" in message.lower()
                or "api key" in message.lower()
            ):
                err = ModelAuthenticationError(
                    f"Gemini authentication failed: {message}",
                    provider="gemini",
                    cause=e,
                )
            elif (
                code == 429 or "resource_exhausted" in message.lower() or "quota" in message.lower()
            ):
                err = ModelRateLimitError(
                    f"Gemini rate limit exceeded: {message}",
                    provider="gemini",
                    cause=e,
                )
            elif code == 400:
                err = ModelConfigurationError(
                    f"Gemini invalid request: {message}",
                    provider="gemini",
                    cause=e,
                )
            elif code and code >= 500:
                err = ModelResponseError(
                    f"Gemini server error ({code}): {message}",
                    provider="gemini",
                    cause=e,
                    retryable=True,
                )
            else:
                err = ModelResponseError(
                    f"Gemini API error: {message}",
                    provider="gemini",
                    cause=e,
                    retryable=False,
                )

            logger.error(
                "model.generate",
                provider="gemini",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=type(err).__name__,
            )
            raise err from e

        except (ModelError, ModelResponseError):
            # Already normalized
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "model.generate",
                provider="gemini",
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
                provider="gemini",
                model=self.model_name,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=type(e).__name__,
            )
            raise ModelResponseError(
                f"Unexpected error communicating with Gemini: {e}",
                provider="gemini",
                cause=e,
                retryable=False,
            ) from e
