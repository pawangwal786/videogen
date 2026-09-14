class ModelError(Exception):
    """Base exception for all model provider errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.cause = cause
        self.retryable = retryable


class ModelConfigurationError(ModelError):
    """Configuration error (e.g. missing API key, invalid model name). Permanent/not retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, cause=cause, retryable=False)


class ModelAuthenticationError(ModelError):
    """Authentication or authorization failure. Permanent/not retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, cause=cause, retryable=False)


class ModelTimeoutError(ModelError):
    """Call to model timed out. Transient/retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, cause=cause, retryable=True)


class ModelRateLimitError(ModelError):
    """Provider rate limit or quota exceeded. Transient/retryable."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, provider=provider, cause=cause, retryable=True)


class ModelResponseError(ModelError):
    """Invalid, empty, or malformed response from model provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        cause: Exception | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message, provider=provider, cause=cause, retryable=retryable)
