class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class ResearchResponseError(ResearchAgentError):
    """Raised when the LLM returns invalid, empty, or unparseable JSON."""


class ResearchValidationError(ResearchAgentError):
    """Raised when parsed LLM output fails schema validation constraints."""
