class ScriptAgentError(Exception):
    """Base exception for all script agent errors."""

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class ScriptResponseError(ScriptAgentError):
    """Raised when the LLM returns invalid, empty, or unparseable JSON."""


class ScriptValidationError(ScriptAgentError):
    """Raised when parsed LLM output fails schema or lineage validation constraints."""
