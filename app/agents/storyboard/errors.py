class StoryboardAgentError(Exception):
    """Base exception for all storyboard agent errors."""

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class StoryboardResponseError(StoryboardAgentError):
    """Raised when the LLM returns invalid, empty, or unparseable JSON."""


class StoryboardValidationError(StoryboardAgentError):
    """Raised when parsed LLM output fails schema, lineage, or duration tolerance validation constraints."""
