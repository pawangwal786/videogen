from typing import Protocol


class TextModel(Protocol):
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> str:
        """Generate text from a model."""
        ...
