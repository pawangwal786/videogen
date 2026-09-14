from abc import ABC, abstractmethod


class Agent[InputT, OutputT](ABC):
    """Base contract for all VideoGen agents."""

    name: str

    @abstractmethod
    async def run(self, input: InputT) -> OutputT:
        """Execute the agent."""
        raise NotImplementedError
