"""Base abstract handler for VideoGen pipeline stages."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.stages.context import EmptyPayload, StageContext


class StageHandler[TInput: BaseModel, TOutput: BaseModel](ABC):
    """Abstract base contract for all VideoGen pipeline stage executors.

    Stage handlers implement pure input resolution and stage execution logic.
    Persistence of resulting artifacts, attempt state tracking, lease heartbeating,
    and database transactions are strictly owned by the worker and orchestrator
    runtime infrastructure.

    Subclasses must:
    1. Declare `stage_name` identifying the logical stage.
    2. Implement `load_input()` to deserialize typed inputs strictly from durable artifacts.
    3. Implement `execute()` to produce a strongly-typed `TOutput` without committing state.
    """

    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Canonical stage name handled by this executor."""
        ...

    @property
    def payload_type(self) -> type[BaseModel]:
        """Typed model for validating the triggering job's input_payload. Defaults to EmptyPayload."""
        return EmptyPayload

    @abstractmethod
    async def load_input(self, context: StageContext) -> TInput:
        """Resolve, validate, and deserialize input artifacts from durable storage.

        Must NOT access in-memory Python state or prior stage instances.
        """
        ...

    @abstractmethod
    async def execute(self, input_data: TInput, context: StageContext) -> TOutput:
        """Execute stage computation and return strongly-typed output DTO.

        Must NOT commit database transactions or persist output artifacts directly.
        """
        ...
