"""Research stage handler for the autonomous video generation pipeline."""

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agents.base import Agent
from app.agents.research.models import ResearchRequest, ResearchResult
from app.stages.artifact_errors import SecurityViolationError
from app.stages.base import StageHandler
from app.stages.context import StageContext
from app.stages.dto import PipelineStage
from app.stages.fingerprint import StageFingerprintCalculator

if TYPE_CHECKING:
    from app.db.models.job import JobModel


class ResearchJobPayload(BaseModel):
    """Validated input payload for a research stage job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1, description="Video topic to research")
    target_duration_seconds: int = Field(
        default=60, ge=15, le=600, description="Target runtime in seconds"
    )
    target_audience: str = Field(
        default="General audience", max_length=200, description="Target audience description"
    )
    research_depth: Literal["quick", "standard", "deep"] = Field(
        default="standard", description="Granularity of research analysis"
    )
    model_name: str = Field(
        default="default", description="Model configuration identifier for fingerprinting"
    )

    @field_validator("topic", "target_audience", "model_name")
    @classmethod
    def validate_non_blank_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be blank or whitespace-only")
        return v.strip()


class ResearchStageHandler(StageHandler[ResearchRequest, ResearchResult]):
    """Stage handler executing the RESEARCH pipeline stage.

    Converts incoming research job requests into typed ResearchRequest DTOs,
    invokes the research agent, and produces a validated ResearchResult without
    directly committing database state or managing file persistence.
    """

    def __init__(self, agent: Agent[ResearchRequest, ResearchResult] | None = None) -> None:
        self._agent = agent

    def _canonical_model_identity(self, payload: ResearchJobPayload) -> str:
        """Resolve canonical model configuration identity for fingerprinting and logging.

        Builds the fingerprint from a normalized effective execution configuration that includes
        the requested model, resolved provider, and relevant fallback configuration.
        """
        has_override = bool(
            payload.model_name and payload.model_name.strip() not in {"", "default"}
        )
        override_val = payload.model_name.strip() if has_override else None

        if self._agent is not None:
            agent_model = getattr(self._agent, "_model", None)
            if agent_model is not None:
                if hasattr(type(agent_model), "effective_model_identity"):
                    eff_fn = getattr(agent_model, "effective_model_identity", None)
                    if callable(eff_fn):
                        res = eff_fn(override_val)
                        if isinstance(res, str):
                            return res.strip()

                agent_id = getattr(agent_model, "model_identity", None) or getattr(
                    agent_model, "model_name", None
                )
                if agent_id and isinstance(agent_id, str):
                    if override_val:
                        if ":fallback=" in agent_id:
                            fb_part = agent_id.split(":fallback=", 1)[1]
                            return f"{override_val}:fallback={fb_part}"
                        return override_val
                    return agent_id.strip()

        return override_val or "default"

    @property
    def stage_name(self) -> str:
        return PipelineStage.RESEARCH.value

    @property
    def payload_type(self) -> type[ResearchJobPayload]:
        return ResearchJobPayload

    def calculate_fingerprint(
        self,
        job: "JobModel | None",
        context: StageContext[ResearchJobPayload],
    ) -> str:
        """Derive canonical deterministic fingerprint for the RESEARCH stage."""
        payload = context.payload
        model_identity = self._canonical_model_identity(payload)
        return StageFingerprintCalculator.for_research(
            topic=payload.topic,
            target_duration_seconds=payload.target_duration_seconds,
            target_audience=payload.target_audience,
            research_depth=payload.research_depth,
            model_name=model_identity,
        )

    async def load_input(self, context: StageContext[ResearchJobPayload]) -> ResearchRequest:
        """Construct typed ResearchRequest from the validated StageContext payload.

        Only passes an explicit override when requested in the job payload.
        Never injects composite cache or router identities into the request.
        Validates model override prefix against configured router if available.
        """
        payload = context.payload
        override = (
            payload.model_name.strip()
            if payload.model_name and payload.model_name.strip() not in {"", "default"}
            else None
        )
        if override and self._agent is not None:
            agent_model = getattr(self._agent, "_model", None)
            if agent_model is not None and hasattr(
                type(agent_model), "_resolve_provider_for_override"
            ):
                resolve_fn = getattr(agent_model, "_resolve_provider_for_override", None)
                if callable(resolve_fn):
                    resolve_fn(override)

        return ResearchRequest(
            workflow_id=context.workflow_id,
            topic=payload.topic,
            target_audience=payload.target_audience,
            target_duration_seconds=payload.target_duration_seconds,
            research_depth=payload.research_depth,
            model_name=override,
        )

    async def execute(
        self,
        input_data: ResearchRequest,
        context: StageContext[ResearchJobPayload],
    ) -> ResearchResult:
        """Execute research generation via the injected agent."""
        if self._agent is None:
            raise RuntimeError(
                "ResearchStageHandler requires a configured ResearchAgent to execute"
            )
        result = await self._agent.run(input_data)
        if result.workflow_id != context.workflow_id:
            raise SecurityViolationError(
                f"ResearchAgent produced result with workflow_id '{result.workflow_id}' "
                f"which does not match active context workflow_id '{context.workflow_id}'"
            )
        return result


__all__ = [
    "ResearchJobPayload",
    "ResearchStageHandler",
]
