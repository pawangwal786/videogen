"""Script stage handler for the autonomous video generation pipeline."""

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agents.base import Agent
from app.agents.research.models import ResearchResult
from app.agents.script.models import ScriptRequest, ScriptResult
from app.stages.artifact_errors import SecurityViolationError
from app.stages.artifact_store import ArtifactStore
from app.stages.base import StageHandler
from app.stages.context import StageContext
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.fingerprint import StageFingerprintCalculator
from app.stages.stage_artifact_repository import StageArtifactRepository

if TYPE_CHECKING:
    from app.db.models.job import JobModel


class ScriptJobPayload(BaseModel):
    """Validated input payload for a script stage job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    research_artifact_id: UUID = Field(
        description="Durable artifact ID of the upstream ResearchResult"
    )
    research_artifact_checksum: str = Field(
        min_length=64, max_length=64, description="SHA-256 checksum of upstream research artifact"
    )
    pacing_wpm: int = Field(
        default=140, ge=80, le=240, description="Narration pacing in words per minute"
    )
    target_duration_seconds: int | None = Field(
        default=None, ge=15, le=600, description="Optional target runtime assertion in seconds"
    )
    tone: str = Field(default="engaging, informative, and cinematic", max_length=200)
    call_to_action: str | None = Field(default=None, max_length=300)

    @field_validator("research_artifact_checksum")
    @classmethod
    def validate_checksum_hex(cls, v: str) -> str:
        clean = v.strip().lower()
        if len(clean) != 64 or not all(c in "0123456789abcdef" for c in clean):
            raise ValueError(
                f"research_artifact_checksum must be a 64-char hexadecimal string, got: '{v}'"
            )
        return clean

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("tone cannot be blank or whitespace-only")
        return v.strip()

    @field_validator("call_to_action")
    @classmethod
    def validate_cta(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v.strip() if v is not None else None


class ScriptStageHandler(StageHandler[ScriptRequest, ScriptResult]):
    """Stage handler executing the SCRIPT pipeline stage.

    Loads the upstream ResearchResult artifact directly from durable storage
    using cryptographic integrity verification, calculates canonical fingerprints
    incorporating the research checksum, narration pacing, tone, and call to action,
    and produces a validated ScriptResult via the injected ScriptAgent.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None,
        store: ArtifactStore | None,
        agent: Agent[ScriptRequest, ScriptResult] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._store = store
        self._agent = agent

    @property
    def stage_name(self) -> str:
        return PipelineStage.SCRIPT.value

    @property
    def payload_type(self) -> type[ScriptJobPayload]:
        return ScriptJobPayload

    def calculate_fingerprint(
        self,
        job: "JobModel | None",
        context: StageContext[ScriptJobPayload],
    ) -> str:
        """Derive canonical deterministic fingerprint for the SCRIPT stage."""
        payload = context.payload
        return StageFingerprintCalculator.for_script(
            research_artifact_checksum=payload.research_artifact_checksum,
            pacing_wpm=payload.pacing_wpm,
            tone=payload.tone,
            call_to_action=payload.call_to_action,
            target_duration_seconds=payload.target_duration_seconds,
        )

    async def load_input(self, context: StageContext[ScriptJobPayload]) -> ScriptRequest:
        """Load and verify upstream ResearchResult artifact strictly from durable storage."""
        if self._session_factory is None or self._store is None:
            raise RuntimeError(
                "ScriptStageHandler requires session_factory and ArtifactStore to load input"
            )

        payload = context.payload
        async with self._session_factory() as session:
            repo = StageArtifactRepository(session, self._store)
            artifact, raw_bytes = await repo.read_verified_artifact(payload.research_artifact_id)

        # 1. Authoritative check: verify supplied expected checksum against verified artifact record
        if artifact.checksum != payload.research_artifact_checksum:
            raise SecurityViolationError(
                f"Integrity failure: research artifact '{artifact.id}' checksum '{artifact.checksum}' "
                f"does not match expected job checksum '{payload.research_artifact_checksum}'"
            )

        # 2. Authoritative check: verify artifact stage and kind
        if artifact.stage != PipelineStage.RESEARCH.value:
            raise SecurityViolationError(
                f"Contract violation: referenced artifact '{artifact.id}' stage is '{artifact.stage}', "
                f"expected '{PipelineStage.RESEARCH.value}'"
            )
        if artifact.artifact_kind != ArtifactKind.RESEARCH_RESULT.value:
            raise SecurityViolationError(
                f"Contract violation: referenced artifact '{artifact.id}' kind is '{artifact.artifact_kind}', "
                f"expected '{ArtifactKind.RESEARCH_RESULT.value}'"
            )

        # 3. Authoritative check: verify workflow identity on artifact record
        if str(artifact.workflow_id) != str(context.workflow_id):
            raise SecurityViolationError(
                f"Contract violation: referenced artifact '{artifact.id}' belongs to workflow '{artifact.workflow_id}', "
                f"which does not match active context workflow '{context.workflow_id}'"
            )

        # 4. Deserialize and verify payload-level workflow identity
        research_result = ResearchResult.model_validate_json(raw_bytes)
        if research_result.workflow_id != context.workflow_id:
            raise SecurityViolationError(
                f"Loaded research artifact workflow_id ({research_result.workflow_id}) "
                f"does not match current context workflow_id ({context.workflow_id})"
            )

        # 5. Consistency check: target duration if explicitly supplied
        if (
            payload.target_duration_seconds is not None
            and payload.target_duration_seconds != research_result.target_duration_seconds
        ):
            raise ValueError(
                f"ScriptJobPayload target_duration_seconds ({payload.target_duration_seconds}) "
                f"does not match research artifact target_duration_seconds ({research_result.target_duration_seconds})"
            )

        return ScriptRequest(
            workflow_id=context.workflow_id,
            research=research_result,
            pacing_wpm=payload.pacing_wpm,
            tone=payload.tone,
            call_to_action=payload.call_to_action,
        )

    async def execute(
        self,
        input_data: ScriptRequest,
        context: StageContext[ScriptJobPayload],
    ) -> ScriptResult:
        """Execute script generation via the injected agent."""
        if self._agent is None:
            raise RuntimeError("ScriptStageHandler requires a configured ScriptAgent to execute")
        result = await self._agent.run(input_data)
        if result.workflow_id != context.workflow_id:
            raise SecurityViolationError(
                f"ScriptAgent produced result with workflow_id '{result.workflow_id}' "
                f"which does not match active context workflow_id '{context.workflow_id}'"
            )
        return result


__all__ = [
    "ScriptJobPayload",
    "ScriptStageHandler",
]
