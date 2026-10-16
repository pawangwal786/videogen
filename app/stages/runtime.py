"""Phase 9 StageRuntime orchestrating typed stage execution and persistence."""

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.stages.artifact_errors import FingerprintConflictError, SecurityViolationError
from app.stages.artifact_store import ArtifactStore
from app.stages.base import StageHandler
from app.stages.context import EmptyPayload, StageContext
from app.stages.dto import ArtifactKind, PipelineStage


class StageExecutionResult[TOutput: BaseModel](BaseModel):
    """Strongly typed outcome of a stage execution through StageRuntime."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    output: TOutput
    artifact_id: UUID
    storage_path: str
    checksum: str
    cached: bool = False


class StageRuntime:
    """Infrastructure-owned runtime managing typed stage execution.

    Responsibilities:
    1. Idempotency verification: queries StageArtifactRepository by deterministic fingerprint.
       If an AVAILABLE artifact exists, deserializes directly from ArtifactStore without recomputation.
    2. Input resolution and pure execution: passes StageContext to handler.
    3. Three-phase persistence protocol:
       a. Database reservation via StageArtifactRepository.create_pending() (prevents untracked orphans)
       b. Physical file write via ArtifactStore.write_artifact()
       c. Lease-fenced finalization via StageArtifactRepository.finalize() (verifies lease ownership)
    """

    def __init__(
        self,
        session: AsyncSession,
        store: ArtifactStore,
        settings: Settings,
    ) -> None:
        from app.stages.stage_artifact_repository import StageArtifactRepository

        self._session = session
        self._store = store
        self._settings = settings
        self._repo = StageArtifactRepository(session, store)

    def _validate_output_identity[TOutput: BaseModel](
        self,
        output_dto: TOutput,
        context: StageContext,
        origin: str = "fresh",
    ) -> None:
        """Enforce workflow identity invariant on stage output across all execution/cache paths."""
        output_wf_id = getattr(output_dto, "workflow_id", None)
        if output_wf_id is not None and str(output_wf_id) != str(context.workflow_id):
            raise SecurityViolationError(
                f"Stage output workflow_id '{output_wf_id}' ({origin}) does not match "
                f"active context workflow_id '{context.workflow_id}'"
            )

    async def execute_stage[TInput: BaseModel, TOutput: BaseModel](
        self,
        handler: StageHandler[TInput, TOutput],
        context: StageContext,
        stage: PipelineStage,
        artifact_kind: ArtifactKind,
        fingerprint: str,
        output_type: type[TOutput],
        input_data: TInput | None = None,
        extension: str = "json",
        mime_type: str = "application/json",
    ) -> StageExecutionResult[TOutput]:
        """Execute a typed stage handler with fingerprint idempotency and lease-fenced persistence."""
        # 1. Idempotency Check (Job Retry != Stage Recomputation)
        cached_artifact = await self._repo.find_by_fingerprint(
            workflow_id=context.workflow_id,
            stage=stage,
            stage_fingerprint=fingerprint,
            artifact_kind=artifact_kind,
        )
        if cached_artifact is not None:
            verified_artifact, raw_bytes = await self._repo.read_and_verify(cached_artifact)
            cached_output = output_type.model_validate_json(raw_bytes)
            self._validate_output_identity(cached_output, context, origin="cached")
            return StageExecutionResult[TOutput](
                output=cached_output,
                artifact_id=UUID(verified_artifact.id),
                storage_path=verified_artifact.storage_path,
                checksum=verified_artifact.checksum or "",
                cached=True,
            )

        # 2. Pure computation
        resolved_input = input_data if input_data is not None else await handler.load_input(context)
        output_dto = await handler.execute(resolved_input, context)
        self._validate_output_identity(output_dto, context, origin="fresh")

        serialized_bytes = output_dto.model_dump_json(indent=2).encode("utf-8")

        # 3. Step A of Persistence: Database reservation (create_pending)
        try:
            pending_record = await self._repo.create_pending(
                workflow_id=context.workflow_id,
                job_id=UUID(context.job_id),
                stage=stage,
                artifact_kind=artifact_kind,
                stage_fingerprint=fingerprint,
                extension=extension,
                mime_type=mime_type,
            )
        except FingerprintConflictError:
            # Concurrent race: another worker may have just finalized this fingerprint
            raced_artifact = await self._repo.find_by_fingerprint(
                workflow_id=context.workflow_id,
                stage=stage,
                stage_fingerprint=fingerprint,
                artifact_kind=artifact_kind,
            )
            if raced_artifact is not None:
                verified_artifact, raw_bytes = await self._repo.read_and_verify(raced_artifact)
                cached_output = output_type.model_validate_json(raw_bytes)
                self._validate_output_identity(cached_output, context, origin="race_recovered")
                return StageExecutionResult[TOutput](
                    output=cached_output,
                    artifact_id=UUID(verified_artifact.id),
                    storage_path=verified_artifact.storage_path,
                    checksum=verified_artifact.checksum or "",
                    cached=True,
                )
            raise

        # 4. Step B of Persistence: Write physical file
        await self._store.write_artifact(Path(pending_record.storage_path), serialized_bytes)

        # 5. Step C of Persistence: Lease-fenced finalization
        finalized_record = await self._repo.finalize(
            UUID(pending_record.id),
            worker_id=context.worker_id,
            lease_token=context.lease_token,
        )

        return StageExecutionResult[TOutput](
            output=output_dto,
            artifact_id=UUID(finalized_record.id),
            storage_path=finalized_record.storage_path,
            checksum=finalized_record.checksum or "",
            cached=False,
        )


def create_stage_context[TPayload: BaseModel](
    workflow_id: str | UUID,
    job_id: str,
    attempt_id: str,
    worker_id: str,
    lease_token: str,
    attempt_number: int,
    settings: Settings,
    payload: TPayload | None = None,
) -> StageContext[TPayload]:
    """Factory creating an immutable StageContext."""
    return StageContext[TPayload](
        workflow_id=UUID(str(workflow_id)),
        job_id=job_id,
        attempt_id=attempt_id,
        worker_id=worker_id,
        lease_token=lease_token,
        attempt_number=attempt_number,
        settings=settings,
        payload=payload if payload is not None else EmptyPayload(),
    )
