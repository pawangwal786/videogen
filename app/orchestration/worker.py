import asyncio
import hashlib
import json
import uuid
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.settings import Settings, get_settings
from app.db.models.job import JobAttemptModel, JobModel
from app.db.session import get_session_factory
from app.logging import get_logger
from app.orchestration.errors import LeaseConflictError
from app.orchestration.orchestrator import WorkflowOrchestrator
from app.orchestration.retry import compute_next_available_at
from app.repositories.job import JobRepository
from app.stages.artifact_store import ArtifactStore
from app.stages.base import StageHandler
from app.stages.context import EmptyPayload, StageContext
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.runtime import StageExecutionResult, StageRuntime, create_stage_context

logger = get_logger(__name__)


class JobHandler(Protocol):
    """Protocol for job execution handlers."""

    async def execute(
        self,
        job: JobModel,
        attempt: JobAttemptModel,
        worker: "JobWorker",
    ) -> dict[str, Any]:
        """Execute the job payload and return output dictionary.

        Implementations can call worker.record_submission_pending() and
        worker.record_submitted() for two-phase provider operations.
        """
        ...


class _StageHandlerAdapter[TInput: BaseModel, TOutput: BaseModel]:
    """Bridges typed StageHandler execution into JobWorker using StageRuntime."""

    def __init__(
        self,
        handler: StageHandler[TInput, TOutput],
        stage: PipelineStage,
        artifact_kind: ArtifactKind,
        output_type: type[TOutput],
        artifact_store: ArtifactStore,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        fingerprint_calculator: Callable[[JobModel, StageContext], str] | None = None,
    ) -> None:
        self.handler = handler
        self.stage = stage
        self.artifact_kind = artifact_kind
        self.output_type = output_type
        self.artifact_store = artifact_store
        self.settings = settings
        self.session_factory = session_factory
        self.fingerprint_calculator = fingerprint_calculator

    async def execute(
        self,
        job: JobModel,
        attempt: JobAttemptModel,
        worker: "JobWorker",
    ) -> dict[str, Any]:
        payload_type = (
            EmptyPayload
            if getattr(self.handler, "payload_type", EmptyPayload) is BaseModel
            else getattr(self.handler, "payload_type", EmptyPayload)
        )
        payload_model = payload_type.model_validate(job.input_payload or {})
        context = create_stage_context(
            workflow_id=job.workflow_id,
            job_id=job.id,
            attempt_id=attempt.id,
            worker_id=worker.worker_id,
            lease_token=attempt.lease_token or "",
            attempt_number=attempt.attempt_number,
            settings=self.settings,
            payload=payload_model,
        )

        if self.fingerprint_calculator is not None:
            fingerprint = self.fingerprint_calculator(job, context)
        elif hasattr(self.handler, "calculate_fingerprint"):
            fingerprint = self.handler.calculate_fingerprint(job, context)
        elif "fingerprint" in job.input_payload:
            fingerprint = str(job.input_payload["fingerprint"])
        else:
            raw = f"{self.stage.value}:{job.logical_key}:{json.dumps(job.input_payload, sort_keys=True)}"
            fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        async with self.session_factory() as session:
            runtime = StageRuntime(
                session=session,
                store=self.artifact_store,
                settings=self.settings,
            )

            result: StageExecutionResult[TOutput] = await runtime.execute_stage(
                handler=self.handler,
                context=context,
                stage=self.stage,
                artifact_kind=self.artifact_kind,
                fingerprint=fingerprint,
                output_type=self.output_type,
            )
            await session.commit()

        payload = result.output.model_dump(mode="json")
        payload["_artifact_id"] = str(result.artifact_id)
        payload["_storage_path"] = result.storage_path
        payload["_checksum"] = result.checksum
        payload["_cached"] = result.cached
        return payload


class JobWorker:
    """Processes background jobs from PostgreSQL using atomic claims and lease heartbeats.

    Implements precise shutdown semantics:
    - On stop/SIGTERM: ceases claiming new work.
    - In-flight provider operations remain durable in PostgreSQL for other workers to resume.
    - Avoids waiting indefinitely for long-running provider polling.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        worker_id: str | None = None,
        heartbeat_interval_seconds: float = 15.0,
        handlers: dict[str, JobHandler] | None = None,
        orchestrator: WorkflowOrchestrator | None = None,
        artifact_store: ArtifactStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._session_factory = session_factory or get_session_factory()
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._heartbeat_interval = heartbeat_interval_seconds
        self._handlers = handlers or {}
        self._orchestrator = orchestrator or WorkflowOrchestrator(self._session_factory)
        self._settings = settings or get_settings()
        self._artifact_store = artifact_store or ArtifactStore(
            self._settings.get_resolved_artifact_root(),
            fsync=self._settings.videogen_artifact_fsync,
        )

        self._running = False
        self._shutdown_requested = False
        self._active_job_id: str | None = None
        self._active_lease_token: str | None = None
        self._active_attempt_number: int | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._main_task: asyncio.Task[None] | None = None

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register a handler for a specific job_type."""
        self._handlers[job_type] = handler

    def register_stage_handler[TInput: BaseModel, TOutput: BaseModel](
        self,
        job_type: str,
        handler: StageHandler[TInput, TOutput],
        stage: PipelineStage,
        artifact_kind: ArtifactKind,
        output_type: type[TOutput],
        fingerprint_calculator: Callable[[JobModel, StageContext], str] | None = None,
    ) -> None:
        """Register a strongly typed StageHandler with StageRuntime bridge."""
        adapter = _StageHandlerAdapter(
            handler=handler,
            stage=stage,
            artifact_kind=artifact_kind,
            output_type=output_type,
            artifact_store=self._artifact_store,
            settings=self._settings,
            session_factory=self._session_factory,
            fingerprint_calculator=fingerprint_calculator,
        )
        self.register_handler(job_type, adapter)

    async def record_submission_pending(
        self,
        provider: str,
        request_payload: dict[str, Any] | None = None,
    ) -> None:
        """Pre-submission boundary: marks current active attempt SUBMISSION_PENDING."""
        if not self._active_job_id or not self._active_lease_token:
            return
        async with self._session_factory() as session:
            repo = JobRepository(session)
            await repo.record_submission_pending(
                job_id=self._active_job_id,
                worker_id=self.worker_id,
                lease_token=self._active_lease_token,
                provider=provider,
                request_payload=request_payload,
            )
            await session.commit()

    async def record_submitted(
        self,
        provider_operation_id: str,
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Post-submission boundary: persists operation_id and marks attempt RUNNING."""
        if not self._active_job_id or not self._active_lease_token:
            return
        async with self._session_factory() as session:
            repo = JobRepository(session)
            await repo.record_submitted(
                job_id=self._active_job_id,
                worker_id=self.worker_id,
                lease_token=self._active_lease_token,
                provider_operation_id=provider_operation_id,
                response_metadata=response_metadata,
            )
            await session.commit()

    async def get_recoverable_provider_operation_id(
        self,
        job_id: str,
        attempt_number: int | None = None,
    ) -> str | None:
        """Query recoverable provider_operation_id from candidate attempt N-1.

        Requires an explicit attempt_number or an active worker attempt.
        """
        target_attempt = (
            attempt_number if attempt_number is not None else self._active_attempt_number
        )
        if target_attempt is None:
            raise RuntimeError(
                "Cannot query recoverable operation ID without an active attempt or explicit attempt_number"
            )
        async with self._session_factory() as session:
            repo = JobRepository(session)
            return await repo.get_recoverable_provider_operation_id(job_id, target_attempt)

    async def _heartbeat_loop(self, job_id: str, lease_token: str) -> None:
        """Background task periodically updating the worker's lease heartbeat."""
        while not self._shutdown_requested and self._active_job_id == job_id:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                async with self._session_factory() as session:
                    repo = JobRepository(session)
                    await repo.heartbeat_attempt(
                        job_id=job_id,
                        worker_id=self.worker_id,
                        lease_token=lease_token,
                    )
                    await session.commit()
            except asyncio.CancelledError:
                break
            except LeaseConflictError:
                logger.warning(
                    "lease_conflict_during_heartbeat",
                    job_id=job_id,
                    worker_id=self.worker_id,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "heartbeat_error",
                    job_id=job_id,
                    worker_id=self.worker_id,
                    error=str(exc),
                )

    async def run_once(self) -> bool:
        """Attempt to claim and process one job from the queue.

        Returns True if a job was processed, False if queue was empty.
        """
        if self._shutdown_requested:
            return False

        # 1. Claim next available job
        claim: tuple[JobModel, JobAttemptModel] | None = None
        async with self._session_factory() as session:
            repo = JobRepository(session)
            claim = await repo.claim_next_job(worker_id=self.worker_id)
            if claim is not None:
                await session.commit()

        if claim is None:
            return False

        job, attempt = claim
        self._active_job_id = job.id
        self._active_lease_token = attempt.lease_token
        self._active_attempt_number = attempt.attempt_number

        # 2. Start lease heartbeat loop
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(job.id, attempt.lease_token)
        )

        handler = self._handlers.get(job.job_type)
        if handler is None:
            # Missing handler is a fatal job configuration error
            async with self._session_factory() as session:
                repo = JobRepository(session)
                await repo.fail_job(
                    job_id=job.id,
                    worker_id=self.worker_id,
                    lease_token=attempt.lease_token,
                    error_code="NO_HANDLER_REGISTERED",
                    error_message=f"No handler registered for job_type '{job.job_type}'",
                    retryable=False,
                )
                await session.commit()
            await self._cleanup_active_job()
            return True

        try:
            # 3. Execute handler
            output_payload = await handler.execute(job, attempt, self)

            # 4. Complete job successfully
            async with self._session_factory() as session:
                repo = JobRepository(session)
                await repo.complete_job(
                    job_id=job.id,
                    worker_id=self.worker_id,
                    lease_token=attempt.lease_token,
                    output_payload=output_payload,
                )
                await session.commit()

            # 5. Advance workflow state machine
            await self._orchestrator.advance_workflow(job.workflow_id)

        except asyncio.CancelledError:
            # Worker shutdown requested: abort-and-recover model. Guarantees no new work
            # is claimed and any already-persisted provider state remains recoverable in PostgreSQL.
            logger.info("job_execution_interrupted_shutdown", job_id=job.id)
            await self._cleanup_active_job()
            raise

        except Exception as exc:  # noqa: BLE001
            # Handle execution failure (determine retryability)
            retryable = getattr(exc, "retryable", True)
            error_code = getattr(exc, "code", "EXECUTION_ERROR")
            next_avail = compute_next_available_at(attempt.attempt_number) if retryable else None

            async with self._session_factory() as session:
                repo = JobRepository(session)
                await repo.fail_job(
                    job_id=job.id,
                    worker_id=self.worker_id,
                    lease_token=attempt.lease_token,
                    error_code=error_code,
                    error_message=str(exc),
                    retryable=retryable,
                    next_available_at=next_avail,
                )
                await session.commit()

            # Advance workflow to register failure if terminal
            await self._orchestrator.advance_workflow(job.workflow_id)

        finally:
            await self._cleanup_active_job()

        return True

    async def _cleanup_active_job(self) -> None:
        """Cancel and await heartbeat task, then reset active job tracking."""
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._active_job_id = None
        self._active_lease_token = None
        self._active_attempt_number = None

    async def start(self, poll_interval_seconds: float = 1.0) -> None:
        """Start long-running worker processing loop."""
        if self._running:
            return
        self._running = True
        self._shutdown_requested = False

        async def _loop():
            while self._running and not self._shutdown_requested:
                try:
                    worked = await self.run_once()
                    if not worked:
                        await asyncio.sleep(poll_interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.error("worker_loop_error", worker_id=self.worker_id, error=str(exc))
                    await asyncio.sleep(poll_interval_seconds)

        self._main_task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        """Initiate worker shutdown under the abort-and-recover model.

        Guarantees that no new work is claimed and that any already-persisted
        provider state remains recoverable in PostgreSQL.
        """
        self._shutdown_requested = True
        self._running = False
        if self._main_task is not None:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None
        await self._cleanup_active_job()


def create_production_worker(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    settings: Settings | None = None,
    worker_id: str | None = None,
    model: Any | None = None,
) -> JobWorker:
    """Production composition root constructing a fully wired JobWorker.

    Registers:
    - ResearchStageHandler with ResearchAgent backed by configured TextModel
    - ScriptStageHandler with ScriptAgent backed by configured TextModel
    - Storage and session factory dependencies
    """
    from app.agents.research.agent import ResearchAgent
    from app.agents.research.models import ResearchResult
    from app.agents.script.agent import ScriptAgent
    from app.agents.script.models import ScriptResult
    from app.models.base import TextModel
    from app.models.router import ModelRouter
    from app.stages.research import ResearchStageHandler
    from app.stages.script import ScriptStageHandler

    effective_settings = settings or get_settings()
    effective_session_factory = session_factory or get_session_factory()
    text_model: TextModel = model or ModelRouter.from_settings(effective_settings)

    worker = JobWorker(
        session_factory=effective_session_factory,
        worker_id=worker_id,
        settings=effective_settings,
    )

    research_agent = ResearchAgent(model=text_model)
    research_handler = ResearchStageHandler(agent=research_agent)
    worker.register_stage_handler(
        job_type="research",
        handler=research_handler,
        stage=PipelineStage.RESEARCH,
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
        output_type=ResearchResult,
    )

    script_agent = ScriptAgent(model=text_model)
    script_handler = ScriptStageHandler(
        session_factory=effective_session_factory,
        store=worker._artifact_store,
        agent=script_agent,
    )
    worker.register_stage_handler(
        job_type="script",
        handler=script_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptResult,
    )

    return worker
