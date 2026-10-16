"""Unit tests for StageRuntime infrastructure."""

import hashlib
import uuid
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import Settings
from app.db.base import Base
from app.db.models.artifact import ArtifactModel
from app.db.models.job import JobAttemptModel, JobModel
from app.db.models.workflow import WorkflowModel
from app.orchestration.errors import LeaseConflictError
from app.orchestration.state_machine import (
    AttemptStatus,
    JobStage,
    JobStatus,
)
from app.stages.artifact_errors import FingerprintConflictError, SecurityViolationError
from app.stages.artifact_store import ArtifactStore
from app.stages.base import StageHandler
from app.stages.context import StageContext
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.runtime import StageExecutionResult, StageRuntime, create_stage_context


class SampleInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    prompt: str


class SampleOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str
    tokens: int


class MockStageHandler(StageHandler[SampleInput, SampleOutput]):
    def __init__(self, output: SampleOutput | None = None) -> None:
        self._output = output or SampleOutput(content="Generated story script", tokens=42)
        self.execute_calls = 0
        self.load_input_calls = 0

    @property
    def stage_name(self) -> str:
        return "script"

    async def load_input(self, context: StageContext) -> SampleInput:
        self.load_input_calls += 1
        return SampleInput(prompt="Default prompt")

    async def execute(self, input_data: SampleInput, context: StageContext) -> SampleOutput:
        self.execute_calls += 1
        return self._output


@pytest.fixture
def tmp_store(tmp_path: Path) -> ArtifactStore:
    store = ArtifactStore(tmp_path / "artifacts", fsync=False)
    yield store
    store.close()


@pytest.fixture
async def sqlite_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.asyncio
async def test_stage_runtime_execution_and_persistence(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Verify standard execution: DB reservation -> file write -> lease-fenced finalize."""
    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "lease-token-abc"

    async with sqlite_session_factory() as session:
        # Setup workflow, job, and active attempt in DB
        wf = WorkflowModel(
            id=str(wf_id),
            topic="Quantum Computing",
            status="RUNNING",
            current_stage=JobStage.SCRIPT.value,
        )
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "a" * 64

        # Execute
        result = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            fingerprint=fingerprint,
            output_type=SampleOutput,
        )

        assert isinstance(result, StageExecutionResult)
        assert result.cached is False
        assert result.output.content == "Generated story script"
        assert result.output.tokens == 42
        assert handler.execute_calls == 1
        assert handler.load_input_calls == 1

        # Check physical file exists and verify content
        file_path = Path(result.storage_path)
        assert file_path.exists()
        raw_bytes = await tmp_store.read_artifact(file_path)
        deserialized = SampleOutput.model_validate_json(raw_bytes)
        assert deserialized == result.output


@pytest.mark.asyncio
async def test_stage_runtime_idempotency_cache_hit(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Verify that if an AVAILABLE artifact with matching fingerprint exists, handler.execute is skipped."""
    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "lease-token-abc"

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Test Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "b" * 64

        # 1. First execution creates and finalizes artifact
        res1 = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            fingerprint=fingerprint,
            output_type=SampleOutput,
        )
        assert res1.cached is False
        assert handler.execute_calls == 1

        # 2. Second execution with same fingerprint reuses available artifact
        res2 = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            fingerprint=fingerprint,
            output_type=SampleOutput,
        )
        assert res2.cached is True
        assert res2.artifact_id == res1.artifact_id
        assert res2.output == res1.output
        # Handler execute was NOT called again
        assert handler.execute_calls == 1


@pytest.mark.asyncio
async def test_stage_runtime_lease_fence_rejection(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Verify that a worker with an expired or mismatched lease token cannot finalize artifacts."""
    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "valid-token-123"

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Test Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        # Create context with a STALE/DIFFERENT lease token
        stale_context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token="stale-token-expired",
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "c" * 64

        with pytest.raises(LeaseConflictError):
            await runtime.execute_stage(
                handler=handler,
                context=stale_context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint=fingerprint,
                output_type=SampleOutput,
            )


@pytest.mark.asyncio
async def test_stage_runtime_fingerprint_conflict_resolves_to_available_cache(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
    monkeypatch,
):
    """Verify that if create_pending raises FingerprintConflictError and artifact is AVAILABLE, it returns cached."""
    from app.db.models.artifact import ArtifactModel
    from app.stages.artifact_errors import FingerprintConflictError

    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "lease-token-abc"

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Test Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "d" * 64

        # Write an existing physical file
        art_id = uuid.uuid4()
        path = tmp_store.build_path(wf_id, PipelineStage.SCRIPT, art_id, "json")
        written_bytes = SampleOutput(content="Raced script", tokens=10).model_dump_json().encode()
        await tmp_store.write_artifact(path, written_bytes)
        import hashlib

        real_checksum = hashlib.sha256(written_bytes).hexdigest()

        cached_mock = ArtifactModel(
            id=str(art_id),
            workflow_id=str(wf_id),
            job_id=job_id,
            artifact_type="SCRIPT_RESULT",
            artifact_kind="SCRIPT_RESULT",
            storage_path=str(path),
            checksum=real_checksum,
            file_size_bytes=len(written_bytes),
            lifecycle_status="AVAILABLE",
            stage="SCRIPT",
            stage_fingerprint=fingerprint,
        )

        # Mock find_by_fingerprint to return None on first check, cached_mock on second check
        calls = 0

        async def mock_find(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return None
            return cached_mock

        async def mock_create_pending(*args, **kwargs):
            raise FingerprintConflictError(str(wf_id), "SCRIPT", fingerprint)

        monkeypatch.setattr(runtime._repo, "find_by_fingerprint", mock_find)
        monkeypatch.setattr(runtime._repo, "create_pending", mock_create_pending)

        res = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            fingerprint=fingerprint,
            output_type=SampleOutput,
        )

        assert res.cached is True
        assert res.output.content == "Raced script"


@pytest.mark.asyncio
async def test_stage_runtime_fingerprint_conflict_reraises_if_still_pending(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
    monkeypatch,
):
    """Verify that if create_pending raises FingerprintConflictError and artifact is NOT AVAILABLE, it re-raises."""
    from app.stages.artifact_errors import FingerprintConflictError

    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "lease-token-abc"

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Test Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "e" * 64

        async def mock_find_none(*args, **kwargs):
            return None

        async def mock_create_pending(*args, **kwargs):
            raise FingerprintConflictError(str(wf_id), "SCRIPT", fingerprint)

        monkeypatch.setattr(runtime._repo, "find_by_fingerprint", mock_find_none)
        monkeypatch.setattr(runtime._repo, "create_pending", mock_create_pending)

        with pytest.raises(FingerprintConflictError):
            await runtime.execute_stage(
                handler=handler,
                context=context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint=fingerprint,
                output_type=SampleOutput,
            )


@pytest.mark.asyncio
async def test_stage_runtime_corrupted_cached_artifact_raises_security_violation(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Verify that a corrupted or tampered cached artifact raises SecurityViolationError on cache hit."""
    from app.stages.artifact_errors import SecurityViolationError

    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "lease-token-abc"

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Test Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "f" * 64

        # 1. Execute to create and finalize valid cached artifact
        res = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            fingerprint=fingerprint,
            output_type=SampleOutput,
        )
        assert res.cached is False

        # 2. Tamper with the physical file on disk
        artifact_path = Path(res.storage_path)
        assert artifact_path.exists()
        artifact_path.write_bytes(b'{"content": "TAMPERED DATA", "tokens": 999}')

        # 3. Second execution with same fingerprint must detect corruption and raise SecurityViolationError
        with pytest.raises(SecurityViolationError, match="checksum mismatch"):
            await runtime.execute_stage(
                handler=handler,
                context=context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint=fingerprint,
                output_type=SampleOutput,
            )


@pytest.mark.asyncio
async def test_stage_runtime_missing_cached_file_raises_unavailable(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Verify that if an AVAILABLE artifact record exists but the physical file is missing, ArtifactUnavailableError is raised."""
    from app.stages.artifact_errors import ArtifactUnavailableError

    settings = Settings(_env_file=None)
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    worker_id = "test-worker-1"
    lease_token = "lease-token-abc"

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Test Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id=worker_id,
            lease_token=lease_token,
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id=worker_id,
            lease_token=lease_token,
            attempt_number=1,
            settings=settings,
        )

        handler = MockStageHandler()
        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = "7" * 64

        res = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            fingerprint=fingerprint,
            output_type=SampleOutput,
        )
        assert res.cached is False

        # Delete physical file
        artifact_path = Path(res.storage_path)
        assert artifact_path.exists()
        artifact_path.unlink()

        with pytest.raises(ArtifactUnavailableError, match="physical artifact file does not exist"):
            await runtime.execute_stage(
                handler=handler,
                context=context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint=fingerprint,
                output_type=SampleOutput,
            )


class WorkflowOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    workflow_id: uuid.UUID
    content: str


class MockWorkflowStageHandler(StageHandler[SampleInput, WorkflowOutput]):
    def __init__(self, output: WorkflowOutput) -> None:
        self._output = output

    @property
    def stage_name(self) -> str:
        return "script"

    async def load_input(self, context: StageContext) -> SampleInput:
        return SampleInput(prompt="Default prompt")

    async def execute(self, input_data: SampleInput, context: StageContext) -> WorkflowOutput:
        return self._output


@pytest.mark.asyncio
async def test_stage_runtime_rejects_output_with_mismatched_workflow_id(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):

    settings = Settings(_env_file=None)
    active_wf_id = uuid.uuid4()
    different_wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(active_wf_id), topic="Topic", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(active_wf_id),
            logical_key="script",
            job_type="script",
            stage="SCRIPT",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id="w-1",
            lease_token="t-1",
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=active_wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id="w-1",
            lease_token="t-1",
            attempt_number=1,
            settings=settings,
        )

        handler = MockWorkflowStageHandler(
            output=WorkflowOutput(workflow_id=different_wf_id, content="Forged output")
        )
        runtime = StageRuntime(session, tmp_store, settings)

        with pytest.raises(
            SecurityViolationError, match="does not match active context workflow_id"
        ):
            await runtime.execute_stage(
                handler=handler,
                context=context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint="8" * 64,
                output_type=WorkflowOutput,
            )


@pytest.mark.asyncio
async def test_stage_runtime_cached_output_with_mismatched_workflow_id_raises(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Verify StageRuntime rejects cached artifacts whose serialized payload has a mismatched workflow_id."""
    settings = Settings(_env_file=None)
    active_wf_id = uuid.uuid4()
    forged_wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    fingerprint = "9" * 64

    forged_dto = WorkflowOutput(
        workflow_id=forged_wf_id, content="Inconsistent workflow cached DTO"
    )
    forged_bytes = forged_dto.model_dump_json(indent=2).encode("utf-8")
    actual_checksum = hashlib.sha256(forged_bytes).hexdigest()

    async with sqlite_session_factory() as session:
        art_id = uuid.uuid4()
        storage_path = tmp_store.build_path(active_wf_id, PipelineStage.SCRIPT, art_id, "json")
        await tmp_store.write_artifact(storage_path, forged_bytes)

        # Artifact metadata claims active_wf_id, but physical serialized bytes contain forged_wf_id
        art_model = ArtifactModel(
            id=str(art_id),
            workflow_id=str(active_wf_id),
            artifact_type="SCRIPT_RESULT",
            artifact_kind="SCRIPT_RESULT",
            storage_path=str(storage_path),
            checksum=actual_checksum,
            file_size_bytes=len(forged_bytes),
            lifecycle_status="AVAILABLE",
            stage="SCRIPT",
            stage_fingerprint=fingerprint,
        )
        session.add(art_model)
        await session.commit()

        context = create_stage_context(
            workflow_id=active_wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id="w-1",
            lease_token="t-1",
            attempt_number=1,
            settings=settings,
        )
        handler = MockWorkflowStageHandler(
            output=WorkflowOutput(workflow_id=active_wf_id, content="Normal output")
        )
        runtime = StageRuntime(session, tmp_store, settings)

        with pytest.raises(
            SecurityViolationError, match="does not match active context workflow_id"
        ):
            await runtime.execute_stage(
                handler=handler,
                context=context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint=fingerprint,
                output_type=WorkflowOutput,
            )


@pytest.mark.asyncio
async def test_stage_runtime_race_recovered_output_with_mismatched_workflow_id_raises(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
    monkeypatch,
):
    """Verify StageRuntime rejects race-recovered artifacts whose serialized payload has a mismatched workflow_id."""
    settings = Settings(_env_file=None)
    active_wf_id = uuid.uuid4()
    forged_wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    fingerprint = "7" * 64

    forged_dto = WorkflowOutput(workflow_id=forged_wf_id, content="Inconsistent race winner DTO")
    forged_bytes = forged_dto.model_dump_json(indent=2).encode("utf-8")
    actual_checksum = hashlib.sha256(forged_bytes).hexdigest()

    async with sqlite_session_factory() as session:
        art_id = uuid.uuid4()
        storage_path = tmp_store.build_path(active_wf_id, PipelineStage.SCRIPT, art_id, "json")
        await tmp_store.write_artifact(storage_path, forged_bytes)

        art_model = ArtifactModel(
            id=str(art_id),
            workflow_id=str(active_wf_id),
            artifact_type="SCRIPT_RESULT",
            artifact_kind="SCRIPT_RESULT",
            storage_path=str(storage_path),
            checksum=actual_checksum,
            file_size_bytes=len(forged_bytes),
            lifecycle_status="AVAILABLE",
            stage="SCRIPT",
            stage_fingerprint=fingerprint,
        )
        session.add(art_model)
        await session.commit()

        context = create_stage_context(
            workflow_id=active_wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id="w-1",
            lease_token="t-1",
            attempt_number=1,
            settings=settings,
        )
        handler = MockWorkflowStageHandler(
            output=WorkflowOutput(workflow_id=active_wf_id, content="Fresh execution")
        )
        runtime = StageRuntime(session, tmp_store, settings)

        # Force a FingerprintConflictError on create_pending to trigger the race recovery branch
        async def mock_create_pending(*args, **kwargs):
            raise FingerprintConflictError(
                "Simulated race conflict", stage="SCRIPT", stage_fingerprint=fingerprint
            )

        monkeypatch.setattr(runtime._repo, "create_pending", mock_create_pending)

        # Ensure find_by_fingerprint returns None initially so it enters step 2,
        # but returns art_model when re-queried in the conflict handler
        find_calls = 0

        async def mock_find_by_fingerprint(*args, **kwargs):
            nonlocal find_calls
            find_calls += 1
            if find_calls == 1:
                return None
            return art_model

        monkeypatch.setattr(runtime._repo, "find_by_fingerprint", mock_find_by_fingerprint)

        with pytest.raises(
            SecurityViolationError, match="does not match active context workflow_id"
        ):
            await runtime.execute_stage(
                handler=handler,
                context=context,
                stage=PipelineStage.SCRIPT,
                artifact_kind=ArtifactKind.SCRIPT_RESULT,
                fingerprint=fingerprint,
                output_type=WorkflowOutput,
            )
