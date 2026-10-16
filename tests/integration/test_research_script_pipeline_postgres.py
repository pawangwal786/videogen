"""PostgreSQL 16 integration tests for the Research -> Script pipeline runtime.

Verifies end-to-end execution, artifact-driven contracts, deterministic stage
idempotency, and cryptographic corruption detection on a live PostgreSQL database.
"""

import hashlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import HttpUrl
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agents.research.models import (
    NarrativeBeat,
    ResearchFact,
    ResearchResult,
    ResearchSource,
)
from app.agents.script.models import (
    ScriptResult,
    ScriptScene,
)
from app.config.settings import Settings
from app.orchestration import JobWorker, create_production_worker
from app.orchestration.errors import LeaseConflictError
from app.orchestration.orchestrator import WorkflowOrchestrator
from app.orchestration.state_machine import JobStage, JobStatus
from app.repositories.job import JobRepository
from app.repositories.workflow import WorkflowRepository
from app.stages.artifact_store import ArtifactStore
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.research import ResearchStageHandler
from app.stages.script import ScriptStageHandler
from app.stages.stage_artifact_repository import StageArtifactRepository


@pytest.fixture(autouse=True)
async def clean_database(pg_engine: AsyncEngine):
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE artifacts, job_attempts, jobs, workflows RESTART IDENTITY CASCADE")
        )
    yield
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE artifacts, job_attempts, jobs, workflows RESTART IDENTITY CASCADE")
        )


def make_valid_research_result(
    wf_id: uuid.UUID, topic: str = "Quantum Mechanics"
) -> ResearchResult:
    return ResearchResult(
        workflow_id=wf_id,
        topic=topic,
        research_depth="standard",
        research_method="llm_synthesis",
        target_duration_seconds=60,
        executive_summary="Executive summary about quantum computing fundamentals and qubits.",
        core_angle="How quantum computing transforms cryptography",
        target_audience="Tech enthusiasts",
        key_takeaways=[
            "Qubits operate in superposition.",
            "Quantum entanglement enables exponential computational acceleration.",
        ],
        facts=[
            ResearchFact(
                claim="Qubits exist in superposition states.",
                verification_status="source_backed",
                sources=[
                    ResearchSource(
                        title="Quantum Physics Overview",
                        url=HttpUrl("https://example.com/quantum"),
                    )
                ],
            ),
            ResearchFact(
                claim="Quantum entanglement connects states across distance.",
                verification_status="source_backed",
                sources=[
                    ResearchSource(
                        title="Entanglement Research",
                        url=HttpUrl("https://example.com/entanglement"),
                    )
                ],
            ),
            ResearchFact(
                claim="Shor's algorithm can factor integers in polynomial time.",
                verification_status="source_backed",
                sources=[
                    ResearchSource(
                        title="Shor Algorithm Analysis",
                        url=HttpUrl("https://example.com/shor"),
                    )
                ],
            ),
        ],
        suggested_narrative_arc=[
            NarrativeBeat(
                sequence=1,
                beat_type="hook",
                description="Intro to quantum computing mystery",
                suggested_duration_ratio=0.2,
            ),
            NarrativeBeat(
                sequence=2,
                beat_type="insight",
                description="Qubit superposition explained",
                suggested_duration_ratio=0.5,
            ),
            NarrativeBeat(
                sequence=3,
                beat_type="climax",
                description="Impact on future cryptography",
                suggested_duration_ratio=0.3,
            ),
        ],
        visual_themes=["neon futuristic circuits", "abstract probability waves"],
    )


def make_valid_script_result(wf_id: uuid.UUID) -> ScriptResult:
    s1_text = (
        "Imagine a supercomputer capable of breaking modern encryption in minutes. Quantum computers harness "
        "the bizarre laws of physics where subatomic particles exist in multiple simultaneous states at once "
        "rather than simple binary zeroes and ones."
    )
    s2_text = (
        "Through the principles of quantum superposition and entanglement, advanced qubits process intricate "
        "equations simultaneously across parallel computational pathways. Mathematical challenges that would demand "
        "ten thousand years from traditional silicon architectures are resolved in mere seconds by manipulating "
        "coherent quantum states. These unprecedented computational leaps will unlock revolutionary discoveries across "
        "molecular medicine, novel superconductive battery chemistry, complex logistical optimization, and financial portfolio risk analysis worldwide."
    )
    s3_text = (
        "As technology giants accelerate their race toward fault-tolerant quantum hardware, modern digital "
        "infrastructure approaches a fundamental transformation. The cryptographic landscape will completely change, "
        "and humanity will step into an extraordinary technological epoch where quantum supremacy redefines the very boundaries of human potential."
    )
    words_1 = len(s1_text.split())
    words_2 = len(s2_text.split())
    words_3 = len(s3_text.split())
    total_words = words_1 + words_2 + words_3
    pacing = 140
    duration = round((total_words / pacing) * 60.0, 2)

    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration=s1_text,
            visual_direction="Close-up of a glowing quantum processor suspended in deep dilution refrigerator.",
            target_keywords=["quantum", "computer", "processor"],
            estimated_duration_seconds=round((words_1 / pacing) * 60.0, 2),
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="insight",
            narration=s2_text,
            visual_direction="Animated laser pulses reflecting off gold coaxial quantum wiring.",
            target_keywords=["superposition", "entanglement", "qubits"],
            estimated_duration_seconds=round((words_2 / pacing) * 60.0, 2),
        ),
        ScriptScene(
            scene_number=3,
            narrative_beat_sequence=3,
            beat_type="climax",
            narration=s3_text,
            visual_direction="Cinematic camera pull-back showcasing glowing computational servers.",
            target_keywords=["cryptography", "future", "technology"],
            estimated_duration_seconds=round((words_3 / pacing) * 60.0, 2),
        ),
    ]

    return ScriptResult(
        workflow_id=wf_id,
        title="Quantum Leap: The Revolution",
        target_duration_seconds=60,
        estimated_duration_seconds=duration,
        total_word_count=total_words,
        scenes=scenes,
        full_narration=f"{s1_text} {s2_text} {s3_text}",
        pacing_wpm=pacing,
    )


@pytest.mark.asyncio
async def test_postgres_research_to_script_pipeline_end_to_end(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify full execution of Research -> Script pipeline on PostgreSQL 16."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_e2e", fsync=False)
    settings = Settings(_env_file=None)

    orchestrator = WorkflowOrchestrator(session_factory=session_factory)
    worker = JobWorker(
        session_factory=session_factory,
        artifact_store=store,
        settings=settings,
        orchestrator=orchestrator,
    )

    # 1. Create workflow -> spawns initial RESEARCH job
    wf = await orchestrator.create_workflow(topic="Quantum Mechanics")
    wf_uuid = uuid.UUID(wf.id)
    assert wf.current_stage == JobStage.RESEARCH.value

    # 2. Register handlers
    mock_research_agent = AsyncMock()
    mock_research_agent.run.return_value = make_valid_research_result(wf_uuid, "Quantum Mechanics")
    research_handler = ResearchStageHandler(agent=mock_research_agent)

    mock_script_agent = AsyncMock()
    mock_script_agent.run.return_value = make_valid_script_result(wf_uuid)
    script_handler = ScriptStageHandler(
        session_factory=session_factory,
        store=store,
        agent=mock_script_agent,
    )

    worker.register_stage_handler(
        job_type="research",
        handler=research_handler,
        stage=PipelineStage.RESEARCH,
        artifact_kind=ArtifactKind.RESEARCH_RESULT,
        output_type=ResearchResult,
    )
    worker.register_stage_handler(
        job_type="script",
        handler=script_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptResult,
    )

    # 3. Worker executes RESEARCH job
    worked = await worker.run_once()
    assert worked is True
    assert mock_research_agent.run.call_count == 1

    # Verify research job output contains artifact ID and checksum
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        research_job = next(j for j in jobs if j.logical_key == "research")
        assert research_job.status == JobStatus.COMPLETED.value
        assert "_artifact_id" in research_job.output_payload
        assert "_checksum" in research_job.output_payload
        research_art_id = research_job.output_payload["_artifact_id"]
        research_checksum = research_job.output_payload["_checksum"]

    # 4. Advance workflow: RESEARCH -> SCRIPT
    advanced_wf = await orchestrator.advance_workflow(wf.id)
    assert advanced_wf.current_stage == JobStage.SCRIPT.value

    # Verify script job was created with durable artifact references, NOT embedded dicts
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        script_job = next(j for j in jobs if j.logical_key == "script")
        assert script_job.status == JobStatus.PENDING.value
        assert script_job.input_payload["research_artifact_id"] == research_art_id
        assert script_job.input_payload["research_artifact_checksum"] == research_checksum
        assert "research_output" not in script_job.input_payload

    # 5. Worker executes SCRIPT job
    worked = await worker.run_once()
    assert worked is True
    assert mock_script_agent.run.call_count == 1

    # Verify script job completed and produced durable SCRIPT_RESULT artifact
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf.id)
        script_job = next(j for j in jobs if j.logical_key == "script")
        assert script_job.status == JobStatus.COMPLETED.value
        assert "_artifact_id" in script_job.output_payload
        assert "_checksum" in script_job.output_payload

    # 6. Advance workflow: SCRIPT -> STORYBOARD
    advanced_wf = await orchestrator.advance_workflow(wf.id)
    assert advanced_wf.current_stage == JobStage.STORYBOARD.value


@pytest.mark.asyncio
async def test_postgres_script_stage_idempotency_cache_hit(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify that repeating a Script job with identical research checksum skips agent execution."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_idem", fsync=False)
    settings = Settings(_env_file=None)

    worker = JobWorker(session_factory=session_factory, artifact_store=store, settings=settings)

    wf_id = uuid.uuid4()
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Superconductors")
        await session.commit()
        wf_id = uuid.UUID(wf.id)

    # Prepare and store valid research artifact
    research_obj = make_valid_research_result(wf_id, "Superconductors")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    research_checksum = hashlib.sha256(research_bytes).hexdigest()
    res_art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.RESEARCH, res_art_id, "json")
    await store.write_artifact(path, research_bytes)

    async with session_factory() as session:
        from app.db.models.artifact import ArtifactModel

        art_model = ArtifactModel(
            id=str(res_art_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=research_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="1" * 64,
        )
        session.add(art_model)
        await session.commit()

    mock_script_agent = AsyncMock()
    mock_script_agent.run.return_value = make_valid_script_result(wf_id)
    script_handler = ScriptStageHandler(
        session_factory=session_factory,
        store=store,
        agent=mock_script_agent,
    )
    worker.register_stage_handler(
        job_type="script",
        handler=script_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptResult,
    )

    # Job 1: First run computes artifact
    async with session_factory() as session:
        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="script_1",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={
                "research_artifact_id": str(res_art_id),
                "research_artifact_checksum": research_checksum,
                "pacing_wpm": 140,
                "target_duration_seconds": 60,
            },
        )
        await session.commit()

    worked_1 = await worker.run_once()
    assert worked_1 is True
    assert mock_script_agent.run.call_count == 1

    # Job 2: Second run with identical inputs reuses AVAILABLE artifact
    async with session_factory() as session:
        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="script_2",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={
                "research_artifact_id": str(res_art_id),
                "research_artifact_checksum": research_checksum,
                "pacing_wpm": 140,
                "target_duration_seconds": 60,
            },
        )
        await session.commit()

    worked_2 = await worker.run_once()
    assert worked_2 is True
    # Agent was NOT executed again (cached hit)
    assert mock_script_agent.run.call_count == 1

    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(str(wf_id))
        j1 = next(j for j in jobs if j.logical_key == "script_1")
        j2 = next(j for j in jobs if j.logical_key == "script_2")
        assert j1.output_payload["_cached"] is False
        assert j2.output_payload["_cached"] is True
        assert j1.output_payload["_artifact_id"] == j2.output_payload["_artifact_id"]


@pytest.mark.asyncio
async def test_postgres_script_stage_detects_corrupted_upstream_artifact(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify that physical file corruption of an upstream artifact aborts downstream execution."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_corrupt", fsync=False)
    settings = Settings(_env_file=None)

    worker = JobWorker(session_factory=session_factory, artifact_store=store, settings=settings)

    wf_id = uuid.uuid4()
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Corrupted Test")
        await session.commit()
        wf_id = uuid.UUID(wf.id)

    research_obj = make_valid_research_result(wf_id, "Corrupted Test")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    research_checksum = hashlib.sha256(research_bytes).hexdigest()
    res_art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.RESEARCH, res_art_id, "json")
    await store.write_artifact(path, research_bytes)

    async with session_factory() as session:
        from app.db.models.artifact import ArtifactModel

        art_model = ArtifactModel(
            id=str(res_art_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=research_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="2" * 64,
        )
        session.add(art_model)
        await session.commit()

    # Tamper with file on disk
    path.write_bytes(b'{"corrupted": true}')

    mock_script_agent = AsyncMock()
    script_handler = ScriptStageHandler(
        session_factory=session_factory,
        store=store,
        agent=mock_script_agent,
    )
    worker.register_stage_handler(
        job_type="script",
        handler=script_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptResult,
    )

    async with session_factory() as session:
        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="script_corrupt",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={
                "research_artifact_id": str(res_art_id),
                "research_artifact_checksum": research_checksum,
                "pacing_wpm": 140,
                "target_duration_seconds": 60,
            },
        )
        await session.commit()

    worked = await worker.run_once()
    assert worked is True
    # Agent was never called because load_input failed cryptographic verification
    assert mock_script_agent.run.call_count == 0

    # Job attempt failed with SecurityViolationError
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(str(wf_id))
        script_job = await job_repo.get_job(jobs[0].id, load_attempts=True)
        assert script_job is not None
        assert len(script_job.attempts) == 1
        assert script_job.attempts[0].status == "FAILED"
        assert "checksum mismatch" in (script_job.attempts[0].error_message or "")


@pytest.mark.asyncio
async def test_postgres_supplied_checksum_mismatch_aborts_script(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify that if the job payload expects checksum A but the artifact record has checksum B, execution halts."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_mismatch", fsync=False)
    settings = Settings(_env_file=None)

    orchestrator = WorkflowOrchestrator(session_factory=session_factory)
    worker = JobWorker(
        session_factory=session_factory,
        artifact_store=store,
        settings=settings,
        orchestrator=orchestrator,
    )

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Quantum Mechanics")
        await session.commit()
        wf_id = uuid.UUID(wf.id)

    res_art_id = uuid.uuid4()
    research_obj = make_valid_research_result(wf_id, "Quantum Mechanics")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    actual_checksum = hashlib.sha256(research_bytes).hexdigest()
    path = store.build_path(wf_id, PipelineStage.RESEARCH, res_art_id, "json")
    await store.write_artifact(path, research_bytes)

    async with session_factory() as session:
        from app.db.models.artifact import ArtifactModel

        art_model = ArtifactModel(
            id=str(res_art_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=actual_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="1" * 64,
        )
        session.add(art_model)
        await session.commit()

    mock_script_agent = AsyncMock()
    script_handler = ScriptStageHandler(
        session_factory=session_factory,
        store=store,
        agent=mock_script_agent,
    )
    worker.register_stage_handler(
        job_type="script",
        handler=script_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptResult,
    )

    mismatched_checksum = "e" * 64
    async with session_factory() as session:
        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="script_mismatch",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={
                "research_artifact_id": str(res_art_id),
                "research_artifact_checksum": mismatched_checksum,
                "pacing_wpm": 140,
            },
        )
        await session.commit()

    worked = await worker.run_once()
    assert worked is True
    assert mock_script_agent.run.call_count == 0

    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(str(wf_id))
        script_job = await job_repo.get_job(jobs[0].id, load_attempts=True)
        assert script_job is not None
        assert script_job.attempts[0].status == "FAILED"
        assert "does not match expected job checksum" in (
            script_job.attempts[0].error_message or ""
        )


@pytest.mark.asyncio
async def test_postgres_script_stage_different_tone_executes_new_script(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify that different creative instructions (e.g. tone) generate distinct cache fingerprints."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_tone", fsync=False)
    settings = Settings(_env_file=None)

    orchestrator = WorkflowOrchestrator(session_factory=session_factory)
    worker = JobWorker(
        session_factory=session_factory,
        artifact_store=store,
        settings=settings,
        orchestrator=orchestrator,
    )

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Quantum Mechanics")
        await session.commit()
        wf_id = uuid.UUID(wf.id)

    res_art_id = uuid.uuid4()
    research_obj = make_valid_research_result(wf_id, "Quantum Mechanics")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    research_checksum = hashlib.sha256(research_bytes).hexdigest()
    path = store.build_path(wf_id, PipelineStage.RESEARCH, res_art_id, "json")
    await store.write_artifact(path, research_bytes)

    async with session_factory() as session:
        from app.db.models.artifact import ArtifactModel

        art_model = ArtifactModel(
            id=str(res_art_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=research_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="1" * 64,
        )
        session.add(art_model)
        await session.commit()

    mock_script_agent = AsyncMock()
    mock_script_agent.run.return_value = make_valid_script_result(wf_id)
    script_handler = ScriptStageHandler(
        session_factory=session_factory,
        store=store,
        agent=mock_script_agent,
    )
    worker.register_stage_handler(
        job_type="script",
        handler=script_handler,
        stage=PipelineStage.SCRIPT,
        artifact_kind=ArtifactKind.SCRIPT_RESULT,
        output_type=ScriptResult,
    )

    # Job 1: tone = cinematic
    async with session_factory() as session:
        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="script_cinematic",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={
                "research_artifact_id": str(res_art_id),
                "research_artifact_checksum": research_checksum,
                "pacing_wpm": 140,
                "tone": "cinematic and inspiring",
            },
        )
        await session.commit()

    worked_1 = await worker.run_once()
    assert worked_1 is True
    assert mock_script_agent.run.call_count == 1

    # Job 2: tone = comedic (must NOT hit cache of cinematic run)
    async with session_factory() as session:
        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="script_comedic",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={
                "research_artifact_id": str(res_art_id),
                "research_artifact_checksum": research_checksum,
                "pacing_wpm": 140,
                "tone": "comedic and lighthearted",
            },
        )
        await session.commit()

    worked_2 = await worker.run_once()
    assert worked_2 is True
    # Agent was called again because fingerprint differs
    assert mock_script_agent.run.call_count == 2


@pytest.mark.asyncio
async def test_postgres_stale_lease_cannot_finalize_artifact(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify that a worker attempt with expired or incorrect lease token fails to finalize on PostgreSQL 16."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    store = ArtifactStore(tmp_path / "artifacts_stale", fsync=False)

    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Quantum Mechanics")
        await session.commit()
        wf_id = uuid.UUID(wf.id)
        job_repo = JobRepository(session)
        created_job = await job_repo.create_job(
            workflow_id=str(wf_id),
            logical_key="test_job",
            job_type="script",
            stage=JobStage.SCRIPT.value,
            input_payload={},
        )
        await session.commit()
        job_id = uuid.UUID(created_job.id)

    async with session_factory() as session:
        repo = StageArtifactRepository(session, store)
        pending = await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.SCRIPT,
            artifact_kind=ArtifactKind.SCRIPT_RESULT,
            stage_fingerprint="a" * 64,
        )
        await session.commit()

        # Write physical file
        path = Path(pending.storage_path)
        await store.write_artifact(path, b'{"valid": true}')

        # Attempt to finalize with stale/unmatched worker credentials
        with pytest.raises(LeaseConflictError):
            await repo.finalize(
                artifact_id=uuid.UUID(pending.id),
                worker_id="forged-worker-id",
                lease_token="forged-lease-token",
            )


@pytest.mark.asyncio
async def test_postgres_production_worker_composition_and_execution(
    pg_engine: AsyncEngine,
    tmp_path: Path,
):
    """Verify create_production_worker wires handlers and executes jobs end-to-end without manual registration."""
    session_factory = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    settings = Settings(
        videogen_artifact_root=str(tmp_path / "prod_artifacts"),
        videogen_artifact_fsync=False,
        _env_file=None,
    )

    # 1. Create workflow and research job first to obtain workflow_id
    async with session_factory() as session:
        wf_repo = WorkflowRepository(session)
        wf = await wf_repo.create_workflow(topic="Superconductivity")
        await session.commit()
        wf_id = wf.id

        job_repo = JobRepository(session)
        await job_repo.create_job(
            workflow_id=wf_id,
            logical_key="research",
            job_type="research",
            stage=JobStage.RESEARCH.value,
            input_payload={
                "topic": "Superconductivity",
                "target_duration_seconds": 60,
                "target_audience": "Engineers",
            },
        )
        await session.commit()

    mock_text_model = AsyncMock()
    mock_text_model.model_name = "gemini-2.5-flash"
    mock_text_model.generate.return_value = make_valid_research_result(
        wf_id=wf_id, topic="Superconductivity"
    ).model_dump_json()

    # 2. Use the production composition root
    worker = create_production_worker(
        session_factory=session_factory,
        settings=settings,
        worker_id="prod-worker-1",
        model=mock_text_model,
    )

    # 3. Verify handlers are pre-registered
    assert "research" in worker._handlers
    assert "script" in worker._handlers

    # 4. Execute through the wired production worker
    worked = await worker.run_once()
    assert worked is True

    # 5. Verify research job completed with valid artifact
    async with session_factory() as session:
        job_repo = JobRepository(session)
        jobs = await job_repo.list_jobs_for_workflow(wf_id)
        r_job = next(j for j in jobs if j.logical_key == "research")
        assert r_job.status == JobStatus.COMPLETED.value
        assert "_artifact_id" in r_job.output_payload
        assert "_checksum" in r_job.output_payload

        # Verify artifact is AVAILABLE in PostgreSQL
        art_repo = StageArtifactRepository(session, worker._artifact_store)
        art = await art_repo.get_available(
            uuid.UUID(r_job.output_payload["_artifact_id"]),
            verify_integrity=True,
        )
        assert art.lifecycle_status == "AVAILABLE"
        assert art.checksum == r_job.output_payload["_checksum"]
