"""Unit tests for ScriptStageHandler."""

import hashlib
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import HttpUrl, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agents.research.models import (
    NarrativeBeat,
    ResearchFact,
    ResearchResult,
    ResearchSource,
)
from app.agents.script.models import (
    ScriptRequest,
    ScriptResult,
    ScriptScene,
)
from app.config.settings import Settings
from app.db.base import Base
from app.db.models.artifact import ArtifactModel
from app.stages.artifact_errors import ArtifactUnavailableError, SecurityViolationError
from app.stages.artifact_store import ArtifactStore
from app.stages.context import StageContext
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.fingerprint import StageFingerprintCalculator
from app.stages.script import ScriptJobPayload, ScriptStageHandler


def make_dummy_research_result(
    wf_id: uuid.UUID, topic: str = "Quantum Computing"
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


def make_dummy_script_result(wf_id: uuid.UUID) -> ScriptResult:
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
    # Check tolerance for 60s: min 51.0, max 69.0.
    # total_words = ~135 -> duration = 57.86s (within 51-69s!)

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
async def test_script_stage_handler_load_input_success(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    wf_id = uuid.uuid4()
    research_id = uuid.uuid4()

    # 1. Write research artifact to store
    research_obj = make_dummy_research_result(wf_id, "Quantum Computing")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    research_checksum = hashlib.sha256(research_bytes).hexdigest()
    path = tmp_store.build_path(wf_id, PipelineStage.RESEARCH, research_id, "json")
    await tmp_store.write_artifact(path, research_bytes)

    # 2. Record AVAILABLE research artifact in DB
    async with sqlite_session_factory() as session:
        art_model = ArtifactModel(
            id=str(research_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=research_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="abc" * 21 + "a",
        )
        session.add(art_model)
        await session.commit()

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )

    payload = ScriptJobPayload(
        research_artifact_id=research_id,
        research_artifact_checksum=research_checksum,
        pacing_wpm=140,
        target_duration_seconds=60,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-script-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    script_req = await handler.load_input(context)
    assert script_req.workflow_id == wf_id
    assert script_req.pacing_wpm == 140
    assert script_req.research.topic == "Quantum Computing"
    assert script_req.research.workflow_id == wf_id


@pytest.mark.asyncio
async def test_script_stage_handler_load_input_detects_corrupted_artifact(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    wf_id = uuid.uuid4()
    research_id = uuid.uuid4()

    # 1. Write valid research artifact
    research_obj = make_dummy_research_result(wf_id, "Quantum Computing")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    research_checksum = hashlib.sha256(research_bytes).hexdigest()
    path = tmp_store.build_path(wf_id, PipelineStage.RESEARCH, research_id, "json")
    await tmp_store.write_artifact(path, research_bytes)

    # 2. Record in DB
    async with sqlite_session_factory() as session:
        art_model = ArtifactModel(
            id=str(research_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=research_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="abc" * 21 + "a",
        )
        session.add(art_model)
        await session.commit()

    # 3. Tamper with the file on disk
    path.write_bytes(b'{"corrupted": "tampered content"}')

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )
    payload = ScriptJobPayload(
        research_artifact_id=research_id,
        research_artifact_checksum=research_checksum,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-script-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(SecurityViolationError, match="checksum mismatch"):
        await handler.load_input(context)


@pytest.mark.asyncio
async def test_script_stage_handler_load_input_missing_artifact(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    wf_id = uuid.uuid4()
    missing_id = uuid.uuid4()

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )
    payload = ScriptJobPayload(
        research_artifact_id=missing_id,
        research_artifact_checksum="0" * 64,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-script-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(ArtifactUnavailableError, match="artifact record not found"):
        await handler.load_input(context)


def test_script_stage_handler_fingerprint_calculation():
    handler = ScriptStageHandler(session_factory=None, store=None)
    wf_id = uuid.uuid4()
    checksum = "a" * 64
    payload = ScriptJobPayload(
        research_artifact_id=uuid.uuid4(),
        research_artifact_checksum=checksum,
        pacing_wpm=150,
        target_duration_seconds=60,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    expected = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        pacing_wpm=150,
        target_duration_seconds=60,
    )
    actual = handler.calculate_fingerprint(None, context)
    assert actual == expected


def test_script_fingerprint_changes_with_tone():
    checksum = "c" * 64
    fp1 = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        tone="engaging, informative, and cinematic",
    )
    fp2 = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        tone="dramatic and intense",
    )
    assert fp1 != fp2


def test_script_fingerprint_changes_with_call_to_action():
    checksum = "c" * 64
    fp1 = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        call_to_action=None,
    )
    fp2 = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        call_to_action="Subscribe for more quantum insights!",
    )
    assert fp1 != fp2


def test_script_payload_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ScriptJobPayload(
            research_artifact_id=uuid.uuid4(),
            research_artifact_checksum="d" * 64,
            unexpected_field="forbidden",
        )


@pytest.mark.asyncio
async def test_script_stage_handler_load_input_detects_supplied_checksum_mismatch(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    """Payload specifies research checksum A, but artifact record has checksum B."""
    wf_id = uuid.uuid4()
    research_id = uuid.uuid4()

    research_obj = make_dummy_research_result(wf_id, "Quantum Computing")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    actual_checksum = hashlib.sha256(research_bytes).hexdigest()
    path = tmp_store.build_path(wf_id, PipelineStage.RESEARCH, research_id, "json")
    await tmp_store.write_artifact(path, research_bytes)

    async with sqlite_session_factory() as session:
        art_model = ArtifactModel(
            id=str(research_id),
            workflow_id=str(wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=actual_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="abc" * 21 + "a",
        )
        session.add(art_model)
        await session.commit()

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )
    # Supplied checksum differs from stored artifact checksum
    mismatched_checksum = "f" * 64
    payload = ScriptJobPayload(
        research_artifact_id=research_id,
        research_artifact_checksum=mismatched_checksum,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-script-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(SecurityViolationError, match="does not match expected job checksum"):
        await handler.load_input(context)


@pytest.mark.asyncio
async def test_script_stage_handler_load_input_rejects_wrong_stage_or_kind(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()

    data_bytes = b'{"not_research": true}'
    checksum = hashlib.sha256(data_bytes).hexdigest()
    path = tmp_store.build_path(wf_id, PipelineStage.SCRIPT, art_id, "json")
    await tmp_store.write_artifact(path, data_bytes)

    async with sqlite_session_factory() as session:
        art_model = ArtifactModel(
            id=str(art_id),
            workflow_id=str(wf_id),
            artifact_type="SCRIPT_RESULT",
            artifact_kind="SCRIPT_RESULT",
            storage_path=str(path),
            checksum=checksum,
            file_size_bytes=len(data_bytes),
            lifecycle_status="AVAILABLE",
            stage="SCRIPT",
            stage_fingerprint="def" * 21 + "d",
        )
        session.add(art_model)
        await session.commit()

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )
    payload = ScriptJobPayload(
        research_artifact_id=art_id,
        research_artifact_checksum=checksum,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-script-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(SecurityViolationError, match="Contract violation"):
        await handler.load_input(context)


@pytest.mark.asyncio
async def test_script_stage_handler_load_input_rejects_wrong_workflow_artifact(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    active_wf_id = uuid.uuid4()
    other_wf_id = uuid.uuid4()
    art_id = uuid.uuid4()

    research_obj = make_dummy_research_result(other_wf_id, "Quantum Computing")
    research_bytes = research_obj.model_dump_json(indent=2).encode("utf-8")
    actual_checksum = hashlib.sha256(research_bytes).hexdigest()
    path = tmp_store.build_path(other_wf_id, PipelineStage.RESEARCH, art_id, "json")
    await tmp_store.write_artifact(path, research_bytes)

    async with sqlite_session_factory() as session:
        art_model = ArtifactModel(
            id=str(art_id),
            workflow_id=str(other_wf_id),
            artifact_type="RESEARCH_RESULT",
            artifact_kind="RESEARCH_RESULT",
            storage_path=str(path),
            checksum=actual_checksum,
            file_size_bytes=len(research_bytes),
            lifecycle_status="AVAILABLE",
            stage="RESEARCH",
            stage_fingerprint="abc" * 21 + "a",
        )
        session.add(art_model)
        await session.commit()

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )
    payload = ScriptJobPayload(
        research_artifact_id=art_id,
        research_artifact_checksum=actual_checksum,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=active_wf_id,
        job_id="job-script-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(SecurityViolationError, match="Contract violation"):
        await handler.load_input(context)


@pytest.mark.asyncio
async def test_script_stage_handler_execute_delegates_to_agent():
    wf_id = uuid.uuid4()
    expected_result = make_dummy_script_result(wf_id)
    mock_agent = AsyncMock()
    mock_agent.run.return_value = expected_result

    handler = ScriptStageHandler(
        session_factory=None,
        store=None,
        agent=mock_agent,
    )
    req = ScriptRequest(
        workflow_id=wf_id,
        research=make_dummy_research_result(wf_id, "Quantum Computing"),
        pacing_wpm=140,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ScriptJobPayload(
            research_artifact_id=uuid.uuid4(),
            research_artifact_checksum="b" * 64,
        ),
    )

    res = await handler.execute(req, context)
    assert res == expected_result
    mock_agent.run.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_script_stage_handler_execute_rejects_wrong_workflow_id():
    active_wf_id = uuid.uuid4()
    other_wf_id = uuid.uuid4()

    mock_agent = AsyncMock()
    mock_agent.run.return_value = make_dummy_script_result(other_wf_id)

    handler = ScriptStageHandler(
        session_factory=None,
        store=None,
        agent=mock_agent,
    )
    req = ScriptRequest(
        workflow_id=active_wf_id,
        research=make_dummy_research_result(active_wf_id, "Quantum Computing"),
        pacing_wpm=140,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=active_wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ScriptJobPayload(
            research_artifact_id=uuid.uuid4(),
            research_artifact_checksum="b" * 64,
        ),
    )

    with pytest.raises(SecurityViolationError, match="does not match active context workflow_id"):
        await handler.execute(req, context)


def test_script_payload_validates_and_normalizes_hex_checksum():
    # Valid uppercase is normalized to lowercase
    p = ScriptJobPayload(
        research_artifact_id=uuid.uuid4(),
        research_artifact_checksum="A" * 64,
    )
    assert p.research_artifact_checksum == "a" * 64

    # Non-hex characters rejected
    with pytest.raises(ValidationError):
        ScriptJobPayload(
            research_artifact_id=uuid.uuid4(),
            research_artifact_checksum="z" * 64,
        )


@pytest.mark.asyncio
async def test_script_stage_handler_load_input_rejects_target_duration_mismatch(
    sqlite_session_factory, tmp_store: ArtifactStore
):
    wf_id = uuid.uuid4()
    job_id = uuid.uuid4()
    research = make_dummy_research_result(wf_id, "Quantum Computing")  # duration is 60
    serialized = research.model_dump_json(indent=2).encode("utf-8")
    real_checksum = hashlib.sha256(serialized).hexdigest()

    async with sqlite_session_factory() as session:
        from app.stages.stage_artifact_repository import StageArtifactRepository

        repo = StageArtifactRepository(session, tmp_store)
        pending = await repo.create_pending(
            workflow_id=wf_id,
            job_id=job_id,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            stage_fingerprint="0" * 64,
        )
        art_id = uuid.UUID(pending.id)
        await session.commit()

        await tmp_store.write_artifact(Path(pending.storage_path), serialized)
        await repo.finalize_administrative(art_id, admin_reason="test fixture")
        await session.commit()

    handler = ScriptStageHandler(
        session_factory=sqlite_session_factory,
        store=tmp_store,
    )
    # Payload specifies 90 seconds, but research output is 60 seconds
    payload = ScriptJobPayload(
        research_artifact_id=art_id,
        research_artifact_checksum=real_checksum,
        target_duration_seconds=90,
    )
    context = StageContext[ScriptJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(
        ValueError, match="does not match research artifact target_duration_seconds"
    ):
        await handler.load_input(context)


def test_script_fingerprint_omitted_vs_explicit_matching_duration_produces_identical_fingerprint():
    """Verify that omitting target_duration_seconds vs providing matching duration produces identical fingerprint."""
    checksum = "a" * 64
    fp_omitted = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        pacing_wpm=140,
        target_duration_seconds=None,
    )
    fp_explicit = StageFingerprintCalculator.for_script(
        research_artifact_checksum=checksum,
        pacing_wpm=140,
        target_duration_seconds=60,
    )
    assert fp_omitted == fp_explicit


def test_stage_fingerprint_version_invalidation():
    """Verify that updating stage version produces a distinct fingerprint for cache invalidation."""
    checksum = "a" * 64
    script_v1 = StageFingerprintCalculator.for_script(checksum, version="v1")
    script_v2 = StageFingerprintCalculator.for_script(checksum, version="v2")
    assert script_v1 != script_v2

    research_v1 = StageFingerprintCalculator.for_research("AI", 60, version="v1")
    research_v2 = StageFingerprintCalculator.for_research("AI", 60, version="v2")
    assert research_v1 != research_v2
