"""Unit tests for ResearchStageHandler."""

import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import HttpUrl, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agents.research.agent import ResearchAgent
from app.agents.research.models import (
    NarrativeBeat,
    ResearchFact,
    ResearchRequest,
    ResearchResult,
    ResearchSource,
)
from app.config.settings import Settings
from app.db.base import Base
from app.db.models.job import JobAttemptModel, JobModel
from app.db.models.workflow import WorkflowModel
from app.orchestration.state_machine import AttemptStatus, JobStatus
from app.stages.artifact_store import ArtifactStore
from app.stages.context import StageContext
from app.stages.dto import ArtifactKind, PipelineStage
from app.stages.fingerprint import StageFingerprintCalculator
from app.stages.research import ResearchJobPayload, ResearchStageHandler
from app.stages.runtime import StageRuntime, create_stage_context


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
async def test_research_stage_handler_load_input():
    handler = ResearchStageHandler()
    wf_id = uuid.uuid4()
    payload = ResearchJobPayload(
        topic="Artificial Intelligence",
        target_duration_seconds=90,
        target_audience="Developers",
        research_depth="deep",
    )
    context = StageContext[ResearchJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    req = await handler.load_input(context)
    assert req.workflow_id == wf_id
    assert req.topic == "Artificial Intelligence"
    assert req.target_duration_seconds == 90
    assert req.target_audience == "Developers"
    assert req.research_depth == "deep"


def test_research_stage_handler_fingerprint_calculation():
    handler = ResearchStageHandler()
    wf_id = uuid.uuid4()
    payload = ResearchJobPayload(
        topic="Robotics",
        target_duration_seconds=60,
        model_name="gemini-2.5-pro",
    )
    context = StageContext[ResearchJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    expected = StageFingerprintCalculator.for_research(
        topic="Robotics",
        target_duration_seconds=60,
        model_name="gemini-2.5-pro",
    )
    actual = handler.calculate_fingerprint(None, context)
    assert actual == expected


def test_research_fingerprint_changes_with_audience():
    fp1 = StageFingerprintCalculator.for_research(
        topic="Physics",
        target_duration_seconds=60,
        target_audience="General audience",
    )
    fp2 = StageFingerprintCalculator.for_research(
        topic="Physics",
        target_duration_seconds=60,
        target_audience="High school students",
    )
    assert fp1 != fp2


def test_research_fingerprint_changes_with_research_depth():
    fp1 = StageFingerprintCalculator.for_research(
        topic="Physics",
        target_duration_seconds=60,
        research_depth="standard",
    )
    fp2 = StageFingerprintCalculator.for_research(
        topic="Physics",
        target_duration_seconds=60,
        research_depth="deep",
    )
    assert fp1 != fp2


def test_research_payload_rejects_blank_or_whitespace_topic():
    with pytest.raises(ValidationError):
        ResearchJobPayload(topic="   ")

    with pytest.raises(ValidationError):
        ResearchJobPayload(topic="")


def test_research_payload_forbids_unknown_fields():
    with pytest.raises(ValidationError):
        ResearchJobPayload(topic="Robotics", unexpected_extra_field="malicious")


@pytest.mark.asyncio
async def test_research_stage_handler_execute_delegates_to_agent():
    wf_id = uuid.uuid4()
    expected_result = make_dummy_research_result(wf_id, "Quantum Computing")

    mock_agent = AsyncMock()
    mock_agent.run.return_value = expected_result

    handler = ResearchStageHandler(agent=mock_agent)
    req = ResearchRequest(
        workflow_id=wf_id,
        topic="Quantum Computing",
        target_duration_seconds=60,
    )
    context = StageContext[ResearchJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ResearchJobPayload(topic="Quantum Computing"),
    )

    res = await handler.execute(req, context)
    assert res == expected_result
    mock_agent.run.assert_awaited_once_with(req)


@pytest.mark.asyncio
async def test_research_stage_runtime_full_execution(
    sqlite_session_factory,
    tmp_store: ArtifactStore,
):
    wf_id = uuid.uuid4()
    job_id = str(uuid.uuid4())
    attempt_id = str(uuid.uuid4())
    settings = Settings(_env_file=None)

    expected_result = make_dummy_research_result(wf_id, "Quantum Mechanics")
    mock_agent = AsyncMock()
    mock_agent.run.return_value = expected_result
    handler = ResearchStageHandler(agent=mock_agent)

    async with sqlite_session_factory() as session:
        wf = WorkflowModel(id=str(wf_id), topic="Quantum Mechanics", status="RUNNING")
        job = JobModel(
            id=job_id,
            workflow_id=str(wf_id),
            logical_key="research",
            job_type="research",
            stage="RESEARCH",
            status=JobStatus.CLAIMED.value,
            current_attempt_id=attempt_id,
        )
        attempt = JobAttemptModel(
            id=attempt_id,
            job_id=job_id,
            attempt_number=1,
            worker_id="worker-1",
            lease_token="token-1",
            status=AttemptStatus.CLAIMED.value,
        )
        session.add_all([wf, job, attempt])
        await session.commit()

        context = create_stage_context(
            workflow_id=wf_id,
            job_id=job_id,
            attempt_id=attempt_id,
            worker_id="worker-1",
            lease_token="token-1",
            attempt_number=1,
            settings=settings,
            payload=ResearchJobPayload(topic="Quantum Mechanics"),
        )

        runtime = StageRuntime(session, tmp_store, settings)
        fingerprint = handler.calculate_fingerprint(job, context)

        result = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            fingerprint=fingerprint,
            output_type=ResearchResult,
        )

        assert result.cached is False
        assert result.output.topic == "Quantum Mechanics"
        assert Path(result.storage_path).exists()
        assert len(result.checksum) == 64

        # Second execution: idempotency cache hit
        cached_result = await runtime.execute_stage(
            handler=handler,
            context=context,
            stage=PipelineStage.RESEARCH,
            artifact_kind=ArtifactKind.RESEARCH_RESULT,
            fingerprint=fingerprint,
            output_type=ResearchResult,
        )
        assert cached_result.cached is True
        assert cached_result.artifact_id == result.artifact_id
        assert cached_result.checksum == result.checksum
        assert mock_agent.run.call_count == 1


@pytest.mark.asyncio
async def test_research_stage_handler_execute_rejects_wrong_workflow_id():
    from app.stages.artifact_errors import SecurityViolationError

    context_wf_id = uuid.uuid4()
    different_wf_id = uuid.uuid4()

    mock_agent = AsyncMock()
    mock_agent.run.return_value = make_dummy_research_result(different_wf_id, "Quantum Computing")

    handler = ResearchStageHandler(agent=mock_agent)
    req = ResearchRequest(
        workflow_id=context_wf_id,
        topic="Quantum Computing",
        target_duration_seconds=60,
    )
    context = StageContext[ResearchJobPayload](
        workflow_id=context_wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ResearchJobPayload(topic="Quantum Computing"),
    )

    with pytest.raises(SecurityViolationError, match="does not match active context workflow_id"):
        await handler.execute(req, context)


@pytest.mark.asyncio
async def test_research_stage_handler_model_selection_propagated_to_request():
    handler = ResearchStageHandler()
    wf_id = uuid.uuid4()
    payload = ResearchJobPayload(
        topic="Neural Networks",
        target_duration_seconds=60,
        model_name="gemini-2.5-pro",
    )
    context = StageContext[ResearchJobPayload](
        workflow_id=wf_id,
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    req = await handler.load_input(context)
    assert req.model_name == "gemini-2.5-pro"
    fp = handler.calculate_fingerprint(None, context)
    assert fp == StageFingerprintCalculator.for_research(
        topic="Neural Networks",
        target_duration_seconds=60,
        model_name="gemini-2.5-pro",
    )


def test_research_payload_rejects_blank_target_audience_or_model():
    with pytest.raises(ValidationError):
        ResearchJobPayload(topic="Robotics", target_audience="   ")
    with pytest.raises(ValidationError):
        ResearchJobPayload(topic="Robotics", model_name="   ")


@pytest.mark.asyncio
async def test_research_agent_executes_requested_model():
    wf_id = uuid.uuid4()
    mock_model = AsyncMock()
    mock_model.generate.return_value = make_dummy_research_result(
        wf_id=wf_id, topic="Artificial Intelligence"
    ).model_dump_json()
    agent = ResearchAgent(model=mock_model)
    req = ResearchRequest(
        workflow_id=wf_id,
        topic="Artificial Intelligence",
        model_name="gemini-2.5-pro",
    )
    await agent.run(req)

    # Assert that model.generate was invoked with model_name="gemini-2.5-pro"
    mock_model.generate.assert_awaited_once()
    _, kwargs = mock_model.generate.await_args
    assert kwargs.get("model_name") == "gemini-2.5-pro"


def test_model_router_configuration_represented_in_fingerprint():
    from app.models.gemini import GeminiTextModel
    from app.models.openrouter import OpenRouterTextModel
    from app.models.router import ModelProvider, ModelRouter

    primary_model = GeminiTextModel(api_key="fake-key", model_name="gemini-2.5-flash")
    fallback_model = OpenRouterTextModel(api_key="fake-key", model_name="google/gemini-2.5-flash")
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: primary_model,
            ModelProvider.OPENROUTER: fallback_model,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )
    # router.model_identity preserves composite configuration with provider prefixes
    assert (
        router.model_identity
        == "gemini:gemini-2.5-flash:fallback=openrouter:google/gemini-2.5-flash"
    )
    # router.model_name preserves clean API-compatible primary model identifier
    assert router.model_name == "gemini-2.5-flash"

    agent = ResearchAgent(model=router)
    handler = ResearchStageHandler(agent=agent)
    payload = ResearchJobPayload(topic="Quantum Physics")
    resolved_identity = handler._canonical_model_identity(payload)
    assert (
        resolved_identity == "gemini:gemini-2.5-flash:fallback=openrouter:google/gemini-2.5-flash"
    )


@pytest.mark.asyncio
async def test_research_stage_handler_load_input_does_not_leak_composite_identity():
    from app.models.gemini import GeminiTextModel
    from app.models.openrouter import OpenRouterTextModel
    from app.models.router import ModelProvider, ModelRouter

    primary_model = GeminiTextModel(api_key="fake-key", model_name="gemini-2.5-flash")
    fallback_model = OpenRouterTextModel(api_key="fake-key", model_name="google/gemini-2.5-flash")
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: primary_model,
            ModelProvider.OPENROUTER: fallback_model,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )
    agent = ResearchAgent(model=router)
    handler = ResearchStageHandler(agent=agent)

    # 1. Default payload without explicit override produces model_name=None in request
    ctx_default = StageContext[ResearchJobPayload](
        workflow_id=uuid.uuid4(),
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ResearchJobPayload(topic="Quantum Physics"),
    )
    req_default = await handler.load_input(ctx_default)
    assert req_default.model_name is None

    # 2. Explicit override in payload is propagated directly
    ctx_override = StageContext[ResearchJobPayload](
        workflow_id=uuid.uuid4(),
        job_id="job-2",
        attempt_id="att-2",
        worker_id="worker-1",
        lease_token="lease-2",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ResearchJobPayload(topic="Quantum Physics", model_name="gemini-2.5-pro"),
    )
    req_override = await handler.load_input(ctx_override)
    assert req_override.model_name == "gemini-2.5-pro"


@pytest.mark.asyncio
async def test_research_agent_internal_type_error_propagates_without_retry():
    """Verify that an internal TypeError raised by a model provider propagates immediately without silent retry."""
    wf_id = uuid.uuid4()
    mock_model = AsyncMock()
    mock_model.generate.side_effect = TypeError(
        "Internal unexpected argument error inside provider"
    )
    agent = ResearchAgent(model=mock_model)
    req = ResearchRequest(
        workflow_id=wf_id,
        topic="Artificial Intelligence",
        model_name="gemini-2.5-pro",
    )

    with pytest.raises(TypeError, match="Internal unexpected argument error inside provider"):
        await agent.run(req)

    # Must have only attempted generation once (no silent retrying without model_name)
    mock_model.generate.assert_awaited_once()


def test_explicit_model_override_fingerprint_changes_when_fallback_model_changes():
    """Verify that an explicit primary model override incorporates fallback configuration into its fingerprint."""
    from app.models.gemini import GeminiTextModel
    from app.models.openrouter import OpenRouterTextModel
    from app.models.router import ModelProvider, ModelRouter

    primary_model = GeminiTextModel(api_key="fake-key", model_name="gemini-2.5-flash")
    fallback_model_1 = OpenRouterTextModel(api_key="fake-key", model_name="google/gemini-2.5-flash")
    router_1 = ModelRouter(
        providers={
            ModelProvider.GEMINI: primary_model,
            ModelProvider.OPENROUTER: fallback_model_1,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    fallback_model_2 = OpenRouterTextModel(
        api_key="fake-key", model_name="anthropic/claude-3.5-sonnet"
    )
    router_2 = ModelRouter(
        providers={
            ModelProvider.GEMINI: primary_model,
            ModelProvider.OPENROUTER: fallback_model_2,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    handler_1 = ResearchStageHandler(agent=ResearchAgent(model=router_1))
    handler_2 = ResearchStageHandler(agent=ResearchAgent(model=router_2))

    payload = ResearchJobPayload(topic="Quantum Physics", model_name="gemini-2.5-pro")
    ctx = StageContext[ResearchJobPayload](
        workflow_id=uuid.uuid4(),
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    # Effective model identities must reflect their respective fallback configurations with provider prefixes
    id_1 = handler_1._canonical_model_identity(payload)
    id_2 = handler_2._canonical_model_identity(payload)
    assert id_1 == "gemini:gemini-2.5-pro:fallback=openrouter:google/gemini-2.5-flash"
    assert id_2 == "gemini:gemini-2.5-pro:fallback=openrouter:anthropic/claude-3.5-sonnet"
    assert id_1 != id_2

    fp_1 = handler_1.calculate_fingerprint(None, ctx)
    fp_2 = handler_2.calculate_fingerprint(None, ctx)

    # Fingerprints must change because the effective fallback execution configuration differs
    assert fp_1 != fp_2


@pytest.mark.asyncio
async def test_research_stage_rejects_unsupported_provider_prefix_in_override():
    """Verify that an unsupported provider prefix raises ModelConfigurationError during stage execution."""
    from app.models.errors import ModelConfigurationError
    from app.models.gemini import GeminiTextModel
    from app.models.openrouter import OpenRouterTextModel
    from app.models.router import ModelProvider, ModelRouter

    primary_model = GeminiTextModel(api_key="fake-key", model_name="gemini-2.5-flash")
    fallback_model = OpenRouterTextModel(api_key="fake-key", model_name="google/gemini-2.5-flash")
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: primary_model,
            ModelProvider.OPENROUTER: fallback_model,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )

    handler = ResearchStageHandler(agent=ResearchAgent(model=router))
    payload = ResearchJobPayload(topic="Quantum Physics", model_name="anthropic:claude-3-opus")
    ctx = StageContext[ResearchJobPayload](
        workflow_id=uuid.uuid4(),
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=payload,
    )

    with pytest.raises(ModelConfigurationError, match="Unsupported provider prefix 'anthropic'"):
        handler.calculate_fingerprint(None, ctx)

    with pytest.raises(ModelConfigurationError, match="Unsupported provider prefix 'anthropic'"):
        await handler.load_input(ctx)


@pytest.mark.asyncio
async def test_research_stage_rejects_malformed_prefixes_in_override():
    """Verify that empty provider prefix or empty model name raises ModelConfigurationError."""
    from app.models.errors import ModelConfigurationError
    from app.models.gemini import GeminiTextModel
    from app.models.openrouter import OpenRouterTextModel
    from app.models.router import ModelProvider, ModelRouter

    primary_model = GeminiTextModel(api_key="fake-key", model_name="gemini-2.5-flash")
    fallback_model = OpenRouterTextModel(api_key="fake-key", model_name="google/gemini-2.5-flash")
    router = ModelRouter(
        providers={
            ModelProvider.GEMINI: primary_model,
            ModelProvider.OPENROUTER: fallback_model,
        },
        primary=ModelProvider.GEMINI,
        fallback=ModelProvider.OPENROUTER,
    )
    handler = ResearchStageHandler(agent=ResearchAgent(model=router))

    ctx_empty_prefix = StageContext[ResearchJobPayload](
        workflow_id=uuid.uuid4(),
        job_id="job-1",
        attempt_id="att-1",
        worker_id="worker-1",
        lease_token="lease-1",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ResearchJobPayload(topic="Quantum Physics", model_name=":gemini-2.5-flash"),
    )
    with pytest.raises(ModelConfigurationError, match="Empty provider prefix"):
        handler.calculate_fingerprint(None, ctx_empty_prefix)

    ctx_empty_model = StageContext[ResearchJobPayload](
        workflow_id=uuid.uuid4(),
        job_id="job-2",
        attempt_id="att-2",
        worker_id="worker-1",
        lease_token="lease-2",
        attempt_number=1,
        settings=Settings(_env_file=None),
        payload=ResearchJobPayload(topic="Quantum Physics", model_name="gemini:"),
    )
    with pytest.raises(
        ModelConfigurationError, match="Empty model name after provider prefix 'gemini'"
    ):
        handler.calculate_fingerprint(None, ctx_empty_model)
