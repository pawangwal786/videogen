import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.research.models import (
    NarrativeBeat,
    ResearchFact,
    ResearchResult,
    ResearchSource,
)
from app.agents.script.agent import ScriptAgent, extract_json_payload
from app.agents.script.errors import (
    ScriptResponseError,
    ScriptValidationError,
)
from app.agents.script.models import ScriptRequest, ScriptResult
from app.models.errors import ModelTimeoutError


@pytest.fixture
def mock_text_model():
    model = AsyncMock()
    model.generate = AsyncMock()
    return model


@pytest.fixture
def sample_research():
    return ResearchResult(
        workflow_id=uuid4(),
        topic="Synthetic Scalable Computing",
        target_duration_seconds=60,
        executive_summary="Synthetic architecture for low-latency asynchronous message dispatch.",
        core_angle="Eliminating synchronous barriers in parallel architectures.",
        target_audience="Software Engineers",
        key_takeaways=[
            "Takeaway 1: Eliminates global synchronization locks.",
            "Takeaway 2: Sub-millisecond queue latency.",
        ],
        facts=[
            ResearchFact(
                claim="Throughput reaches 5 million operations per second on laboratory nodes.",
                sources=[
                    ResearchSource(
                        title="Benchmark Report",
                        url="https://example.invalid/benchmark-01",
                    )
                ],
                verification_status="source_backed",
            ),
            ResearchFact(claim="Memory footprint scales linearly with active queues."),
            ResearchFact(claim="Deterministic latency profiles under peak burst load."),
        ],
        suggested_narrative_arc=[
            NarrativeBeat(
                sequence=1,
                beat_type="hook",
                description="Hook showing traditional lock contention bottlenecks.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=2,
                beat_type="problem",
                description="The core problem of synchronized clock hierarchies.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=3,
                beat_type="solution",
                description="How asynchronous message routing solves contention.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=4,
                beat_type="call_to_action",
                description="Explore the open benchmark suite and build your first pipeline.",
                suggested_duration_ratio=0.25,
            ),
        ],
        visual_themes=[
            "Theme A: Heat maps of lock contention on silicon dies",
            "Theme B: Clean signal routing through distributed topologies",
        ],
    )


@pytest.fixture
def valid_script_payload():
    # 4 scenes with exactly 35 words each = 140 words total
    # At 140 WPM, 140 words = exactly 60.0 seconds
    narration_35_words = " ".join(["word"] * 35)
    return {
        "title": "Unlocking Ultra-Fast Asynchronous Architectures",
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat_sequence": 1,
                "beat_type": "hook",
                "narration": narration_35_words,
                "visual_direction": "Slow pan over congested processor dies with pulsing red hotspots.",
                "target_keywords": ["cpu die", "congestion", "hardware bottleneck"],
            },
            {
                "scene_number": 2,
                "narrative_beat_sequence": 2,
                "beat_type": "problem",
                "narration": narration_35_words,
                "visual_direction": "Split screen diagram comparing lock contention versus queuing latency.",
                "target_keywords": ["queuing theory", "latency graphs"],
            },
            {
                "scene_number": 3,
                "narrative_beat_sequence": 3,
                "beat_type": "solution",
                "narration": narration_35_words,
                "visual_direction": "Fluorescent signal pulses streaming effortlessly across asynchronous mesh networks.",
                "target_keywords": ["mesh network", "signal propagation"],
            },
            {
                "scene_number": 4,
                "narrative_beat_sequence": 4,
                "beat_type": "call_to_action",
                "narration": narration_35_words,
                "visual_direction": "High-contrast title card displaying repository benchmark links.",
                "target_keywords": ["open source", "benchmarks"],
            },
        ],
    }


def test_extract_json_payload_clean():
    data = {"title": "Test Title", "scenes": []}
    assert extract_json_payload(json.dumps(data)) == data


def test_extract_json_payload_markdown_fenced():
    data = {"title": "Test Title", "scenes": []}
    raw = f"```json\n{json.dumps(data)}\n```"
    assert extract_json_payload(raw) == data


def test_extract_json_payload_invalid_syntax():
    with pytest.raises(ScriptResponseError) as exc_info:
        extract_json_payload("Plain text not containing any valid JSON.")
    assert "Failed to decode JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_success(mock_text_model, sample_research, valid_script_payload):
    mock_text_model.generate.return_value = json.dumps(valid_script_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(
        research=sample_research,
        pacing_wpm=140,
        tone="cinematic and technical",
    )

    result = await agent.run(request)

    assert isinstance(result, ScriptResult)
    assert result.workflow_id == sample_research.workflow_id
    assert result.title == "Unlocking Ultra-Fast Asynchronous Architectures"
    assert result.target_duration_seconds == 60
    assert result.total_word_count == 140
    assert result.estimated_duration_seconds == 60.0
    assert len(result.scenes) == 4

    # Validate scene durations were derived in application code
    for scene in result.scenes:
        assert scene.estimated_duration_seconds == 15.0

    # Validate full_narration was derived
    expected_full = " ".join(s.narration.strip() for s in result.scenes)
    assert result.full_narration == expected_full

    mock_text_model.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_script_agent_preserves_workflow_id(
    mock_text_model, sample_research, valid_script_payload
):
    # Even if LLM response somehow included a workflow_id field
    payload = dict(valid_script_payload)
    payload["workflow_id"] = str(uuid4())

    mock_text_model.generate.return_value = json.dumps(payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    result = await agent.run(request)

    assert result.workflow_id == sample_research.workflow_id


@pytest.mark.asyncio
async def test_script_agent_invalid_json(mock_text_model, sample_research):
    mock_text_model.generate.return_value = "Sorry, I encountered an internal error."
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptResponseError) as exc_info:
        await agent.run(request)
    assert "Failed to decode JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_schema_violation(mock_text_model, sample_research):
    # Missing 'scenes' list entirely
    bad_payload = {"title": "A title without scenes"}
    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptValidationError) as exc_info:
        await agent.run(request)
    assert "non-empty 'scenes' list" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_lineage_unknown_beat_raises_validation_error(
    mock_text_model, sample_research, valid_script_payload
):
    # Scene 1 references non-existent narrative beat 99
    bad_payload = dict(valid_script_payload)
    bad_payload["scenes"][0]["narrative_beat_sequence"] = 99

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptValidationError) as exc_info:
        await agent.run(request)
    assert "references unknown narrative beat sequence 99" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_lineage_mismatched_beat_type_raises_validation_error(
    mock_text_model, sample_research, valid_script_payload
):
    # Beat 1 in sample_research is 'hook', scene claims 'climax'
    bad_payload = dict(valid_script_payload)
    bad_payload["scenes"][0]["beat_type"] = "climax"

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptValidationError) as exc_info:
        await agent.run(request)
    assert "does not match referenced research beat 1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_lineage_missing_beat_raises_validation_error(
    mock_text_model, sample_research, valid_script_payload
):
    # Omit scene 4 (beat 4 call_to_action)
    bad_payload = dict(valid_script_payload)
    bad_payload["scenes"] = [s for s in valid_script_payload["scenes"] if s["narrative_beat_sequence"] != 4]

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptValidationError) as exc_info:
        await agent.run(request)
    assert "Script scenes must cover all research narrative beats" in str(exc_info.value)
    assert "Missing narrative beat sequence(s): [4]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_invalid_keywords_raises_validation_error(
    mock_text_model, sample_research, valid_script_payload
):
    # Scene 1 has only 1 keyword (< 2 required)
    bad_payload = dict(valid_script_payload)
    bad_payload["scenes"][0]["target_keywords"] = ["only_one"]

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptValidationError) as exc_info:
        await agent.run(request)
    assert "target_keywords" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_duration_out_of_tolerance_raises_validation_error(
    mock_text_model, sample_research, valid_script_payload
):
    # 70 words per scene * 4 = 280 words -> at 140 WPM = 120s (target is 60s, ±15% is [51s, 69s])
    bad_payload = dict(valid_script_payload)
    for scene in bad_payload["scenes"]:
        scene["narration"] = " ".join(["word"] * 70)

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ScriptValidationError) as exc_info:
        await agent.run(request)
    assert "outside target_duration_seconds ±15% tolerance" in str(exc_info.value)


@pytest.mark.asyncio
async def test_script_agent_model_error(mock_text_model, sample_research):
    mock_text_model.generate.side_effect = ModelTimeoutError(
        "OpenRouter request timed out", provider="openrouter"
    )
    agent = ScriptAgent(model=mock_text_model)

    request = ScriptRequest(research=sample_research)
    with pytest.raises(ModelTimeoutError):
        await agent.run(request)


@pytest.mark.asyncio
async def test_script_agent_custom_system_prompt(
    mock_text_model, sample_research, valid_script_payload
):
    mock_text_model.generate.return_value = json.dumps(valid_script_payload)
    custom_prompt = "You are a comedic scriptwriter."
    agent = ScriptAgent(model=mock_text_model, system_prompt=custom_prompt)

    request = ScriptRequest(research=sample_research)
    await agent.run(request)

    assert mock_text_model.generate.await_args.kwargs["system_prompt"] == custom_prompt
