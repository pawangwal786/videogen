import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.research.agent import ResearchAgent, extract_json_payload
from app.agents.research.errors import (
    ResearchResponseError,
    ResearchValidationError,
)
from app.agents.research.models import ResearchRequest, ResearchResult
from app.models.errors import ModelTimeoutError


@pytest.fixture
def mock_text_model():
    model = AsyncMock()
    model.generate = AsyncMock()
    return model


@pytest.fixture
def valid_research_payload():
    # Synthetic test fixture; not live research.
    return {
        "topic": "Synthetic Hardware Architecture",
        "research_depth": "standard",
        "research_method": "llm_synthesis",
        "executive_summary": "Synthetic hardware architecture demonstrates event-driven low-power computation paradigms.",
        "core_angle": "How asynchronous event routing overcomes traditional synchronization bottlenecks.",
        "target_audience": "Technical audience and systems engineers",
        "key_takeaways": [
            "Takeaway 1: Event-driven routing scales sub-linearly with clock distribution.",
            "Takeaway 2: Dynamic energy scaling minimizes quiescent leakage.",
            "Takeaway 3: Silicon area efficiency improves on high-density process nodes.",
        ],
        "facts": [
            {
                "claim": "Synthetic core variant achieves 10x lower switching energy in laboratory benchmarks.",
                "context": "Fabricated on synthetic test node.",
                "sources": [
                    {
                        "title": "Synthetic Technical Benchmark 001",
                        "url": "https://example.invalid/benchmark-001",
                        "publisher": "Synthetic Institute",
                    }
                ],
                "verification_status": "source_backed",
            },
            {
                "claim": "The reference architecture operates within a 25 watt power envelope.",
                "context": "Measurement observed during peak synthetic workload simulation.",
                "sources": [],
                "verification_status": "needs_verification",
            },
            {
                "claim": "Event dispatchers route discrete packet bursts rather than continuous signals.",
                "context": "Architectural principle for sparse computational pipelines.",
                "sources": [],
                "verification_status": "needs_verification",
            },
        ],
        "suggested_narrative_arc": [
            {
                "sequence": 1,
                "beat_type": "hook",
                "description": "Demonstrate the scaling limits of continuous clock networks.",
                "suggested_duration_ratio": 0.20,
            },
            {
                "sequence": 2,
                "beat_type": "explanation",
                "description": "Explain asynchronous event-driven switching mechanics.",
                "suggested_duration_ratio": 0.35,
            },
            {
                "sequence": 3,
                "beat_type": "breakthroughs",
                "description": "Show experimental benchmark curves and trade-offs.",
                "suggested_duration_ratio": 0.30,
            },
            {
                "sequence": 4,
                "beat_type": "conclusion",
                "description": "Key takeaways and next-generation roadmaps.",
                "suggested_duration_ratio": 0.15,
            },
        ],
        "visual_themes": [
            "Theme A: Sparse pulse waves traversing dark circuit topologies",
            "Theme B: High-contrast telemetry graphs and timing diagrams",
        ],
    }


def test_extract_json_payload_clean():
    data = {"key": "value", "count": 42}
    raw = json.dumps(data)
    assert extract_json_payload(raw) == data


def test_extract_json_payload_markdown_fenced():
    data = {"key": "fenced_value"}
    raw = f"Here is the research output:\n```json\n{json.dumps(data)}\n```\nHope this helps!"
    assert extract_json_payload(raw) == data


def test_extract_json_payload_invalid_syntax():
    with pytest.raises(ResearchResponseError) as exc_info:
        extract_json_payload("Not JSON content at all")
    assert "Failed to decode JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_research_agent_run_success(mock_text_model, valid_research_payload):
    mock_text_model.generate.return_value = json.dumps(valid_research_payload)
    agent = ResearchAgent(model=mock_text_model)

    wf_id = uuid4()
    request = ResearchRequest(
        workflow_id=wf_id,
        topic="Synthetic Hardware Architecture",
        target_audience="Technical audience",
        target_duration_seconds=90,
        research_depth="standard",
    )

    result = await agent.run(request)

    assert isinstance(result, ResearchResult)
    assert result.workflow_id == wf_id
    assert result.topic == "Synthetic Hardware Architecture"
    assert len(result.facts) == 3
    assert len(result.suggested_narrative_arc) == 4
    mock_text_model.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_research_agent_preserves_workflow_id(mock_text_model, valid_research_payload):
    # Even if LLM returns a different or missing workflow_id, agent injects request.workflow_id
    payload = dict(valid_research_payload)
    payload["workflow_id"] = str(uuid4())  # Discrepant ID

    mock_text_model.generate.return_value = json.dumps(payload)
    agent = ResearchAgent(model=mock_text_model)

    expected_wf_id = uuid4()
    request = ResearchRequest(
        workflow_id=expected_wf_id,
        topic="Preserve ID Topic",
    )

    result = await agent.run(request)
    assert result.workflow_id == expected_wf_id


@pytest.mark.asyncio
async def test_research_agent_invalid_json_raises_response_error(mock_text_model):
    mock_text_model.generate.return_value = "Sorry, I am unable to analyze this topic."
    agent = ResearchAgent(model=mock_text_model)

    request = ResearchRequest(
        workflow_id=uuid4(),
        topic="Any Topic",
    )

    with pytest.raises(ResearchResponseError) as exc_info:
        await agent.run(request)
    assert "Failed to decode JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_research_agent_schema_violation_raises_validation_error(
    mock_text_model, valid_research_payload
):
    # Corrupt payload: narrative beats ratios sum to 2.0
    bad_payload = dict(valid_research_payload)
    bad_payload["suggested_narrative_arc"][0]["suggested_duration_ratio"] = 1.0
    bad_payload["suggested_narrative_arc"][1]["suggested_duration_ratio"] = 1.0

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = ResearchAgent(model=mock_text_model)

    request = ResearchRequest(
        workflow_id=uuid4(),
        topic="Synthetic Hardware Architecture",
    )

    with pytest.raises(ResearchValidationError) as exc_info:
        await agent.run(request)
    assert "ResearchResult schema validation failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_research_agent_model_error_propagates(mock_text_model):
    mock_text_model.generate.side_effect = ModelTimeoutError(
        "Model call timed out", provider="gemini"
    )
    agent = ResearchAgent(model=mock_text_model)

    request = ResearchRequest(
        workflow_id=uuid4(),
        topic="Synthetic Hardware Architecture",
    )

    with pytest.raises(ModelTimeoutError):
        await agent.run(request)


@pytest.mark.asyncio
async def test_research_agent_custom_system_prompt(mock_text_model, valid_research_payload):
    mock_text_model.generate.return_value = json.dumps(valid_research_payload)
    custom_prompt = "You are a specialized biology research agent."
    agent = ResearchAgent(model=mock_text_model, system_prompt=custom_prompt)

    request = ResearchRequest(
        workflow_id=uuid4(),
        topic="Synthetic Hardware Architecture",
    )

    await agent.run(request)
    assert mock_text_model.generate.await_args.kwargs["system_prompt"] == custom_prompt


def test_build_research_user_prompt_with_optional_guidelines_and_aspects():
    from app.agents.research.prompts import build_research_user_prompt

    request = ResearchRequest(
        workflow_id=uuid4(),
        topic="Quantum Computing",
        target_audience="Engineers",
        target_duration_seconds=60,
        research_depth="deep",
        style_guidelines=["concise and rigorous", "use SI units"],
        key_aspects_to_cover=["qubit coherence", "error correction"],
    )
    prompt = build_research_user_prompt(request)
    assert "Topic: Quantum Computing" in prompt
    assert "Style Guidelines:" in prompt
    assert "- concise and rigorous" in prompt
    assert "- use SI units" in prompt
    assert "Key Aspects to Cover:" in prompt
    assert "- qubit coherence" in prompt
    assert "- error correction" in prompt


@pytest.mark.asyncio
async def test_base_agent_abstract_run_raises_not_implemented():
    from app.agents.base import Agent

    class ConcreteAgent(Agent[str, str]):
        name = "test_agent"

        async def run(self, input: str) -> str:
            return await super().run(input)

    agent = ConcreteAgent()
    with pytest.raises(NotImplementedError):
        await agent.run("test_input")
