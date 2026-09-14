from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.research.models import (
    NarrativeBeat,
    ResearchFact,
    ResearchRequest,
    ResearchResult,
    ResearchSource,
)


def test_research_request_valid():
    wf_id = uuid4()
    req = ResearchRequest(
        workflow_id=wf_id,
        topic="Synthetic Topic Title",
        target_audience="Tech enthusiasts",
        target_duration_seconds=90,
        research_depth="deep",
        style_guidelines=["Fast-paced", "Visual analogies"],
        key_aspects_to_cover=["Architecture", "Protocols", "Current limitations"],
    )

    assert req.workflow_id == wf_id
    assert req.topic == "Synthetic Topic Title"
    assert req.research_depth == "deep"
    assert req.target_duration_seconds == 90
    assert len(req.style_guidelines) == 2


def test_research_request_topic_too_short():
    with pytest.raises(ValidationError) as exc_info:
        ResearchRequest(
            workflow_id=uuid4(),
            topic="AI",  # Min length 3
        )
    assert "topic" in str(exc_info.value)


def test_research_request_duration_bounds():
    with pytest.raises(ValidationError):
        ResearchRequest(
            workflow_id=uuid4(),
            topic="Valid topic",
            target_duration_seconds=10,  # Minimum 15
        )

    with pytest.raises(ValidationError):
        ResearchRequest(
            workflow_id=uuid4(),
            topic="Valid topic",
            target_duration_seconds=1000,  # Maximum 600
        )


def test_research_source_valid_and_invalid_url():
    # Synthetic URL; never used for live research.
    source = ResearchSource(
        title="Synthetic Research Paper",
        url="https://example.invalid/paper-001",
        publisher="Example Journal",
    )
    assert source.title == "Synthetic Research Paper"
    assert str(source.url) == "https://example.invalid/paper-001"
    assert source.published_at is None


def test_research_source_rejects_invalid_url():
    with pytest.raises(ValidationError) as exc_info:
        ResearchSource(
            title="Invalid URL Source",
            url="not-a-valid-http-url",
        )
    assert "url" in str(exc_info.value)


def test_source_backed_fact_requires_source():
    # Attempting source_backed with empty sources must fail
    with pytest.raises(ValidationError) as exc_info:
        ResearchFact(
            claim="A factual assertion claiming backing without sources.",
            sources=[],
            verification_status="source_backed",
        )
    assert "source_backed" in str(exc_info.value)

    # Adding a valid source allows source_backed to pass
    source = ResearchSource(
        title="Synthetic Benchmark",
        url="https://example.invalid/benchmark",
    )
    fact = ResearchFact(
        claim="A factual assertion with a verifiable source.",
        sources=[source],
        verification_status="source_backed",
    )
    assert fact.verification_status == "source_backed"
    assert len(fact.sources) == 1


def test_research_fact_defaults():
    fact_default = ResearchFact(claim="An unverified assertion about technology.")
    assert fact_default.verification_status == "needs_verification"
    assert fact_default.sources == []


def test_research_result_valid_and_serializable():
    wf_id = uuid4()
    # Synthetic test fixture; not live research.
    result = ResearchResult(
        workflow_id=wf_id,
        topic="Synthetic Topic Investigation",
        research_depth="standard",
        executive_summary="Executive summary describing the synthetic topic findings.",
        core_angle="Why the synthetic architecture is fundamentally changing paradigms.",
        target_audience="General technical audience",
        key_takeaways=[
            "First synthetic takeaway point.",
            "Second synthetic takeaway point.",
            "Third synthetic takeaway point.",
        ],
        facts=[
            ResearchFact(
                claim="First synthetic verified benchmark claim.",
                sources=[
                    ResearchSource(
                        title="Synthetic Source 001",
                        url="https://example.invalid/source-001",
                    )
                ],
                verification_status="source_backed",
            ),
            ResearchFact(
                claim="Second synthetic claim requiring external verification.",
                verification_status="needs_verification",
            ),
            ResearchFact(
                claim="Third synthetic claim requiring external verification.",
                verification_status="needs_verification",
            ),
        ],
        suggested_narrative_arc=[
            NarrativeBeat(
                sequence=1,
                beat_type="hook",
                description="Opening hook explaining the core question.",
                suggested_duration_ratio=0.20,
            ),
            NarrativeBeat(
                sequence=2,
                beat_type="explanation",
                description="Detailed explanation of underlying concepts.",
                suggested_duration_ratio=0.35,
            ),
            NarrativeBeat(
                sequence=3,
                beat_type="challenge",
                description="Key challenges and trade-offs.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=4,
                beat_type="conclusion",
                description="Summary and takeaway implications.",
                suggested_duration_ratio=0.20,
            ),
        ],
        visual_themes=[
            "Theme A: Minimalist geometric animation",
            "Theme B: High-contrast technical diagrams",
        ],
    )

    assert result.workflow_id == wf_id
    assert len(result.suggested_narrative_arc) == 4

    # Verify JSON serialization and roundtrip
    json_data = result.model_dump_json(indent=2)
    assert "Synthetic Topic Investigation" in json_data

    restored = ResearchResult.model_validate_json(json_data)
    assert restored.workflow_id == wf_id
    assert restored.topic == result.topic
    assert len(restored.facts) == 3


def test_duration_ratio_boundary_0_98():
    # Exactly 0.98 total (boundary minimum)
    result = ResearchResult(
        workflow_id=uuid4(),
        topic="Boundary Min Topic",
        executive_summary="A brief summary for testing validation.",
        core_angle="A testing angle.",
        target_audience="Testing audience",
        key_takeaways=["Point 1", "Point 2"],
        facts=[
            ResearchFact(claim="Fact 1"),
            ResearchFact(claim="Fact 2"),
            ResearchFact(claim="Fact 3"),
        ],
        suggested_narrative_arc=[
            NarrativeBeat(
                sequence=1, beat_type="b1", description="desc 1", suggested_duration_ratio=0.33
            ),
            NarrativeBeat(
                sequence=2, beat_type="b2", description="desc 2", suggested_duration_ratio=0.33
            ),
            NarrativeBeat(
                sequence=3, beat_type="b3", description="desc 3", suggested_duration_ratio=0.32
            ),
        ],
        visual_themes=["Theme 1", "Theme 2"],
    )
    assert sum(b.suggested_duration_ratio for b in result.suggested_narrative_arc) == pytest.approx(
        0.98
    )


def test_duration_ratio_boundary_1_02():
    # Exactly 1.02 total (boundary maximum)
    result = ResearchResult(
        workflow_id=uuid4(),
        topic="Boundary Max Topic",
        executive_summary="A brief summary for testing validation.",
        core_angle="A testing angle.",
        target_audience="Testing audience",
        key_takeaways=["Point 1", "Point 2"],
        facts=[
            ResearchFact(claim="Fact 1"),
            ResearchFact(claim="Fact 2"),
            ResearchFact(claim="Fact 3"),
        ],
        suggested_narrative_arc=[
            NarrativeBeat(
                sequence=1, beat_type="b1", description="desc 1", suggested_duration_ratio=0.34
            ),
            NarrativeBeat(
                sequence=2, beat_type="b2", description="desc 2", suggested_duration_ratio=0.34
            ),
            NarrativeBeat(
                sequence=3, beat_type="b3", description="desc 3", suggested_duration_ratio=0.34
            ),
        ],
        visual_themes=["Theme 1", "Theme 2"],
    )
    assert sum(b.suggested_duration_ratio for b in result.suggested_narrative_arc) == pytest.approx(
        1.02
    )


def test_narrative_beat_duration_ratio_sum_validation():
    # Sum is 0.50 (well below 0.98)
    with pytest.raises(ValidationError) as exc_info:
        ResearchResult(
            workflow_id=uuid4(),
            topic="Test Topic",
            executive_summary="A brief summary for testing validation.",
            core_angle="A testing angle.",
            target_audience="Testing audience",
            key_takeaways=["Point 1", "Point 2"],
            facts=[
                ResearchFact(claim="Fact 1"),
                ResearchFact(claim="Fact 2"),
                ResearchFact(claim="Fact 3"),
            ],
            suggested_narrative_arc=[
                NarrativeBeat(
                    sequence=1, beat_type="b1", description="desc 1", suggested_duration_ratio=0.25
                ),
                NarrativeBeat(
                    sequence=2, beat_type="b2", description="desc 2", suggested_duration_ratio=0.25
                ),
                NarrativeBeat(
                    sequence=3, beat_type="b3", description="desc 3", suggested_duration_ratio=0.10
                ),
            ],
            visual_themes=["Theme 1", "Theme 2"],
        )
    assert "Narrative beat duration ratios must sum to approximately 1.0" in str(exc_info.value)


def test_narrative_beat_sequence_ordering_validation():
    # Sequences out of order: [1, 3, 2]
    with pytest.raises(ValidationError) as exc_info:
        ResearchResult(
            workflow_id=uuid4(),
            topic="Test Topic",
            executive_summary="A brief summary for testing validation.",
            core_angle="A testing angle.",
            target_audience="Testing audience",
            key_takeaways=["Point 1", "Point 2"],
            facts=[
                ResearchFact(claim="Fact 1"),
                ResearchFact(claim="Fact 2"),
                ResearchFact(claim="Fact 3"),
            ],
            suggested_narrative_arc=[
                NarrativeBeat(
                    sequence=1, beat_type="b1", description="desc 1", suggested_duration_ratio=0.34
                ),
                NarrativeBeat(
                    sequence=3, beat_type="b2", description="desc 2", suggested_duration_ratio=0.33
                ),
                NarrativeBeat(
                    sequence=2, beat_type="b3", description="desc 3", suggested_duration_ratio=0.33
                ),
            ],
            visual_themes=["Theme 1", "Theme 2"],
        )
    assert "Narrative beat sequences must be strictly sequential" in str(exc_info.value)


def test_research_result_minimum_items_constraints():
    # Facts fewer than 3
    with pytest.raises(ValidationError) as exc_info:
        ResearchResult(
            workflow_id=uuid4(),
            topic="Test Topic",
            executive_summary="A brief summary for testing validation.",
            core_angle="A testing angle.",
            target_audience="Testing audience",
            key_takeaways=["Point 1", "Point 2"],
            facts=[ResearchFact(claim="Only one fact")],
            suggested_narrative_arc=[
                NarrativeBeat(
                    sequence=1, beat_type="b1", description="desc 1", suggested_duration_ratio=0.5
                ),
                NarrativeBeat(
                    sequence=2, beat_type="b2", description="desc 2", suggested_duration_ratio=0.5
                ),
            ],
            visual_themes=["Theme 1", "Theme 2"],
        )
    assert "facts" in str(exc_info.value)
