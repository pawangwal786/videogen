from uuid import uuid4

import pytest
from pydantic import ValidationError

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
    validate_scenes_against_research,
)


def create_sample_research_result(
    target_duration_seconds: int = 60,
) -> ResearchResult:
    """Helper creating a valid synthetic ResearchResult fixture."""
    return ResearchResult(
        workflow_id=uuid4(),
        topic="Synthetic Video Topic",
        target_duration_seconds=target_duration_seconds,
        executive_summary="A synthetic executive summary describing the core findings.",
        core_angle="A fresh synthetic angle for testing.",
        target_audience="General Audience",
        key_takeaways=["Takeaway one.", "Takeaway two."],
        facts=[
            ResearchFact(
                claim="First synthetic fact claim.",
                sources=[
                    ResearchSource(
                        title="Source 1",
                        url="https://example.invalid/source1",
                    )
                ],
                verification_status="source_backed",
            ),
            ResearchFact(claim="Second synthetic fact claim."),
            ResearchFact(claim="Third synthetic fact claim."),
        ],
        suggested_narrative_arc=[
            NarrativeBeat(
                sequence=1,
                beat_type="hook",
                description="Opening hook grabber.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=2,
                beat_type="problem",
                description="The core problem definition.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=3,
                beat_type="solution",
                description="The synthetic solution.",
                suggested_duration_ratio=0.25,
            ),
            NarrativeBeat(
                sequence=4,
                beat_type="call_to_action",
                description="Final call to action.",
                suggested_duration_ratio=0.25,
            ),
        ],
        visual_themes=["Theme Alpha", "Theme Beta"],
    )


def test_script_request_valid():
    research = create_sample_research_result(target_duration_seconds=60)
    req = ScriptRequest(
        research=research,
        pacing_wpm=150,
        tone="cinematic and bold",
        call_to_action="Visit our website to learn more.",
    )

    # Workflow ID must match research
    assert req.workflow_id == research.workflow_id
    # Target duration must be derived from research
    assert req.target_duration_seconds == 60
    assert req.pacing_wpm == 150
    assert req.tone == "cinematic and bold"
    assert req.call_to_action == "Visit our website to learn more."


def test_script_request_workflow_id_mismatch():
    research = create_sample_research_result()
    mismatched_id = uuid4()

    with pytest.raises(ValidationError) as exc_info:
        ScriptRequest(
            workflow_id=mismatched_id,
            research=research,
        )

    assert "must match research.workflow_id" in str(exc_info.value)


def test_script_scene_validation():
    # Valid scene
    scene = ScriptScene(
        scene_number=1,
        narrative_beat_sequence=1,
        beat_type="hook",
        narration="This is the opening narration for the scene.",
        visual_direction="Cinematic camera zoom into the main subject.",
        target_keywords=["zoom", "cinematic"],
        estimated_duration_seconds=3.5,
    )
    assert scene.scene_number == 1
    assert scene.beat_type == "hook"
    assert scene.estimated_duration_seconds == 3.5
    assert len(scene.target_keywords) == 2

    # Invalid keywords: fewer than 2
    with pytest.raises(ValidationError) as exc_info:
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Text",
            visual_direction="Cinematic camera visual.",
            target_keywords=["single_keyword"],
            estimated_duration_seconds=3.0,
        )
    assert "target_keywords" in str(exc_info.value)

    # Invalid keywords: more than 5
    with pytest.raises(ValidationError) as exc_info:
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Text",
            visual_direction="Cinematic camera visual.",
            target_keywords=["k1", "k2", "k3", "k4", "k5", "k6"],
            estimated_duration_seconds=3.0,
        )
    assert "target_keywords" in str(exc_info.value)

    # Invalid scene number (must be >= 1)
    with pytest.raises(ValidationError):
        ScriptScene(
            scene_number=0,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Text",
            visual_direction="Visuals",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=3.0,
        )

    # Visual direction too short
    with pytest.raises(ValidationError):
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Text",
            visual_direction="Shot",  # < 5 chars
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=3.0,
        )


def test_script_scene_references_existing_research_beat():
    research = create_sample_research_result()
    # 5 scenes covering all 4 research beats (beat 3 expanded by 2 scenes)
    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Opening line.",
            visual_direction="Opening camera visual.",
            target_keywords=["hook", "opening"],
            estimated_duration_seconds=2.0,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration="Problem line.",
            visual_direction="Problem camera visual.",
            target_keywords=["problem", "conflict"],
            estimated_duration_seconds=2.0,
        ),
        ScriptScene(
            scene_number=3,
            narrative_beat_sequence=3,
            beat_type="solution",
            narration="Solution part one.",
            visual_direction="Solution camera visual part one.",
            target_keywords=["solution", "breakthrough"],
            estimated_duration_seconds=2.0,
        ),
        ScriptScene(
            scene_number=4,
            narrative_beat_sequence=3,
            beat_type="solution",
            narration="Solution part two.",
            visual_direction="Solution camera visual part two.",
            target_keywords=["architecture", "efficiency"],
            estimated_duration_seconds=2.0,
        ),
        ScriptScene(
            scene_number=5,
            narrative_beat_sequence=4,
            beat_type="call_to_action",
            narration="Call to action line.",
            visual_direction="CTA visual card.",
            target_keywords=["cta", "subscribe"],
            estimated_duration_seconds=2.0,
        ),
    ]

    # Must succeed without error
    validate_scenes_against_research(scenes, research)


def test_script_scene_rejects_missing_research_beat():
    research = create_sample_research_result()
    # Missing beat 4 (call_to_action)
    incomplete_scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Opening line.",
            visual_direction="Opening camera visual.",
            target_keywords=["hook", "opening"],
            estimated_duration_seconds=2.0,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration="Problem line.",
            visual_direction="Problem camera visual.",
            target_keywords=["problem", "conflict"],
            estimated_duration_seconds=2.0,
        ),
        ScriptScene(
            scene_number=3,
            narrative_beat_sequence=3,
            beat_type="solution",
            narration="Solution line.",
            visual_direction="Solution camera visual.",
            target_keywords=["solution", "breakthrough"],
            estimated_duration_seconds=2.0,
        ),
    ]

    with pytest.raises(ValueError) as exc_info:
        validate_scenes_against_research(incomplete_scenes, research)
    assert "Script scenes must cover all research narrative beats" in str(exc_info.value)
    assert "Missing narrative beat sequence(s): [4]" in str(exc_info.value)


def test_script_scene_rejects_unknown_research_beat():
    research = create_sample_research_result()

    # Unknown narrative beat sequence 99
    invalid_scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=99,
            beat_type="hook",
            narration="Line.",
            visual_direction="Visual.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=2.0,
        )
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_scenes_against_research(invalid_scenes, research)
    assert "references unknown narrative beat sequence 99" in str(exc_info.value)

    # Mismatched beat type (sequence 1 is 'hook', not 'climax')
    type_mismatched_scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="climax",
            narration="Line.",
            visual_direction="Visual.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=2.0,
        )
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_scenes_against_research(type_mismatched_scenes, research)
    assert "beat_type 'climax' does not match referenced research beat 1" in str(exc_info.value)


def test_script_result_scene_sequence():
    wf_id = uuid4()
    # Non-sequential scene numbers [1, 3]
    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Word " * 70,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
        ScriptScene(
            scene_number=3,  # Skipping 2
            narrative_beat_sequence=2,
            beat_type="problem",
            narration="Word " * 70,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
    ]

    with pytest.raises(ValidationError) as exc_info:
        ScriptResult(
            workflow_id=wf_id,
            title="Synthetic Video Title",
            target_duration_seconds=60,
            estimated_duration_seconds=60.0,
            total_word_count=140,
            scenes=scenes,
            full_narration=" ".join(s.narration.strip() for s in scenes),
            pacing_wpm=140,
        )
    assert "Script scenes must be strictly sequential starting from 1" in str(exc_info.value)


def test_script_result_full_narration_is_derived():
    wf_id = uuid4()
    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Word " * 70,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration="Word " * 70,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
    ]

    # Tampered full_narration
    with pytest.raises(ValidationError) as exc_info:
        ScriptResult(
            workflow_id=wf_id,
            title="Synthetic Video Title",
            target_duration_seconds=60,
            estimated_duration_seconds=60.0,
            total_word_count=140,
            scenes=scenes,
            full_narration="A completely different narration text.",
            pacing_wpm=140,
        )
    assert "full_narration must match the concatenated scene narrations" in str(exc_info.value)


def test_script_result_word_count_is_derived():
    wf_id = uuid4()
    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration="Word " * 70,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration="Word " * 70,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
    ]

    # Tampered total_word_count (says 500 when actual is 140)
    with pytest.raises(ValidationError) as exc_info:
        ScriptResult(
            workflow_id=wf_id,
            title="Synthetic Video Title",
            target_duration_seconds=60,
            estimated_duration_seconds=60.0,
            total_word_count=500,
            scenes=scenes,
            full_narration=" ".join(s.narration.strip() for s in scenes),
            pacing_wpm=140,
        )
    assert "total_word_count (500) must match sum of scene word counts (140)" in str(exc_info.value)


def test_script_result_duration_calculation():
    wf_id = uuid4()
    # 140 words at 140 WPM = exactly 60.0 seconds
    narration_s1 = " ".join(["test"] * 70)
    narration_s2 = " ".join(["test"] * 70)
    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration=narration_s1,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration=narration_s2,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=30.0,
        ),
    ]
    full_narration = " ".join(s.narration.strip() for s in scenes)

    # 1. Valid duration calculation (60.0s for target 60s)
    result = ScriptResult(
        workflow_id=wf_id,
        title="Synthetic Video Title",
        target_duration_seconds=60,
        estimated_duration_seconds=60.0,
        total_word_count=140,
        scenes=scenes,
        full_narration=full_narration,
        pacing_wpm=140,
    )
    assert result.estimated_duration_seconds == 60.0

    # 2. Duration outside ±15% tolerance: 60s target allows [51.0s, 69.0s]
    # 200 words at 140 WPM = 85.71s (> 69s)
    narration_long = " ".join(["test"] * 100)
    long_scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration=narration_long,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=42.86,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration=narration_long,
            visual_direction="Visual directions.",
            target_keywords=["k1", "k2"],
            estimated_duration_seconds=42.86,
        ),
    ]
    with pytest.raises(ValidationError) as exc_info:
        ScriptResult(
            workflow_id=wf_id,
            title="Too Long Script",
            target_duration_seconds=60,
            estimated_duration_seconds=85.71,
            total_word_count=200,
            scenes=long_scenes,
            full_narration=" ".join(s.narration.strip() for s in long_scenes),
            pacing_wpm=140,
        )
    assert "outside target_duration_seconds ±15% tolerance" in str(exc_info.value)
