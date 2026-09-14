from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.agents.script.models import ScriptResult, ScriptScene
from app.agents.storyboard.models import (
    DURATION_TOLERANCE_RATIO,
    SCENE_DURATION_TOLERANCE_RATIO,
    StoryboardRequest,
    StoryboardResult,
    StoryboardShot,
    validate_shots_against_script,
)


def create_sample_script_result(target_duration_seconds: int = 60) -> ScriptResult:
    """Helper creating a valid synthetic ScriptResult fixture with 4 scenes."""
    wf_id = uuid4()
    # 4 scenes with 35 words each = 140 words total -> at 140 WPM = exactly 60.0s
    narration_35 = " ".join(["word"] * 35)
    scenes = [
        ScriptScene(
            scene_number=1,
            narrative_beat_sequence=1,
            beat_type="hook",
            narration=narration_35,
            visual_direction="Visual description for hook scene.",
            target_keywords=["keyword_1", "keyword_2"],
            estimated_duration_seconds=15.0,
        ),
        ScriptScene(
            scene_number=2,
            narrative_beat_sequence=2,
            beat_type="problem",
            narration=narration_35,
            visual_direction="Visual description for problem scene.",
            target_keywords=["keyword_1", "keyword_2"],
            estimated_duration_seconds=15.0,
        ),
        ScriptScene(
            scene_number=3,
            narrative_beat_sequence=3,
            beat_type="solution",
            narration=narration_35,
            visual_direction="Visual description for solution scene.",
            target_keywords=["keyword_1", "keyword_2"],
            estimated_duration_seconds=15.0,
        ),
        ScriptScene(
            scene_number=4,
            narrative_beat_sequence=4,
            beat_type="call_to_action",
            narration=narration_35,
            visual_direction="Visual description for CTA scene.",
            target_keywords=["keyword_1", "keyword_2"],
            estimated_duration_seconds=15.0,
        ),
    ]
    return ScriptResult(
        workflow_id=wf_id,
        title="Synthetic Video Title",
        target_duration_seconds=target_duration_seconds,
        estimated_duration_seconds=60.0,
        total_word_count=140,
        scenes=scenes,
        full_narration=" ".join(s.narration.strip() for s in scenes),
        pacing_wpm=140,
    )


def test_storyboard_shot_validation():
    # Valid shot
    shot = StoryboardShot(
        shot_number=1,
        scene_number=1,
        shot_framing="wide_shot",
        camera_movement="tracking",
        visual_description="Cinematic wide establishing shot of a server rack.",
        video_prompt="4k cinematic footage of glowing server racks in dark datacenter, steady tracking camera.",
        negative_prompt="blurry, watermark, low quality",
        estimated_duration_seconds=7.5,
    )
    assert shot.shot_number == 1
    assert shot.scene_number == 1
    assert shot.shot_framing == "wide_shot"
    assert shot.camera_movement == "tracking"
    assert shot.estimated_duration_seconds == 7.5

    # Invalid shot_number (< 1)
    with pytest.raises(ValidationError):
        StoryboardShot(
            shot_number=0,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="tracking",
            visual_description="Valid visual description.",
            video_prompt="Valid video prompt text.",
            estimated_duration_seconds=5.0,
        )

    # Invalid shot_framing
    with pytest.raises(ValidationError):
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="bird_eye_angle",  # Not in ShotFraming Literal
            camera_movement="tracking",
            visual_description="Valid visual description.",
            video_prompt="Valid video prompt text.",
            estimated_duration_seconds=5.0,
        )

    # Invalid camera_movement
    with pytest.raises(ValidationError):
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="spinning_violently",  # Not in CameraMovement Literal
            visual_description="Valid visual description.",
            video_prompt="Valid video prompt text.",
            estimated_duration_seconds=5.0,
        )

    # Visual description too short (< 5 chars)
    with pytest.raises(ValidationError):
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Tiny",
            video_prompt="Valid video prompt text.",
            estimated_duration_seconds=5.0,
        )

    # Video prompt too short (< 10 chars)
    with pytest.raises(ValidationError):
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Valid visual description.",
            video_prompt="Short",
            estimated_duration_seconds=5.0,
        )

    # Shot duration too short (< 0.5s)
    with pytest.raises(ValidationError):
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Valid visual description.",
            video_prompt="Valid video prompt text.",
            estimated_duration_seconds=0.1,
        )


def test_storyboard_request_validation():
    script = create_sample_script_result()

    # Valid request with auto-inferred workflow_id
    req = StoryboardRequest(
        script=script,
        aspect_ratio="9:16",
        visual_style="cinematic, anamorphic lens, high contrast",
    )
    assert req.workflow_id == script.workflow_id
    assert req.target_duration_seconds == 60
    assert req.aspect_ratio == "9:16"
    assert req.visual_style == "cinematic, anamorphic lens, high contrast"

    # Mismatched workflow_id raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        StoryboardRequest(
            workflow_id=uuid4(),
            script=script,
        )
    assert "must match script.workflow_id" in str(exc_info.value)

    # Invalid aspect ratio
    with pytest.raises(ValidationError):
        StoryboardRequest(
            script=script,
            aspect_ratio="4:3",  # Not in AspectRatio Literal
        )


def test_validate_shots_against_script_success():
    script = create_sample_script_result()
    # 5 shots for 4 scenes (scene 1 is broken into 2 shots: 7.5s + 7.5s = 15.0s)
    shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="pan_right",
            visual_description="Wide shot of datacenter.",
            video_prompt="Cinematic wide pan of datacenter servers with blue LED accents.",
            estimated_duration_seconds=7.5,
        ),
        StoryboardShot(
            shot_number=2,
            scene_number=1,
            shot_framing="close_up",
            camera_movement="zoom_in",
            visual_description="Close-up of fiber cable plugged into switch.",
            video_prompt="Extreme macro close up of glowing yellow optical fiber connectors.",
            estimated_duration_seconds=7.5,
        ),
        StoryboardShot(
            shot_number=3,
            scene_number=2,
            shot_framing="medium_shot",
            camera_movement="static",
            visual_description="Engineer looking concerned at monitor.",
            video_prompt="Cinematic medium shot of software engineer staring at red error logs.",
            estimated_duration_seconds=15.0,
        ),
        StoryboardShot(
            shot_number=4,
            scene_number=3,
            shot_framing="medium_close_up",
            camera_movement="tracking",
            visual_description="Diagram of asynchronous mesh network routing.",
            video_prompt="Dynamic 3D rendering of packet flows routing through mesh network nodes.",
            estimated_duration_seconds=15.0,
        ),
        StoryboardShot(
            shot_number=5,
            scene_number=4,
            shot_framing="wide_shot",
            camera_movement="orbit",
            visual_description="Clean title card with repository link.",
            video_prompt="High-contrast minimalist title card floating in abstract 3D space.",
            estimated_duration_seconds=15.0,
        ),
    ]

    validate_shots_against_script(shots, script)


def test_validate_shots_against_script_unknown_scene():
    script = create_sample_script_result()
    shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=99,  # Unknown scene
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Some visual description.",
            video_prompt="Some video prompt text.",
            estimated_duration_seconds=15.0,
        )
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_shots_against_script(shots, script)
    assert "references unknown scene number 99" in str(exc_info.value)


def test_validate_shots_against_script_missing_scene():
    script = create_sample_script_result()
    # Only cover scenes 1, 2, 3 (missing scene 4)
    shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 1 visual.",
            video_prompt="Video prompt for scene 1.",
            estimated_duration_seconds=15.0,
        ),
        StoryboardShot(
            shot_number=2,
            scene_number=2,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 2 visual.",
            video_prompt="Video prompt for scene 2.",
            estimated_duration_seconds=15.0,
        ),
        StoryboardShot(
            shot_number=3,
            scene_number=3,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 3 visual.",
            video_prompt="Video prompt for scene 3.",
            estimated_duration_seconds=15.0,
        ),
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_shots_against_script(shots, script)
    assert "Storyboard shots must cover all script scenes" in str(exc_info.value)
    assert "Missing scene number(s): [4]" in str(exc_info.value)


def test_validate_shots_against_script_scene_duration_deviation():
    script = create_sample_script_result()
    # Scene 1 is 15.0s, but shots for scene 1 total 30.0s (deviates well beyond allowed slack)
    shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 1 visual.",
            video_prompt="Video prompt for scene 1.",
            estimated_duration_seconds=30.0,
        ),
        StoryboardShot(
            shot_number=2,
            scene_number=2,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 2 visual.",
            video_prompt="Video prompt for scene 2.",
            estimated_duration_seconds=15.0,
        ),
        StoryboardShot(
            shot_number=3,
            scene_number=3,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 3 visual.",
            video_prompt="Video prompt for scene 3.",
            estimated_duration_seconds=15.0,
        ),
        StoryboardShot(
            shot_number=4,
            scene_number=4,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Scene 4 visual.",
            video_prompt="Video prompt for scene 4.",
            estimated_duration_seconds=15.0,
        ),
    ]
    with pytest.raises(ValueError) as exc_info:
        validate_shots_against_script(shots, script)
    assert "Scene 1 shot duration total (30.0s) deviates beyond allowed tolerance" in str(
        exc_info.value
    )


def test_storyboard_result_invariants():
    wf_id = uuid4()
    shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Visual description 1.",
            video_prompt="Prompt for shot 1.",
            estimated_duration_seconds=30.0,
        ),
        StoryboardShot(
            shot_number=2,
            scene_number=2,
            shot_framing="close_up",
            camera_movement="zoom_in",
            visual_description="Visual description 2.",
            video_prompt="Prompt for shot 2.",
            estimated_duration_seconds=30.0,
        ),
    ]

    # Valid result
    result = StoryboardResult(
        workflow_id=wf_id,
        title="Valid Storyboard",
        aspect_ratio="9:16",
        visual_style="Cinematic",
        target_duration_seconds=60,
        estimated_duration_seconds=60.0,
        total_shots=2,
        shots=shots,
    )
    assert result.total_shots == 2
    assert result.estimated_duration_seconds == 60.0

    # Non-sequential shot numbers ([1, 3])
    bad_shots = [
        shots[0],
        StoryboardShot(
            shot_number=3,  # Skipping 2
            scene_number=2,
            shot_framing="close_up",
            camera_movement="zoom_in",
            visual_description="Visual description 2.",
            video_prompt="Prompt for shot 2.",
            estimated_duration_seconds=30.0,
        ),
    ]
    with pytest.raises(ValidationError) as exc_info:
        StoryboardResult(
            workflow_id=wf_id,
            title="Bad Sequence Storyboard",
            aspect_ratio="9:16",
            visual_style="Cinematic",
            target_duration_seconds=60,
            estimated_duration_seconds=60.0,
            total_shots=2,
            shots=bad_shots,
        )
    assert "Storyboard shots must be strictly sequential starting from 1" in str(exc_info.value)

    # total_shots mismatch
    with pytest.raises(ValidationError) as exc_info:
        StoryboardResult(
            workflow_id=wf_id,
            title="Mismatched Shots Storyboard",
            aspect_ratio="9:16",
            visual_style="Cinematic",
            target_duration_seconds=60,
            estimated_duration_seconds=60.0,
            total_shots=10,  # Actually 2
            shots=shots,
        )
    assert "total_shots (10) must match number of shots (2)" in str(exc_info.value)

    # Total duration outside ±15% tolerance: 60s target allows [51.0s, 69.0s]
    long_shots = [
        StoryboardShot(
            shot_number=1,
            scene_number=1,
            shot_framing="wide_shot",
            camera_movement="static",
            visual_description="Visual description 1.",
            video_prompt="Prompt for shot 1.",
            estimated_duration_seconds=45.0,
        ),
        StoryboardShot(
            shot_number=2,
            scene_number=2,
            shot_framing="close_up",
            camera_movement="zoom_in",
            visual_description="Visual description 2.",
            video_prompt="Prompt for shot 2.",
            estimated_duration_seconds=45.0,
        ),
    ]
    with pytest.raises(ValidationError) as exc_info:
        StoryboardResult(
            workflow_id=wf_id,
            title="Too Long Storyboard",
            aspect_ratio="9:16",
            visual_style="Cinematic",
            target_duration_seconds=60,
            estimated_duration_seconds=90.0,
            total_shots=2,
            shots=long_shots,
        )
    assert "outside target_duration_seconds ±15% tolerance" in str(exc_info.value)
