import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.script.models import ScriptResult, ScriptScene
from app.agents.storyboard.agent import StoryboardAgent, extract_json_payload
from app.agents.storyboard.errors import (
    StoryboardResponseError,
    StoryboardValidationError,
)
from app.agents.storyboard.models import StoryboardRequest, StoryboardResult
from app.models.errors import ModelTimeoutError


@pytest.fixture
def mock_text_model():
    model = AsyncMock()
    model.generate = AsyncMock()
    return model


@pytest.fixture
def sample_script():
    wf_id = uuid4()
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
        title="Scalable Async Video",
        target_duration_seconds=60,
        estimated_duration_seconds=60.0,
        total_word_count=140,
        scenes=scenes,
        full_narration=" ".join(s.narration.strip() for s in scenes),
        pacing_wpm=140,
    )


@pytest.fixture
def valid_storyboard_payload():
    return {
        "shots": [
            {
                "shot_number": 1,
                "scene_number": 1,
                "shot_framing": "wide_shot",
                "camera_movement": "tracking",
                "visual_description": "Wide pan of congested microchip circuitry glowing with red heat.",
                "video_prompt": "Cinematic wide shot of microscopic silicon wafers with pulsing red thermal traces, 35mm lens.",
                "negative_prompt": "watermark, text overlay, blur",
                "estimated_duration_seconds": 15.0,
            },
            {
                "shot_number": 2,
                "scene_number": 2,
                "shot_framing": "medium_close_up",
                "camera_movement": "static",
                "visual_description": "Split screen diagram of latency curves colliding.",
                "video_prompt": "Clean holographic HUD displaying divergent latency line charts with soft neon cyan and magenta glow.",
                "negative_prompt": "grain, artifacting, jitter",
                "estimated_duration_seconds": 15.0,
            },
            {
                "shot_number": 3,
                "scene_number": 3,
                "shot_framing": "medium_shot",
                "camera_movement": "zoom_in",
                "visual_description": "Pulses of emerald light traversing an asynchronous distributed network mesh.",
                "video_prompt": "Fluid dynamic motion graphic of glowing emerald signal pulses routing seamlessly through fiber nodes.",
                "negative_prompt": "low resolution, dark spots",
                "estimated_duration_seconds": 15.0,
            },
            {
                "shot_number": 4,
                "scene_number": 4,
                "shot_framing": "wide_shot",
                "camera_movement": "orbit",
                "visual_description": "Minimalist graphic title card showing repository link and start command.",
                "video_prompt": "Sleek modern typography floating over geometric dark glass backdrop, subtle camera orbit.",
                "negative_prompt": "spelling error, unreadable text, blurry",
                "estimated_duration_seconds": 15.0,
            },
        ]
    }


def test_extract_json_payload_clean():
    data = {"shots": []}
    assert extract_json_payload(json.dumps(data)) == data


def test_extract_json_payload_markdown_fenced():
    data = {"shots": []}
    raw = f"```json\n{json.dumps(data)}\n```"
    assert extract_json_payload(raw) == data


def test_extract_json_payload_invalid_syntax():
    with pytest.raises(StoryboardResponseError) as exc_info:
        extract_json_payload("Plain text not containing any JSON.")
    assert "Failed to decode JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_success(mock_text_model, sample_script, valid_storyboard_payload):
    mock_text_model.generate.return_value = json.dumps(valid_storyboard_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(
        script=sample_script,
        aspect_ratio="9:16",
        visual_style="cinematic, photorealistic, 4k",
    )

    result = await agent.run(request)

    assert isinstance(result, StoryboardResult)
    assert result.workflow_id == sample_script.workflow_id
    assert result.title == sample_script.title
    assert result.aspect_ratio == "9:16"
    assert result.visual_style == "cinematic, photorealistic, 4k"
    assert result.target_duration_seconds == 60
    assert result.total_shots == 4
    assert result.estimated_duration_seconds == 60.0
    assert len(result.shots) == 4

    mock_text_model.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_storyboard_agent_preserves_workflow_id(
    mock_text_model, sample_script, valid_storyboard_payload
):
    payload = dict(valid_storyboard_payload)
    payload["workflow_id"] = str(uuid4())  # Even if response has arbitrary workflow_id

    mock_text_model.generate.return_value = json.dumps(payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    result = await agent.run(request)

    assert result.workflow_id == sample_script.workflow_id


@pytest.mark.asyncio
async def test_storyboard_agent_invalid_json(mock_text_model, sample_script):
    mock_text_model.generate.return_value = "Non-JSON response from LLM"
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardResponseError) as exc_info:
        await agent.run(request)
    assert "Failed to decode JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_missing_shots_key(mock_text_model, sample_script):
    bad_payload = {"visual_breakdown": []}
    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardValidationError) as exc_info:
        await agent.run(request)
    assert "non-empty 'shots' list" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_shot_validation_failure(
    mock_text_model, sample_script, valid_storyboard_payload
):
    bad_payload = dict(valid_storyboard_payload)
    bad_payload["shots"][0]["shot_framing"] = "invalid_framing"

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardValidationError) as exc_info:
        await agent.run(request)
    assert "failed validation" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_unknown_scene_raises_validation_error(
    mock_text_model, sample_script, valid_storyboard_payload
):
    bad_payload = dict(valid_storyboard_payload)
    bad_payload["shots"][0]["scene_number"] = 99

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardValidationError) as exc_info:
        await agent.run(request)
    assert "references unknown scene number 99" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_missing_scene_raises_validation_error(
    mock_text_model, sample_script, valid_storyboard_payload
):
    # Drop shot 4 (scene 4)
    bad_payload = dict(valid_storyboard_payload)
    bad_payload["shots"] = [s for s in valid_storyboard_payload["shots"] if s["scene_number"] != 4]

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardValidationError) as exc_info:
        await agent.run(request)
    assert "Storyboard shots must cover all script scenes" in str(exc_info.value)
    assert "Missing scene number(s): [4]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_scene_duration_deviation_raises_validation_error(
    mock_text_model, sample_script, valid_storyboard_payload
):
    bad_payload = dict(valid_storyboard_payload)
    # Scene 1 is 15.0s, set shot 1 to 25.0s (deviates beyond allowed slack)
    bad_payload["shots"][0]["estimated_duration_seconds"] = 25.0

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardValidationError) as exc_info:
        await agent.run(request)
    assert "Scene 1 shot duration total (25.0s) deviates beyond allowed tolerance" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_storyboard_agent_total_duration_out_of_tolerance_raises_validation_error(
    mock_text_model, sample_script, valid_storyboard_payload
):
    bad_payload = dict(valid_storyboard_payload)
    # Each shot 18.0s:
    # Per-scene check: 18.0s <= 15.0 + 3.75 (allowed slack) -> passes per-scene check
    # Total duration: 4 * 18.0s = 72.0s (> 69.0s max allowed for 60s target ±15%) -> fails StoryboardResult validation
    for s in bad_payload["shots"]:
        s["estimated_duration_seconds"] = 18.0

    mock_text_model.generate.return_value = json.dumps(bad_payload)
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(StoryboardValidationError) as exc_info:
        await agent.run(request)
    assert "outside target_duration_seconds" in str(exc_info.value)
    assert "tolerance" in str(exc_info.value)


@pytest.mark.asyncio
async def test_storyboard_agent_model_error_propagates(mock_text_model, sample_script):
    mock_text_model.generate.side_effect = ModelTimeoutError(
        "OpenRouter request timed out", provider="openrouter"
    )
    agent = StoryboardAgent(model=mock_text_model)

    request = StoryboardRequest(script=sample_script)
    with pytest.raises(ModelTimeoutError):
        await agent.run(request)


@pytest.mark.asyncio
async def test_storyboard_agent_custom_system_prompt(
    mock_text_model, sample_script, valid_storyboard_payload
):
    mock_text_model.generate.return_value = json.dumps(valid_storyboard_payload)
    custom_prompt = "You are an anime animation director."
    agent = StoryboardAgent(model=mock_text_model, system_prompt=custom_prompt)

    request = StoryboardRequest(script=sample_script)
    await agent.run(request)

    assert mock_text_model.generate.await_args.kwargs["system_prompt"] == custom_prompt
