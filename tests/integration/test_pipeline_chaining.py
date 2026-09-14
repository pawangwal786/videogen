import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.research.agent import ResearchAgent
from app.agents.research.models import ResearchRequest
from app.agents.script.agent import ScriptAgent
from app.agents.script.models import ScriptRequest
from app.agents.storyboard.agent import StoryboardAgent
from app.agents.storyboard.models import StoryboardRequest


@pytest.mark.asyncio
async def test_end_to_end_research_script_storyboard_pipeline():
    """Verify end-to-end chaining across Research, Script, and Storyboard agents."""
    workflow_id = uuid4()

    # 1. Research Agent Setup
    research_payload = {
        "executive_summary": "Deep dive into next-generation optical interconnects for distributed AI cluster scaling.",
        "core_angle": "Overcoming silicon copper physical limits with photonic routing.",
        "target_audience": "Systems Architects and AI Engineers",
        "key_takeaways": [
            "Optical interconnects reduce thermal throttling in dense compute fabrics.",
            "Latency is cut by 60% compared to traditional copper backplanes.",
        ],
        "facts": [
            {
                "claim": "Optical IO bandwidth exceeds 10 Terabits per second per socket.",
                "context": "Demonstrated across experimental rack-scale prototypes.",
                "sources": [
                    {
                        "title": "IEEE Photonic Fabric Review 2025",
                        "url": "https://example.com/photonic-fabric",
                        "publisher": "IEEE",
                    }
                ],
                "verification_status": "source_backed",
            },
            {
                "claim": "Energy per transferred bit drops by 4x over copper SerDes.",
                "verification_status": "needs_verification",
            },
            {
                "claim": "Commercial optical chiplets are scheduled for production deployment.",
                "verification_status": "needs_verification",
            },
        ],
        "suggested_narrative_arc": [
            {
                "sequence": 1,
                "beat_type": "hook",
                "description": "Visualizing the heat and limits of modern copper traces.",
                "suggested_duration_ratio": 0.25,
            },
            {
                "sequence": 2,
                "beat_type": "problem",
                "description": "Explaining the physical impedance wall in datacenter scale-up.",
                "suggested_duration_ratio": 0.25,
            },
            {
                "sequence": 3,
                "beat_type": "solution",
                "description": "Introducing photonic waveguides directly integrated on silicon.",
                "suggested_duration_ratio": 0.25,
            },
            {
                "sequence": 4,
                "beat_type": "call_to_action",
                "description": "Summary and invitation to read the architecture benchmark whitepaper.",
                "suggested_duration_ratio": 0.25,
            },
        ],
        "visual_themes": [
            "Glowing laser signals through translucent silicon waveguides",
            "Infrared thermal comparisons between copper and optical interconnects",
        ],
    }

    research_model = AsyncMock()
    research_model.generate = AsyncMock(return_value=json.dumps(research_payload))
    research_agent = ResearchAgent(model=research_model)

    research_req = ResearchRequest(
        workflow_id=workflow_id,
        topic="Photonic AI Interconnects",
        target_duration_seconds=60,
    )
    research_res = await research_agent.run(research_req)

    assert research_res.workflow_id == workflow_id
    assert research_res.target_duration_seconds == 60
    assert len(research_res.suggested_narrative_arc) == 4

    # 2. Script Agent Execution
    # 4 scenes with 35 words each = 140 words total -> 60s at 140 WPM
    words_35 = " ".join(["word"] * 35)
    script_payload = {
        "title": "Light Speed: How Silicon Photonics Powers Next-Gen AI",
        "scenes": [
            {
                "scene_number": 1,
                "narrative_beat_sequence": 1,
                "beat_type": "hook",
                "narration": words_35,
                "visual_direction": "Dramatic macro camera sweep across smoking copper cables.",
                "target_keywords": ["silicon chip", "thermal heat"],
            },
            {
                "scene_number": 2,
                "narrative_beat_sequence": 2,
                "beat_type": "problem",
                "narration": words_35,
                "visual_direction": "Infrared thermal scan showing bandwidth bottleneck throttling.",
                "target_keywords": ["bandwidth bottleneck", "thermal scan"],
            },
            {
                "scene_number": 3,
                "narrative_beat_sequence": 3,
                "beat_type": "solution",
                "narration": words_35,
                "visual_direction": "Crystal clear optical waveguides carrying laser pulses through silicon.",
                "target_keywords": ["optical fiber", "photonic waveguide"],
            },
            {
                "scene_number": 4,
                "narrative_beat_sequence": 4,
                "beat_type": "call_to_action",
                "narration": words_35,
                "visual_direction": "Minimalist graphic title card showing whitepaper link.",
                "target_keywords": ["whitepaper", "datacenter benchmark"],
            },
        ],
    }

    script_model = AsyncMock()
    script_model.generate = AsyncMock(return_value=json.dumps(script_payload))
    script_agent = ScriptAgent(model=script_model)

    script_req = ScriptRequest(
        research=research_res,
        pacing_wpm=140,
        tone="cinematic and cutting-edge",
    )
    script_res = await script_agent.run(script_req)

    assert script_res.workflow_id == workflow_id
    assert script_res.target_duration_seconds == 60
    assert script_res.estimated_duration_seconds == 60.0
    assert len(script_res.scenes) == 4

    # Verify complete beat coverage invariant held
    covered_beats = {s.narrative_beat_sequence for s in script_res.scenes}
    assert covered_beats == {1, 2, 3, 4}

    # 3. Storyboard Agent Execution
    # 5 shots breaking down the 4 scenes (scene 1 has 2 shots of 7.5s each)
    storyboard_payload = {
        "shots": [
            {
                "shot_number": 1,
                "scene_number": 1,
                "shot_framing": "wide_shot",
                "camera_movement": "tracking",
                "visual_description": "Wide establishing shot of massive AI cluster racks with glowing red status LEDs.",
                "video_prompt": "Cinematic wide tracking shot through dense AI supercomputer room with glowing LEDs.",
                "negative_prompt": "blurry, low resolution, watermark",
                "estimated_duration_seconds": 7.5,
            },
            {
                "shot_number": 2,
                "scene_number": 1,
                "shot_framing": "close_up",
                "camera_movement": "zoom_in",
                "visual_description": "Extreme macro close-up of thick copper cable vibrating under immense electrical load.",
                "video_prompt": "Macro zoom into hot copper server cable with heat ripples and dark moody lighting.",
                "negative_prompt": "blurry, low resolution, watermark",
                "estimated_duration_seconds": 7.5,
            },
            {
                "shot_number": 3,
                "scene_number": 2,
                "shot_framing": "medium_shot",
                "camera_movement": "static",
                "visual_description": "Thermal camera perspective revealing scorching hotspots on copper motherboard bus.",
                "video_prompt": "Infrared thermal video visualization of silicon chip heating up under extreme workload.",
                "negative_prompt": "watermark, text",
                "estimated_duration_seconds": 15.0,
            },
            {
                "shot_number": 4,
                "scene_number": 3,
                "shot_framing": "medium_close_up",
                "camera_movement": "pan_right",
                "visual_description": "Brilliant cyan laser light pulsing effortlessly through optical glass channels on silicon die.",
                "video_prompt": "Photorealistic 4k shot of bright cyan laser pulses streaming through transparent silicon waveguides.",
                "negative_prompt": "grain, artifacting",
                "estimated_duration_seconds": 15.0,
            },
            {
                "shot_number": 5,
                "scene_number": 4,
                "shot_framing": "wide_shot",
                "camera_movement": "orbit",
                "visual_description": "High-contrast geometric title card displaying research download URL.",
                "video_prompt": "Clean futuristic title card floating against dark matte backdrop with subtle orbital camera.",
                "negative_prompt": "spelling error, unreadable text",
                "estimated_duration_seconds": 15.0,
            },
        ]
    }

    storyboard_model = AsyncMock()
    storyboard_model.generate = AsyncMock(return_value=json.dumps(storyboard_payload))
    storyboard_agent = StoryboardAgent(model=storyboard_model)

    storyboard_req = StoryboardRequest(
        script=script_res,
        aspect_ratio="9:16",
        visual_style="cinematic sci-fi documentary, anamorphic lens flare, 4k",
    )
    storyboard_res = await storyboard_agent.run(storyboard_req)

    # 4. End-to-end invariant assertions
    assert storyboard_res.workflow_id == workflow_id
    assert storyboard_res.title == script_res.title
    assert storyboard_res.target_duration_seconds == 60
    assert storyboard_res.estimated_duration_seconds == 60.0
    assert storyboard_res.total_shots == 5
    assert len(storyboard_res.shots) == 5

    # Verify complete scene coverage invariant held
    covered_scenes = {shot.scene_number for shot in storyboard_res.shots}
    assert covered_scenes == {1, 2, 3, 4}

    # Verify per-scene duration mapping
    scene1_duration = sum(
        s.estimated_duration_seconds for s in storyboard_res.shots if s.scene_number == 1
    )
    assert scene1_duration == 15.0
