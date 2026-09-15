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


@pytest.mark.asyncio
async def test_end_to_end_video_generation_to_media_assembly(tmp_path):
    """Verify chaining from StoryboardResult through VideoGenerationService to MediaAssemblyService."""
    import shutil
    import subprocess

    from app.agents.storyboard.models import StoryboardResult, StoryboardShot
    from app.models.video import VideoModel, VideoOperation
    from app.mpt.ffmpeg import FFmpegMediaProcessor
    from app.mpt.models import MediaAssemblyRequest, MediaClip, MediaProfile
    from app.mpt.service import MediaAssemblyService
    from app.video.service import VideoGenerationService

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg not available")

    workflow_id = uuid4()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Generate two real small video clips as provider output
    clip1_bytes_path = raw_dir / "clip1.mp4"
    clip2_bytes_path = raw_dir / "clip2.mp4"

    import asyncio

    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=5:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip1_bytes_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    await asyncio.to_thread(
        subprocess.run,
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=5:size=320x240:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip2_bytes_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    clip1_bytes = clip1_bytes_path.read_bytes()
    clip2_bytes = clip2_bytes_path.read_bytes()

    # 1. Storyboard with 3 shots (15s total)
    storyboard = StoryboardResult(
        workflow_id=workflow_id,
        title="Pipeline Chaining Demo",
        aspect_ratio="9:16",
        visual_style="Cinematic",
        target_duration_seconds=15,
        estimated_duration_seconds=15.0,
        total_shots=3,
        shots=[
            StoryboardShot(
                shot_number=1,
                scene_number=1,
                shot_framing="wide_shot",
                camera_movement="static",
                visual_description="Shot 1 establishing scene.",
                video_prompt="Futuristic city at sunset, 4k.",
                estimated_duration_seconds=5.0,
            ),
            StoryboardShot(
                shot_number=2,
                scene_number=1,
                shot_framing="close_up",
                camera_movement="zoom_in",
                visual_description="Shot 2 close up.",
                video_prompt="Detailed view of quantum server racks.",
                estimated_duration_seconds=5.0,
            ),
            StoryboardShot(
                shot_number=3,
                scene_number=2,
                shot_framing="medium_shot",
                camera_movement="tracking",
                visual_description="Shot 3 tracking.",
                video_prompt="Engineer inspecting holographic console.",
                estimated_duration_seconds=5.0,
            ),
        ],
    )

    # 2. Mock VideoModel returning real generated video bytes
    mock_video_model = AsyncMock(spec=VideoModel)
    mock_video_model.provider = "veo"
    mock_video_model.model_name = "veo-2.0-generate-001"

    async def fake_submit(req):
        return VideoOperation(
            operation_id=f"ops/{req.shot_number}",
            provider="veo",
            status="submitted",
            done=False,
        )

    async def fake_status(op_id):
        shot_num = int(op_id.split("/")[-1])
        data = clip1_bytes if shot_num == 1 else clip2_bytes
        return VideoOperation(
            operation_id=op_id,
            provider="veo",
            status="completed",
            done=True,
            video_bytes=data,
        )

    mock_video_model.submit_generation.side_effect = fake_submit
    mock_video_model.get_operation_status.side_effect = fake_status

    video_staging = tmp_path / "video_staging"
    video_service = VideoGenerationService(
        video_model=mock_video_model,
        staging_dir=video_staging,
        poll_interval_seconds=0.01,
    )

    # Generate shots through VideoGenerationService
    job_records = await video_service.generate_storyboard(storyboard)
    assert len(job_records) == 3
    assert all(r.status == "completed" for r in job_records)

    # 3. MediaAssemblyService chains generated shots
    processor = FFmpegMediaProcessor()
    assembly_staging = tmp_path / "assembly_staging"
    assembly_service = MediaAssemblyService(
        media_processor=processor,
        staging_dir=assembly_staging,
    )

    assembly_clips = []
    for r in job_records:
        local_shot_file = video_staging / str(workflow_id) / f"shot_{r.shot_number}.mp4"
        assert local_shot_file.exists()
        assembly_clips.append(
            MediaClip(
                shot_number=r.shot_number,
                source_path=local_shot_file,
                target_duration_seconds=5.0,
            )
        )

    assembly_req = MediaAssemblyRequest(
        workflow_id=workflow_id,
        clips=assembly_clips,
        profile=MediaProfile.from_aspect_ratio(storyboard.aspect_ratio),
    )

    assembly_result = await assembly_service.assemble(assembly_req)

    # 4. Invariant assertions
    assert assembly_result.workflow_id == workflow_id
    assert assembly_result.output_path.exists()
    assert abs(assembly_result.duration_seconds - 15.0) < 0.5
    assert assembly_result.media_info.has_video is True
    assert assembly_result.media_info.width == 1080
    assert assembly_result.media_info.height == 1920
