from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.script.models import ScriptResult

DURATION_TOLERANCE_RATIO: float = 0.15
SCENE_DURATION_TOLERANCE_RATIO: float = 0.25

ShotFraming = Literal[
    "extreme_close_up",
    "close_up",
    "medium_close_up",
    "medium_shot",
    "cowboy_shot",
    "wide_shot",
    "extreme_wide_shot",
]

CameraMovement = Literal[
    "static",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "zoom_in",
    "zoom_out",
    "tracking",
    "drone_aerial",
    "orbit",
]

AspectRatio = Literal["9:16", "16:9", "1:1"]


class StoryboardShot(BaseModel):
    """A single visual shot definition in the storyboard."""

    model_config = ConfigDict(frozen=True)

    shot_number: int = Field(ge=1, description="1-indexed sequence order of this shot")
    scene_number: int = Field(
        ge=1,
        description="Reference to the corresponding ScriptScene.scene_number",
    )
    shot_framing: ShotFraming = Field(
        description="Cinematic shot framing (e.g., wide_shot, medium_shot, close_up)"
    )
    camera_movement: CameraMovement = Field(
        description="Camera movement direction or style (e.g., static, pan_left, tracking, zoom_in)"
    )
    visual_description: str = Field(
        min_length=5,
        description="Detailed description of visual staging, subject, lighting, and action",
    )
    video_prompt: str = Field(
        min_length=10,
        max_length=1000,
        description="Optimized visual prompt for downstream text-to-video generation",
    )
    negative_prompt: str | None = Field(
        default=None,
        max_length=500,
        description="Optional negative prompt specifying visual artifacts to avoid",
    )
    estimated_duration_seconds: float = Field(
        ge=0.5,
        le=60.0,
        description="Duration of this shot in seconds",
    )


class StoryboardRequest(BaseModel):
    """Input contract requesting storyboard shot breakdown from a ScriptResult."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID | None = Field(
        default=None,
        description="Workflow identifier; if omitted, inferred from script.workflow_id",
    )
    script: ScriptResult
    aspect_ratio: AspectRatio = Field(
        default="9:16",
        description="Video aspect ratio (9:16 vertical short, 16:9 widescreen, 1:1 square)",
    )
    visual_style: str = Field(
        default="cinematic, realistic lighting, 4k high detail, natural color grade",
        max_length=300,
        description="Visual style, aesthetic, or camera render guidance for all shots",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_workflow_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            workflow_id = data.get("workflow_id")
            script = data.get("script")
            if script is not None:
                script_wf_id = getattr(script, "workflow_id", None)
                if script_wf_id is None and isinstance(script, dict):
                    script_wf_id = script.get("workflow_id")

                if workflow_id is not None and script_wf_id is not None:
                    if str(workflow_id) != str(script_wf_id):
                        raise ValueError(
                            f"StoryboardRequest workflow_id ({workflow_id}) must match script.workflow_id ({script_wf_id})."
                        )
                elif workflow_id is None and script_wf_id is not None:
                    data["workflow_id"] = script_wf_id
        return data

    @model_validator(mode="after")
    def validate_workflow_id(self) -> "StoryboardRequest":
        if self.workflow_id != self.script.workflow_id:
            raise ValueError(
                f"StoryboardRequest workflow_id ({self.workflow_id}) must match script.workflow_id ({self.script.workflow_id})."
            )
        return self

    @property
    def target_duration_seconds(self) -> int:
        return self.script.target_duration_seconds


class StoryboardResult(BaseModel):
    """Structured storyboard output consumed by downstream video generation layer."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    title: str = Field(min_length=3, max_length=300)
    aspect_ratio: AspectRatio
    visual_style: str = Field(min_length=3, max_length=300)
    target_duration_seconds: int = Field(ge=15, le=600)
    estimated_duration_seconds: float = Field(ge=0.5)
    total_shots: int = Field(ge=1)
    shots: list[StoryboardShot] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_storyboard_invariants(self) -> "StoryboardResult":
        # 1. Shot numbers must be strictly sequential starting from 1
        expected_numbers = list(range(1, len(self.shots) + 1))
        actual_numbers = [s.shot_number for s in self.shots]
        if actual_numbers != expected_numbers:
            raise ValueError(
                f"Storyboard shots must be strictly sequential starting from 1 (got {actual_numbers}, expected {expected_numbers})."
            )

        # 2. total_shots must match len(shots)
        if self.total_shots != len(self.shots):
            raise ValueError(
                f"total_shots ({self.total_shots}) must match number of shots ({len(self.shots)})."
            )

        # 3. estimated_duration_seconds must match sum of shot durations
        sum_durations = round(sum(s.estimated_duration_seconds for s in self.shots), 2)
        if abs(self.estimated_duration_seconds - sum_durations) > 0.1:
            raise ValueError(
                f"estimated_duration_seconds ({self.estimated_duration_seconds}s) does not match sum of shot durations ({sum_durations}s)."
            )

        # 4. Total duration must be within target_duration_seconds ± DURATION_TOLERANCE_RATIO
        min_allowed = round(self.target_duration_seconds * (1.0 - DURATION_TOLERANCE_RATIO), 2)
        max_allowed = round(self.target_duration_seconds * (1.0 + DURATION_TOLERANCE_RATIO), 2)
        if not (min_allowed <= self.estimated_duration_seconds <= max_allowed):
            tolerance_percent = int(DURATION_TOLERANCE_RATIO * 100)
            raise ValueError(
                f"estimated_duration_seconds ({self.estimated_duration_seconds}s) is outside target_duration_seconds "
                f"±{tolerance_percent}% tolerance ([{min_allowed}s, {max_allowed}s] for target {self.target_duration_seconds}s)."
            )

        return self


def validate_shots_against_script(shots: list[StoryboardShot], script: ScriptResult) -> None:
    """Validate shot-to-scene lineage, complete scene coverage, and per-scene duration alignment."""
    scene_map = {scene.scene_number: scene for scene in script.scenes}

    # 1. Every shot must reference a valid scene_number
    for shot in shots:
        if shot.scene_number not in scene_map:
            raise ValueError(
                f"Shot {shot.shot_number} references unknown scene number {shot.scene_number}. Available scenes: {sorted(scene_map.keys())}."
            )

    # 2. Complete scene coverage invariant: every scene must appear in at least one shot
    covered_scenes = {shot.scene_number for shot in shots}
    expected_scenes = set(scene_map.keys())
    missing_scenes = expected_scenes - covered_scenes
    if missing_scenes:
        raise ValueError(
            f"Storyboard shots must cover all script scenes. Missing scene number(s): {sorted(missing_scenes)}."
        )

    # 3. Per-scene duration tolerance check (non-brittle float slack)
    for scene_num, scene in scene_map.items():
        scene_shots = [shot for shot in shots if shot.scene_number == scene_num]
        scene_shots_duration = round(sum(shot.estimated_duration_seconds for shot in scene_shots), 2)
        allowed_slack = max(
            1.5,
            round(scene.estimated_duration_seconds * SCENE_DURATION_TOLERANCE_RATIO, 2),
        )
        if abs(scene_shots_duration - scene.estimated_duration_seconds) > allowed_slack:
            raise ValueError(
                f"Scene {scene_num} shot duration total ({scene_shots_duration}s) deviates beyond "
                f"allowed tolerance ({allowed_slack}s) from scene estimated duration ({scene.estimated_duration_seconds}s)."
            )
