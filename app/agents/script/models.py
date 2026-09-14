from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.research.models import ResearchResult

DURATION_TOLERANCE_RATIO: float = 0.15


class ScriptScene(BaseModel):
    """A single timed scene in the generated script."""

    model_config = ConfigDict(frozen=True)

    scene_number: int = Field(ge=1, description="1-indexed sequence order of this scene")
    narrative_beat_sequence: int = Field(
        ge=1,
        description="Reference to the corresponding NarrativeBeat.sequence in ResearchResult",
    )
    beat_type: str = Field(
        min_length=1,
        description="Narrative beat type matching the referenced research beat",
    )
    narration: str = Field(
        min_length=1,
        description="Spoken voiceover narration for this scene",
    )
    visual_direction: str = Field(
        min_length=5,
        description="Visual description, camera staging, or b-roll instructions",
    )
    target_keywords: list[str] = Field(
        min_length=2,
        max_length=5,
        description="Keywords for visual search or prompt hints (2 to 5 keywords required)",
    )
    estimated_duration_seconds: float = Field(
        ge=0.1,
        description="Calculated duration for this scene in seconds",
    )


class ScriptRequest(BaseModel):
    """Input contract requesting a video script from a ResearchResult."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID | None = Field(
        default=None,
        description="Workflow identifier; if omitted, inferred from research.workflow_id",
    )
    research: ResearchResult
    pacing_wpm: int = Field(
        default=140,
        ge=80,
        le=240,
        description="Narration speaking pace in words per minute",
    )
    tone: str = Field(
        default="engaging, informative, and cinematic",
        max_length=200,
        description="Voice and tone direction for the narration",
    )
    call_to_action: str | None = Field(
        default=None,
        max_length=300,
        description="Optional closing call to action",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_workflow_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            workflow_id = data.get("workflow_id")
            research = data.get("research")
            if research is not None:
                res_wf_id = getattr(research, "workflow_id", None)
                if res_wf_id is None and isinstance(research, dict):
                    res_wf_id = research.get("workflow_id")

                if workflow_id is not None and res_wf_id is not None:
                    if str(workflow_id) != str(res_wf_id):
                        raise ValueError(
                            f"ScriptRequest workflow_id ({workflow_id}) must match research.workflow_id ({res_wf_id})."
                        )
                elif workflow_id is None and res_wf_id is not None:
                    data["workflow_id"] = res_wf_id
        return data

    @model_validator(mode="after")
    def validate_workflow_id(self) -> "ScriptRequest":
        if self.workflow_id != self.research.workflow_id:
            raise ValueError(
                f"ScriptRequest workflow_id ({self.workflow_id}) must match research.workflow_id ({self.research.workflow_id})."
            )
        return self

    @property
    def target_duration_seconds(self) -> int:
        return self.research.target_duration_seconds


class ScriptResult(BaseModel):
    """Structured script output consumed by the downstream Storyboard Agent."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    title: str = Field(min_length=3, max_length=300)
    target_duration_seconds: int = Field(ge=15, le=600)
    estimated_duration_seconds: float = Field(ge=0.1)
    total_word_count: int = Field(ge=1)
    scenes: list[ScriptScene] = Field(min_length=1)
    full_narration: str = Field(min_length=1)
    pacing_wpm: int = Field(default=140, ge=80, le=240)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_script_invariants(self) -> "ScriptResult":
        # 1. Scene sequence must be strictly sequential 1..N
        expected_numbers = list(range(1, len(self.scenes) + 1))
        actual_numbers = [s.scene_number for s in self.scenes]
        if actual_numbers != expected_numbers:
            raise ValueError(
                f"Script scenes must be strictly sequential starting from 1 (got {actual_numbers}, expected {expected_numbers})."
            )

        # 2. full_narration must be exactly the joined scene narrations
        expected_narration = " ".join(s.narration.strip() for s in self.scenes)
        if self.full_narration.strip() != expected_narration:
            raise ValueError("full_narration must match the concatenated scene narrations.")

        # 3. total_word_count must match the sum of scene words
        expected_word_count = sum(len(s.narration.split()) for s in self.scenes)
        if self.total_word_count != expected_word_count:
            raise ValueError(
                f"total_word_count ({self.total_word_count}) must match sum of scene word counts ({expected_word_count})."
            )

        # 4. estimated_duration_seconds must match calculated duration from total words and pacing
        expected_duration = round((self.total_word_count / self.pacing_wpm) * 60.0, 2)
        if abs(self.estimated_duration_seconds - expected_duration) > 0.5:
            raise ValueError(
                f"estimated_duration_seconds ({self.estimated_duration_seconds}s) does not match expected duration ({expected_duration}s) calculated from {self.total_word_count} words at {self.pacing_wpm} WPM."
            )

        # 5. estimated_duration_seconds must be within target_duration_seconds ± DURATION_TOLERANCE_RATIO
        min_allowed = round(self.target_duration_seconds * (1.0 - DURATION_TOLERANCE_RATIO), 2)
        max_allowed = round(self.target_duration_seconds * (1.0 + DURATION_TOLERANCE_RATIO), 2)
        if not (min_allowed <= self.estimated_duration_seconds <= max_allowed):
            tolerance_percent = int(DURATION_TOLERANCE_RATIO * 100)
            raise ValueError(
                f"estimated_duration_seconds ({self.estimated_duration_seconds}s) is outside target_duration_seconds "
                f"±{tolerance_percent}% tolerance ([{min_allowed}s, {max_allowed}s] for target {self.target_duration_seconds}s)."
            )

        return self


def validate_scenes_against_research(scenes: list[ScriptScene], research: ResearchResult) -> None:
    """Validate that every scene references a valid narrative beat from ResearchResult,
    and that every narrative beat in the research is represented by at least one scene.
    """
    beat_map = {beat.sequence: beat for beat in research.suggested_narrative_arc}
    for scene in scenes:
        if scene.narrative_beat_sequence not in beat_map:
            raise ValueError(
                f"Scene {scene.scene_number} references unknown narrative beat sequence {scene.narrative_beat_sequence}. Available beats: {sorted(beat_map.keys())}."
            )
        ref_beat = beat_map[scene.narrative_beat_sequence]
        if scene.beat_type != ref_beat.beat_type:
            raise ValueError(
                f"Scene {scene.scene_number} beat_type '{scene.beat_type}' does not match referenced research beat {scene.narrative_beat_sequence} beat_type '{ref_beat.beat_type}'."
            )

    # Complete beat-coverage invariant: every research narrative beat must appear in at least one scene
    covered_beats = {s.narrative_beat_sequence for s in scenes}
    expected_beats = set(beat_map.keys())
    missing_beats = expected_beats - covered_beats
    if missing_beats:
        raise ValueError(
            f"Script scenes must cover all research narrative beats. Missing narrative beat sequence(s): {sorted(missing_beats)}."
        )
