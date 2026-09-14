from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ResearchSource(BaseModel):
    """Source reference supporting a factual research claim."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1, description="Title of the source or article")
    url: HttpUrl = Field(description="Valid HTTP/HTTPS URL of the source")
    publisher: str | None = Field(default=None, description="Publisher or publication name")
    published_at: datetime | None = Field(default=None, description="Publication date if available")


class ResearchFact(BaseModel):
    """A factual claim with associated citations and verification state."""

    model_config = ConfigDict(frozen=True)

    claim: str = Field(
        min_length=3, description="A verifiable factual claim, statistic, or milestone"
    )
    context: str | None = Field(
        default=None, description="Nuance, background, or explanatory context"
    )
    sources: list[ResearchSource] = Field(
        default_factory=list, description="Associated source citations"
    )
    verification_status: Literal["source_backed", "needs_verification"] = Field(
        default="needs_verification",
        description="Whether the claim has verifiable source attribution",
    )

    @model_validator(mode="after")
    def validate_source_backed_invariant(self) -> "ResearchFact":
        """Enforce that source_backed facts must include at least one source."""
        if self.verification_status == "source_backed" and not self.sources:
            raise ValueError(
                "A 'source_backed' fact must contain at least one source in 'sources'."
            )
        return self


class NarrativeBeat(BaseModel):
    """A sequential beat in the suggested video narrative arc."""

    model_config = ConfigDict(frozen=True)

    sequence: int = Field(ge=1, description="1-indexed sequence order of this narrative beat")
    beat_type: str = Field(
        min_length=1,
        description="Type of beat, e.g. 'hook', 'problem', 'insight', 'climax', 'call_to_action'",
    )
    description: str = Field(
        min_length=3,
        description="Guidance on what this segment should communicate",
    )
    suggested_duration_ratio: float = Field(
        default=0.2,
        ge=0.05,
        le=1.0,
        description="Suggested proportion of total video runtime (0.05 - 1.0)",
    )


class ResearchRequest(BaseModel):
    """Input contract requesting research for a video topic."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    topic: str = Field(min_length=3, max_length=500)
    target_audience: str = Field(default="General audience", max_length=200)
    target_duration_seconds: int = Field(default=60, ge=15, le=600)
    research_depth: Literal["quick", "standard", "deep"] = "standard"
    style_guidelines: list[str] = Field(default_factory=list)
    key_aspects_to_cover: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Structured research output consumed by the downstream Script Agent."""

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    topic: str = Field(min_length=1)
    research_depth: Literal["quick", "standard", "deep"] = "standard"
    research_method: str = Field(default="llm_synthesis")
    target_duration_seconds: int = Field(default=60, ge=15, le=600)
    executive_summary: str = Field(min_length=10, max_length=2000)
    core_angle: str = Field(
        min_length=5,
        max_length=500,
        description="The unique narrative hook or framing angle",
    )
    target_audience: str = Field(min_length=1)
    key_takeaways: list[str] = Field(
        min_length=2,
        max_length=8,
        description="Core takeaways or thesis points",
    )
    facts: list[ResearchFact] = Field(
        min_length=3,
        description="At least 3 structured factual claims",
    )
    suggested_narrative_arc: list[NarrativeBeat] = Field(
        min_length=3,
        description="Sequential narrative beats",
    )
    visual_themes: list[str] = Field(
        min_length=2,
        description="Visual themes, moods, or motifs for storyboard guidance",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_narrative_arc(self) -> "ResearchResult":
        """Validate aggregate duration ratio and sequence numbering."""
        beats = self.suggested_narrative_arc

        # 1. Validate collective duration ratio sum
        total_ratio = sum(b.suggested_duration_ratio for b in beats)
        if not (0.98 <= total_ratio <= 1.02):
            raise ValueError(
                f"Narrative beat duration ratios must sum to approximately 1.0 (got {total_ratio:.3f})."
            )

        # 2. Validate sequence ordering
        sequences = [b.sequence for b in beats]
        expected_sequences = list(range(1, len(beats) + 1))
        if sequences != expected_sequences:
            raise ValueError(
                f"Narrative beat sequences must be strictly sequential starting from 1 (got {sequences}, expected {expected_sequences})."
            )

        return self
