from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

VideoJobStatus = Literal[
    "queued",
    "submitted",
    "processing",
    "completed",
    "failed",
    "cancelled",
]

TERMINAL_VIDEO_STATUSES: frozenset[VideoJobStatus] = frozenset({"completed", "failed", "cancelled"})
ACTIVE_VIDEO_STATUSES: frozenset[VideoJobStatus] = frozenset({"queued", "submitted", "processing"})

VALID_TRANSITIONS: dict[VideoJobStatus, set[VideoJobStatus]] = {
    "queued": {"submitted", "cancelled"},
    "submitted": {"processing", "completed", "failed", "cancelled"},
    "processing": {"completed", "failed", "cancelled"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}


class VideoGenerationRequest(BaseModel):
    """Provider-neutral video generation request for a discrete shot.

    Created after provider capability resolution has determined the supported duration.
    """

    model_config = ConfigDict(frozen=True)

    workflow_id: UUID
    shot_number: int = Field(ge=1, description="1-indexed storyboard shot number")
    prompt: str = Field(min_length=10, max_length=1000, description="Visual video prompt")
    negative_prompt: str | None = Field(
        default=None,
        max_length=500,
        description="Visual elements to avoid",
    )
    duration_seconds: int = Field(
        ge=1,
        le=60,
        description="Discrete clip duration in seconds supported by the target provider",
    )
    aspect_ratio: str = Field(
        default="9:16",
        description="Video aspect ratio (e.g., '9:16', '16:9', '1:1')",
    )
    attempt: int = Field(
        default=1,
        ge=1,
        description="Execution attempt counter (metadata, distinct from logical identity)",
    )

    @property
    def logical_key(self) -> str:
        """Stable logical identity for idempotency checks (same logical shot)."""
        return f"{self.workflow_id}:shot:{self.shot_number}"

    @property
    def attempt_key(self) -> str:
        """Unique key identifying a specific execution attempt."""
        return f"{self.workflow_id}:shot:{self.shot_number}:attempt:{self.attempt}"


class VideoOperation(BaseModel):
    """Provider-returned state of an asynchronous video generation operation."""

    model_config = ConfigDict(frozen=True)

    operation_id: str = Field(min_length=1, description="Provider-assigned operation identifier")
    provider: str = Field(min_length=1, description="Provider name (e.g., 'veo')")
    status: VideoJobStatus = Field(default="submitted")
    done: bool = Field(default=False)
    error_message: str | None = None
    video_bytes: bytes | None = None
    video_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoJobRecord(BaseModel):
    """Persistent tracking record for a video generation job across its lifecycle."""

    job_id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID
    shot_number: int = Field(ge=1)
    attempt: int = Field(default=1, ge=1)
    logical_key: str
    provider: str
    model_name: str
    operation_id: str | None = None
    status: VideoJobStatus = "queued"
    prompt: str
    negative_prompt: str | None = None
    duration_seconds: int
    aspect_ratio: str
    output_artifact_id: UUID | None = None
    output_storage_path: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def transition_to(self, new_status: VideoJobStatus, error_message: str | None = None) -> None:
        """Validate and transition to a new lifecycle state."""
        allowed = VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid lifecycle transition for job {self.job_id} from '{self.status}' to '{new_status}'."
            )
        self.status = new_status
        now = datetime.now(UTC)

        if new_status in {"submitted", "processing"} and self.started_at is None:
            self.started_at = now
        elif new_status in TERMINAL_VIDEO_STATUSES:
            self.completed_at = now

        if error_message:
            self.error_message = error_message


class VideoModel(Protocol):
    """Protocol for asynchronous video generation models."""

    async def submit_generation(
        self,
        request: VideoGenerationRequest,
    ) -> VideoOperation:
        """Submit a video generation job to the underlying provider."""
        ...

    async def get_operation_status(
        self,
        operation_id: str,
    ) -> VideoOperation:
        """Poll the status of an ongoing video generation operation."""
        ...

    async def cancel_generation(
        self,
        operation_id: str,
    ) -> None:
        """Request cancellation of an active generation operation."""
        ...
