from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.orchestrator.state import TaskStatus


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID

    agent_name: str = Field(min_length=1, max_length=100)
    status: TaskStatus = TaskStatus.PENDING

    attempt: int = Field(default=0, ge=0)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
