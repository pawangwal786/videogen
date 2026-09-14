from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.orchestrator.state import WorkflowStatus


class Workflow(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(min_length=1, max_length=500)
    status: WorkflowStatus = WorkflowStatus.CREATED

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
