"""Workflow repository providing persistence and idempotency handling."""

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.base import utc_now
from app.db.models.workflow import WorkflowModel
from app.orchestration.errors import IdempotencyConflictError, WorkflowNotFoundError
from app.orchestration.state_machine import (
    JobStage,
    WorkflowStatus,
    validate_workflow_transition,
)


class WorkflowRepository:
    """Repository managing workflow entities and idempotency guarantees."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_workflow(
        self,
        topic: str,
        idempotency_key: str | None = None,
    ) -> WorkflowModel:
        """Create a new workflow, returning the existing workflow if idempotency key matches identical topic."""
        if idempotency_key is not None:
            stmt = select(WorkflowModel).where(WorkflowModel.idempotency_key == idempotency_key)
            result = await self._session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is not None:
                if existing.topic == topic:
                    return existing
                raise IdempotencyConflictError(idempotency_key, existing.topic, topic)

        workflow = WorkflowModel(
            topic=topic,
            idempotency_key=idempotency_key,
            status=WorkflowStatus.PENDING.value,
            current_stage=JobStage.RESEARCH.value,
        )
        self._session.add(workflow)
        try:
            await self._session.flush()
        except IntegrityError:
            # Handle race condition where another transaction committed the same idempotency_key
            if idempotency_key is not None:
                stmt = select(WorkflowModel).where(WorkflowModel.idempotency_key == idempotency_key)
                result = await self._session.execute(stmt)
                existing = result.scalar_one_or_none()
                if existing is not None:
                    if existing.topic == topic:
                        return existing
                    raise IdempotencyConflictError(idempotency_key, existing.topic, topic)
            raise
        return workflow

    async def get_workflow(
        self,
        workflow_id: str,
        load_relations: bool = False,
    ) -> WorkflowModel | None:
        """Retrieve a workflow by ID, optionally eager-loading jobs and artifacts."""
        stmt = select(WorkflowModel).where(WorkflowModel.id == workflow_id)
        if load_relations:
            stmt = stmt.options(
                selectinload(WorkflowModel.jobs),
                selectinload(WorkflowModel.artifacts),
            )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def update_status(
        self,
        workflow_id: str,
        status: str,
        current_stage: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> WorkflowModel:
        """Update workflow status and optional stage/error details, validating state transition."""
        workflow = await self.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowNotFoundError(workflow_id)

        validate_workflow_transition(WorkflowStatus(workflow.status), WorkflowStatus(status))
        workflow.status = status
        if current_stage is not None:
            workflow.current_stage = current_stage
        if error_code is not None:
            workflow.error_code = error_code
        if error_message is not None:
            workflow.error_message = error_message
        if status in {WorkflowStatus.COMPLETED.value, WorkflowStatus.FAILED.value, WorkflowStatus.CANCELLED.value}:
            workflow.completed_at = utc_now()
        await self._session.flush()
        return workflow

    async def list_workflows(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
    ) -> list[WorkflowModel]:
        """List workflows ordered by creation date descending with optional status filter."""
        stmt = select(WorkflowModel).order_by(WorkflowModel.created_at.desc()).limit(limit).offset(offset)
        if status is not None:
            stmt = stmt.where(WorkflowModel.status == status)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def count_workflows(self, status: str | None = None) -> int:
        """Return the count of workflows matching optional status filter."""
        stmt = select(func.count(WorkflowModel.id))
        if status is not None:
            stmt = stmt.where(WorkflowModel.status == status)
        result = await self._session.execute(stmt)
        return int(result.scalar_one())
