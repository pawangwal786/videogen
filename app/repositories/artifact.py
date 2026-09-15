"""Artifact repository for tracking pipeline outputs and storage paths."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import utc_now
from app.db.models.artifact import ArtifactModel
from app.orchestration.state_machine import ArtifactLifecycleStatus


class ArtifactRepository:
    """Repository managing artifact records linked to workflows and jobs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_artifact(
        self,
        workflow_id: str,
        artifact_type: str,
        storage_path: str,
        job_id: str | None = None,
        gdrive_file_id: str | None = None,
        file_size_bytes: int | None = None,
        checksum: str | None = None,
        mime_type: str | None = None,
        artifact_metadata: dict[str, Any] | None = None,
        lifecycle_status: str = ArtifactLifecycleStatus.AVAILABLE.value,
    ) -> ArtifactModel:
        """Persist a new artifact reference."""
        artifact = ArtifactModel(
            workflow_id=workflow_id,
            job_id=job_id,
            artifact_type=artifact_type,
            storage_path=storage_path,
            gdrive_file_id=gdrive_file_id,
            file_size_bytes=file_size_bytes,
            checksum=checksum,
            mime_type=mime_type,
            lifecycle_status=lifecycle_status,
            artifact_metadata=artifact_metadata,
        )
        self._session.add(artifact)
        await self._session.flush()
        return artifact

    async def get_artifact(self, artifact_id: str) -> ArtifactModel | None:
        """Retrieve an artifact by ID."""
        stmt = select(ArtifactModel).where(ArtifactModel.id == artifact_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_artifacts_for_workflow(
        self,
        workflow_id: str,
        artifact_type: str | None = None,
    ) -> list[ArtifactModel]:
        """List all artifacts for a given workflow, optionally filtered by artifact_type."""
        stmt = (
            select(ArtifactModel)
            .where(ArtifactModel.workflow_id == workflow_id)
            .order_by(ArtifactModel.created_at.asc())
        )
        if artifact_type is not None:
            stmt = stmt.where(ArtifactModel.artifact_type == artifact_type)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_artifacts_for_job(self, job_id: str) -> list[ArtifactModel]:
        """List all artifacts produced by a specific job."""
        stmt = (
            select(ArtifactModel)
            .where(ArtifactModel.job_id == job_id)
            .order_by(ArtifactModel.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_lifecycle_status(
        self,
        artifact_id: str,
        status: str,
    ) -> ArtifactModel | None:
        """Update the lifecycle status of an artifact."""
        artifact = await self.get_artifact(artifact_id)
        if artifact is not None:
            artifact.lifecycle_status = status
            artifact.updated_at = utc_now()
            await self._session.flush()
        return artifact
