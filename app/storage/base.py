from pathlib import Path
from typing import Protocol
from uuid import UUID

from app.artifacts.models import ArtifactRef


class ArtifactStorage(Protocol):
    async def upload(
        self,
        local_path: Path,
        *,
        destination: str,
        workflow_id: UUID,
        artifact_type: str,
        mime_type: str | None = None,
        overwrite: bool = False,
    ) -> ArtifactRef: ...

    async def download(
        self,
        artifact: ArtifactRef,
        destination: Path,
    ) -> None: ...

    async def delete(
        self,
        artifact: ArtifactRef,
    ) -> None: ...
