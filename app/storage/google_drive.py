from __future__ import annotations

import asyncio
import io
import mimetypes
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from app.artifacts.models import ArtifactRef

if TYPE_CHECKING:
    from app.config.settings import Settings
from app.logging import get_logger
from app.storage.base import ArtifactStorage
from app.storage.errors import (
    StorageAuthenticationError,
    StorageConfigurationError,
    StorageDuplicateError,
    StorageError,
    StorageNotFoundError,
    StorageOperationError,
)

logger = get_logger(__name__)


class GoogleDriveStorage(ArtifactStorage):
    """Production Google Drive implementation of ArtifactStorage.

    All Google Drive API interactions run in background threads to preserve
    non-blocking async execution. Folder structures are resolved idempotently.
    """

    def __init__(
        self,
        root_folder_id: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        *,
        service: Resource | None = None,
    ) -> None:
        if not root_folder_id or not root_folder_id.strip():
            raise StorageConfigurationError("root_folder_id must not be empty.")

        self.root_folder_id = root_folder_id.strip()

        if service is not None:
            self._service = service
        else:
            if not client_id or not client_secret or not refresh_token:
                raise StorageConfigurationError(
                    "client_id, client_secret, and refresh_token are required when service is not provided."
                )
            try:
                credentials = Credentials(
                    token=None,
                    refresh_token=refresh_token,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=["https://www.googleapis.com/auth/drive.file"],
                )
                self._service = build(
                    "drive",
                    "v3",
                    credentials=credentials,
                    cache_discovery=False,
                )
            except Exception as e:
                raise StorageConfigurationError(
                    f"Failed to initialize Google Drive client: {e}",
                    cause=e,
                ) from e

    @classmethod
    def from_settings(cls, settings: Settings) -> GoogleDriveStorage:
        """Factory method to instantiate GoogleDriveStorage from application Settings."""
        client_id, client_secret, refresh_token, root_folder_id = settings.require_google_drive()
        return cls(
            root_folder_id=root_folder_id,
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
        )

    # -------------------------------------------------------------------------
    # Synchronous helper methods executed via asyncio.to_thread
    # -------------------------------------------------------------------------

    def _find_folder_sync(self, name: str, parent_id: str) -> str | None:
        """Find an un-trashed folder by name within a parent folder."""
        query = (
            f"name = '{name}' and '{parent_id}' in parents and "
            f"mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        response = (
            self._service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
        )
        files = response.get("files", [])
        return files[0]["id"] if files else None

    def _get_or_create_folder_sync(self, name: str, parent_id: str) -> str:
        """Idempotently get or create a folder within a parent folder."""
        existing_id = self._find_folder_sync(name, parent_id)
        if existing_id:
            return existing_id

        folder_metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        created = self._service.files().create(body=folder_metadata, fields="id").execute()
        return created["id"]

    def _find_file_sync(self, name: str, parent_id: str) -> dict[str, Any] | None:
        """Find an un-trashed file by name within a parent folder."""
        query = (
            f"name = '{name}' and '{parent_id}' in parents and "
            f"trashed = false and mimeType != 'application/vnd.google-apps.folder'"
        )
        response = (
            self._service.files()
            .list(q=query, spaces="drive", fields="files(id, name, size, mimeType)")
            .execute()
        )
        files = response.get("files", [])
        return files[0] if files else None

    def _resolve_target_folder_sync(
        self, workflow_id: UUID, artifact_type: str, destination: str
    ) -> tuple[str, str]:
        """Resolve the nested folder structure:

        root -> workflows -> <workflow_id> -> [<subfolders>]
        Returns (target_parent_folder_id, target_filename).
        """
        # 1. workflows folder
        workflows_folder_id = self._get_or_create_folder_sync("workflows", self.root_folder_id)

        # 2. workflow_id folder
        wf_folder_id = self._get_or_create_folder_sync(str(workflow_id), workflows_folder_id)

        # 3. Parse destination path into folders and filename
        clean_dest = destination.replace("\\", "/").strip("/")
        parts = [p for p in clean_dest.split("/") if p]

        if len(parts) > 1:
            # Subfolders explicitly specified in destination
            current_folder_id = wf_folder_id
            for subfolder in parts[:-1]:
                current_folder_id = self._get_or_create_folder_sync(subfolder, current_folder_id)
            target_filename = parts[-1]
            return current_folder_id, target_filename

        # If no subfolder in destination, use artifact_type as subfolder
        type_folder_id = self._get_or_create_folder_sync(artifact_type, wf_folder_id)
        target_filename = parts[0] if parts else "artifact.bin"
        return type_folder_id, target_filename

    def _upload_sync(
        self,
        local_path: Path,
        destination: str,
        workflow_id: UUID,
        artifact_type: str,
        mime_type: str | None,
        overwrite: bool,
    ) -> ArtifactRef:
        """Synchronous upload execution."""
        if not local_path.is_file():
            raise StorageNotFoundError(f"Local source file not found: {local_path}")

        resolved_mime = mime_type or (
            mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        )
        file_size = local_path.stat().st_size

        parent_folder_id, target_filename = self._resolve_target_folder_sync(
            workflow_id=workflow_id,
            artifact_type=artifact_type,
            destination=destination,
        )

        existing_file = self._find_file_sync(target_filename, parent_folder_id)
        media = MediaFileUpload(
            str(local_path),
            mimetype=resolved_mime,
            resumable=file_size > 5 * 1024 * 1024,
        )

        if existing_file:
            if not overwrite:
                raise StorageDuplicateError(
                    f"Artifact '{target_filename}' already exists in destination and overwrite=False."
                )
            # Update existing file content
            updated = (
                self._service.files()
                .update(
                    fileId=existing_file["id"],
                    media_body=media,
                    fields="id, name, size, mimeType",
                )
                .execute()
            )
            drive_file_id = updated["id"]
        else:
            # Create new file
            body = {
                "name": target_filename,
                "parents": [parent_folder_id],
            }
            created = (
                self._service.files()
                .create(
                    body=body,
                    media_body=media,
                    fields="id, name, size, mimeType",
                )
                .execute()
            )
            drive_file_id = created["id"]

        return ArtifactRef(
            workflow_id=workflow_id,
            artifact_type=artifact_type,
            filename=target_filename,
            mime_type=resolved_mime,
            drive_file_id=drive_file_id,
            size_bytes=file_size,
        )

    def _download_sync(self, drive_file_id: str, destination: Path) -> None:
        """Synchronous download execution."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = self._service.files().get_media(fileId=drive_file_id)

        with io.FileIO(str(destination), mode="wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def _delete_sync(self, drive_file_id: str) -> None:
        """Synchronous delete execution."""
        self._service.files().delete(fileId=drive_file_id).execute()

    # -------------------------------------------------------------------------
    # ArtifactStorage Protocol Implementation (Async)
    # -------------------------------------------------------------------------

    async def upload(
        self,
        local_path: Path,
        *,
        destination: str,
        workflow_id: UUID,
        artifact_type: str,
        mime_type: str | None = None,
        overwrite: bool = False,
    ) -> ArtifactRef:
        """Upload a local file to Google Drive and return an immutable ArtifactRef."""
        start_time = time.perf_counter()
        try:
            artifact_ref = await asyncio.to_thread(
                self._upload_sync,
                local_path=local_path,
                destination=destination,
                workflow_id=workflow_id,
                artifact_type=artifact_type,
                mime_type=mime_type,
                overwrite=overwrite,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "storage.upload",
                provider="google_drive",
                workflow_id=str(workflow_id),
                artifact_type=artifact_type,
                filename=artifact_ref.filename,
                size_bytes=artifact_ref.size_bytes,
                duration_ms=round(duration_ms, 2),
                success=True,
            )
            return artifact_ref

        except StorageDuplicateError:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning(
                "storage.upload.duplicate",
                provider="google_drive",
                workflow_id=str(workflow_id),
                destination=destination,
                duration_ms=round(duration_ms, 2),
                success=False,
            )
            raise

        except HttpError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = e.resp.status if hasattr(e, "resp") else None
            logger.error(
                "storage.upload.http_error",
                provider="google_drive",
                workflow_id=str(workflow_id),
                destination=destination,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                success=False,
            )
            if status_code in (401, 403):
                raise StorageAuthenticationError(
                    f"Google Drive authentication/permission failure: {e}",
                    cause=e,
                ) from e
            if status_code == 404:
                raise StorageNotFoundError(
                    f"Google Drive parent folder or file not found: {e}",
                    cause=e,
                ) from e
            raise StorageOperationError(
                f"Google Drive API failure during upload: {e}",
                cause=e,
            ) from e

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "storage.upload.error",
                provider="google_drive",
                workflow_id=str(workflow_id),
                destination=destination,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=type(e).__name__,
            )
            if isinstance(e, (StorageError, FileNotFoundError)):
                raise
            raise StorageOperationError(
                f"Failed to upload artifact to Google Drive: {e}",
                cause=e,
            ) from e

    async def download(
        self,
        artifact: ArtifactRef,
        destination: Path,
    ) -> None:
        """Download an artifact from Google Drive to a local destination path."""
        start_time = time.perf_counter()
        try:
            await asyncio.to_thread(
                self._download_sync,
                drive_file_id=artifact.drive_file_id,
                destination=destination,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "storage.download",
                provider="google_drive",
                drive_file_id=artifact.drive_file_id,
                destination=str(destination),
                duration_ms=round(duration_ms, 2),
                success=True,
            )
        except HttpError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = e.resp.status if hasattr(e, "resp") else None
            logger.error(
                "storage.download.http_error",
                provider="google_drive",
                drive_file_id=artifact.drive_file_id,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                success=False,
            )
            if status_code in (401, 403):
                raise StorageAuthenticationError(
                    f"Google Drive authentication failure on download: {e}",
                    cause=e,
                ) from e
            if status_code == 404:
                raise StorageNotFoundError(
                    f"Artifact with Drive file ID '{artifact.drive_file_id}' not found in Drive.",
                    cause=e,
                ) from e
            raise StorageOperationError(
                f"Google Drive download operation failed: {e}",
                cause=e,
            ) from e
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "storage.download.error",
                provider="google_drive",
                drive_file_id=artifact.drive_file_id,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=type(e).__name__,
            )
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(
                f"Failed to download artifact from Google Drive: {e}",
                cause=e,
            ) from e

    async def delete(
        self,
        artifact: ArtifactRef,
    ) -> None:
        """Delete an artifact from Google Drive."""
        start_time = time.perf_counter()
        try:
            await asyncio.to_thread(
                self._delete_sync,
                drive_file_id=artifact.drive_file_id,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "storage.delete",
                provider="google_drive",
                drive_file_id=artifact.drive_file_id,
                duration_ms=round(duration_ms, 2),
                success=True,
            )
        except HttpError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = e.resp.status if hasattr(e, "resp") else None
            logger.error(
                "storage.delete.http_error",
                provider="google_drive",
                drive_file_id=artifact.drive_file_id,
                status_code=status_code,
                duration_ms=round(duration_ms, 2),
                success=False,
            )
            if status_code in (401, 403):
                raise StorageAuthenticationError(
                    f"Google Drive authentication failure on delete: {e}",
                    cause=e,
                ) from e
            if status_code == 404:
                raise StorageNotFoundError(
                    f"Artifact with Drive file ID '{artifact.drive_file_id}' not found in Drive.",
                    cause=e,
                ) from e
            raise StorageOperationError(
                f"Google Drive delete operation failed: {e}",
                cause=e,
            ) from e
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "storage.delete.error",
                provider="google_drive",
                drive_file_id=artifact.drive_file_id,
                duration_ms=round(duration_ms, 2),
                success=False,
                error_type=type(e).__name__,
            )
            if isinstance(e, StorageError):
                raise
            raise StorageOperationError(
                f"Failed to delete artifact from Google Drive: {e}",
                cause=e,
            ) from e
