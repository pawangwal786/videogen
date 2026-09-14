from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from googleapiclient.errors import HttpError

from app.artifacts.models import ArtifactRef
from app.storage.errors import (
    StorageAuthenticationError,
    StorageConfigurationError,
    StorageDuplicateError,
    StorageNotFoundError,
    StorageOperationError,
)
from app.storage.google_drive import GoogleDriveStorage


@pytest.fixture
def mock_drive_service():
    service = MagicMock()
    service.files = MagicMock()
    return service


@pytest.fixture
def sample_file(tmp_path: Path):
    file_path = tmp_path / "script.txt"
    file_path.write_text("Scene 1: Introduction to AI", encoding="utf-8")
    return file_path


def test_drive_storage_missing_root_folder():
    with pytest.raises(StorageConfigurationError):
        GoogleDriveStorage(root_folder_id="")


def test_drive_storage_missing_credentials_without_service():
    with pytest.raises(StorageConfigurationError):
        GoogleDriveStorage(root_folder_id="root-123")


@pytest.mark.asyncio
async def test_drive_upload_new_file_success(mock_drive_service, sample_file):
    # Setup mock responses
    # 1. list for workflows folder -> not found -> create
    # 2. list for workflow_id folder -> not found -> create
    # 3. list for artifact_type folder -> not found -> create
    # 4. list for existing file -> not found -> create file
    mock_files_resource = mock_drive_service.files.return_value

    # list returns empty list of files for all queries
    mock_files_resource.list.return_value.execute.return_value = {"files": []}

    # create returns synthetic IDs
    created_counter = 0

    def mock_create(body, **kwargs):
        nonlocal created_counter
        created_counter += 1
        return MagicMock(execute=lambda: {"id": f"drive-id-{created_counter}"})

    mock_files_resource.create.side_effect = mock_create

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    wf_id = uuid4()
    artifact = await storage.upload(
        sample_file,
        destination="script.txt",
        workflow_id=wf_id,
        artifact_type="script",
        mime_type="text/plain",
        overwrite=False,
    )

    assert isinstance(artifact, ArtifactRef)
    assert artifact.workflow_id == wf_id
    assert artifact.artifact_type == "script"
    assert artifact.filename == "script.txt"
    assert artifact.mime_type == "text/plain"
    assert artifact.drive_file_id.startswith("drive-id-")
    assert artifact.size_bytes == sample_file.stat().st_size


@pytest.mark.asyncio
async def test_drive_upload_duplicate_without_overwrite_raises(mock_drive_service, sample_file):
    mock_files_resource = mock_drive_service.files.return_value

    # Simulate folder lookups finding existing folders, and file lookup finding existing file
    def mock_list(q, **kwargs):
        if "mimeType = 'application/vnd.google-apps.folder'" in q:
            return MagicMock(execute=lambda: {"files": [{"id": "existing-folder-id"}]})
        # File query returns an existing file
        return MagicMock(
            execute=lambda: {
                "files": [
                    {
                        "id": "existing-file-id",
                        "name": "script.txt",
                        "size": "100",
                        "mimeType": "text/plain",
                    }
                ]
            }
        )

    mock_files_resource.list.side_effect = mock_list

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    wf_id = uuid4()
    with pytest.raises(StorageDuplicateError) as exc_info:
        await storage.upload(
            sample_file,
            destination="script.txt",
            workflow_id=wf_id,
            artifact_type="script",
            overwrite=False,
        )
    assert "already exists in destination and overwrite=False" in str(exc_info.value)


@pytest.mark.asyncio
async def test_drive_upload_duplicate_with_overwrite_updates(mock_drive_service, sample_file):
    mock_files_resource = mock_drive_service.files.return_value

    def mock_list(q, **kwargs):
        if "mimeType = 'application/vnd.google-apps.folder'" in q:
            return MagicMock(execute=lambda: {"files": [{"id": "existing-folder-id"}]})
        return MagicMock(
            execute=lambda: {
                "files": [
                    {
                        "id": "existing-file-id",
                        "name": "script.txt",
                        "size": "100",
                        "mimeType": "text/plain",
                    }
                ]
            }
        )

    mock_files_resource.list.side_effect = mock_list
    mock_files_resource.update.return_value.execute.return_value = {
        "id": "existing-file-id",
        "name": "script.txt",
        "size": str(sample_file.stat().st_size),
        "mimeType": "text/plain",
    }

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    wf_id = uuid4()
    artifact = await storage.upload(
        sample_file,
        destination="script.txt",
        workflow_id=wf_id,
        artifact_type="script",
        overwrite=True,
    )

    assert artifact.drive_file_id == "existing-file-id"
    mock_files_resource.update.assert_called_once()


@pytest.mark.asyncio
async def test_drive_upload_nonexistent_local_file_raises(mock_drive_service, tmp_path):
    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )
    non_existent = tmp_path / "missing.txt"

    with pytest.raises(StorageNotFoundError):
        await storage.upload(
            non_existent,
            destination="missing.txt",
            workflow_id=uuid4(),
            artifact_type="script",
        )


@pytest.mark.asyncio
async def test_drive_download_success(mock_drive_service, tmp_path):
    mock_files_resource = mock_drive_service.files.return_value

    # Mock get_media
    mock_request = MagicMock()
    mock_files_resource.get_media.return_value = mock_request

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    dest_path = tmp_path / "downloaded" / "output.txt"
    artifact = ArtifactRef(
        workflow_id=uuid4(),
        artifact_type="script",
        filename="output.txt",
        mime_type="text/plain",
        drive_file_id="drive-12345",
        size_bytes=42,
    )

    # In MediaIoBaseDownload, mock it so next_chunk returns done=True
    # To avoid needing real network in unit test, monkeypatch or write dummy chunk
    with pytest.MonkeyPatch.context() as mp:

        class DummyDownloader:
            def __init__(self, fh, req):
                self.fh = fh

            def next_chunk(self):
                self.fh.write(b"Downloaded artifact content")
                return None, True

        mp.setattr("app.storage.google_drive.MediaIoBaseDownload", DummyDownloader)
        await storage.download(artifact, dest_path)

    assert dest_path.is_file()
    assert dest_path.read_text(encoding="utf-8") == "Downloaded artifact content"


@pytest.mark.asyncio
async def test_drive_delete_success(mock_drive_service):
    mock_files_resource = mock_drive_service.files.return_value
    mock_files_resource.delete.return_value.execute.return_value = ""

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    artifact = ArtifactRef(
        workflow_id=uuid4(),
        artifact_type="script",
        filename="output.txt",
        mime_type="text/plain",
        drive_file_id="drive-delete-123",
    )

    await storage.delete(artifact)
    mock_files_resource.delete.assert_called_once_with(fileId="drive-delete-123")


@pytest.mark.asyncio
async def test_drive_delete_404_raises_storage_not_found(mock_drive_service):
    mock_files_resource = mock_drive_service.files.return_value
    resp = MagicMock(status=404)
    mock_files_resource.delete.return_value.execute.side_effect = HttpError(
        resp=resp, content=b"File not found"
    )

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    artifact = ArtifactRef(
        workflow_id=uuid4(),
        artifact_type="script",
        filename="output.txt",
        mime_type="text/plain",
        drive_file_id="missing-id",
    )

    with pytest.raises(StorageNotFoundError):
        await storage.delete(artifact)


def test_drive_storage_from_settings():
    from pydantic import SecretStr

    from app.config.settings import Settings

    settings = Settings(
        google_drive_client_id=SecretStr("mock-client-id"),
        google_drive_client_secret=SecretStr("mock-client-secret"),
        google_drive_refresh_token=SecretStr("mock-refresh-token"),
        google_drive_root_folder_id="mock-root-folder",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("app.storage.google_drive.Credentials", MagicMock())
        mp.setattr("app.storage.google_drive.build", MagicMock())
        storage = GoogleDriveStorage.from_settings(settings)
        assert storage.root_folder_id == "mock-root-folder"


def test_drive_storage_credentials_initialization_failure():
    with pytest.MonkeyPatch.context() as mp:
        mock_creds = MagicMock(side_effect=RuntimeError("OAuth discovery failure"))
        mp.setattr("app.storage.google_drive.Credentials", mock_creds)
        with pytest.raises(StorageConfigurationError) as exc_info:
            GoogleDriveStorage(
                root_folder_id="root-123",
                client_id="client",
                client_secret="secret",
                refresh_token="token",
            )
        assert "Failed to initialize Google Drive client" in str(exc_info.value)


@pytest.mark.asyncio
async def test_drive_upload_nested_destination_path(mock_drive_service, sample_file):
    mock_files = mock_drive_service.files.return_value
    mock_files.list.return_value.execute.return_value = {"files": []}
    mock_files.create.return_value.execute.return_value = {"id": "nested-file-id"}

    storage = GoogleDriveStorage(
        root_folder_id="root-folder-id",
        service=mock_drive_service,
    )

    ref = await storage.upload(
        sample_file,
        destination="shots/take1/angleA/scene_1.mp4",
        workflow_id=uuid4(),
        artifact_type="video_shot",
        mime_type="video/mp4",
    )
    assert ref.drive_file_id == "nested-file-id"
    assert ref.filename == "scene_1.mp4"


@pytest.mark.asyncio
async def test_drive_upload_http_errors_and_generic_error(mock_drive_service, sample_file):
    mock_files = mock_drive_service.files.return_value
    storage = GoogleDriveStorage(root_folder_id="root-123", service=mock_drive_service)
    wf_id = uuid4()

    # 401/403 -> StorageAuthenticationError
    mock_files.list.return_value.execute.side_effect = HttpError(
        MagicMock(status=401), b"Unauthorized"
    )
    with pytest.raises(StorageAuthenticationError):
        await storage.upload(
            sample_file,
            destination="out.txt",
            workflow_id=wf_id,
            artifact_type="text/plain",
            mime_type="text/plain",
        )

    # 404 -> StorageNotFoundError
    mock_files.list.return_value.execute.side_effect = HttpError(
        MagicMock(status=404), b"Not found"
    )
    with pytest.raises(StorageNotFoundError):
        await storage.upload(
            sample_file,
            destination="out.txt",
            workflow_id=wf_id,
            artifact_type="text/plain",
            mime_type="text/plain",
        )

    # 500 -> StorageOperationError
    mock_files.list.return_value.execute.side_effect = HttpError(
        MagicMock(status=500), b"Server error"
    )
    with pytest.raises(StorageOperationError):
        await storage.upload(
            sample_file,
            destination="out.txt",
            workflow_id=wf_id,
            artifact_type="text/plain",
            mime_type="text/plain",
        )

    # Unexpected non-storage error -> StorageOperationError
    mock_files.list.return_value.execute.side_effect = RuntimeError("Disk full")
    with pytest.raises(StorageOperationError) as exc_info:
        await storage.upload(
            sample_file,
            destination="out.txt",
            workflow_id=wf_id,
            artifact_type="text/plain",
            mime_type="text/plain",
        )
    assert "Disk full" in str(exc_info.value)


@pytest.mark.asyncio
async def test_drive_download_http_errors_and_generic_error(mock_drive_service, tmp_path):
    storage = GoogleDriveStorage(root_folder_id="root-123", service=mock_drive_service)
    dest_path = tmp_path / "downloaded.txt"
    artifact = ArtifactRef(
        workflow_id=uuid4(),
        artifact_type="script",
        filename="output.txt",
        mime_type="text/plain",
        drive_file_id="file-id-123",
    )

    # 401/403 -> StorageAuthenticationError
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.storage.google_drive.MediaIoBaseDownload",
            MagicMock(side_effect=HttpError(MagicMock(status=403), b"Forbidden")),
        )
        with pytest.raises(StorageAuthenticationError):
            await storage.download(artifact, dest_path)

    # 500 -> StorageOperationError
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.storage.google_drive.MediaIoBaseDownload",
            MagicMock(side_effect=HttpError(MagicMock(status=500), b"Server error")),
        )
        with pytest.raises(StorageOperationError):
            await storage.download(artifact, dest_path)

    # Generic unexpected exception -> StorageOperationError
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.storage.google_drive.MediaIoBaseDownload",
            MagicMock(side_effect=RuntimeError("Pipe broken")),
        )
        with pytest.raises(StorageOperationError) as exc_info:
            await storage.download(artifact, dest_path)
        assert "Pipe broken" in str(exc_info.value)


@pytest.mark.asyncio
async def test_drive_delete_http_errors_and_generic_error(mock_drive_service):
    mock_files = mock_drive_service.files.return_value
    storage = GoogleDriveStorage(root_folder_id="root-123", service=mock_drive_service)
    artifact = ArtifactRef(
        workflow_id=uuid4(),
        artifact_type="script",
        filename="output.txt",
        mime_type="text/plain",
        drive_file_id="file-id-123",
    )

    # 401/403 -> StorageAuthenticationError
    mock_files.delete.return_value.execute.side_effect = HttpError(
        MagicMock(status=401), b"Unauthorized"
    )
    with pytest.raises(StorageAuthenticationError):
        await storage.delete(artifact)

    # 500 -> StorageOperationError
    mock_files.delete.return_value.execute.side_effect = HttpError(
        MagicMock(status=500), b"Server error"
    )
    with pytest.raises(StorageOperationError):
        await storage.delete(artifact)

    # Generic unexpected exception -> StorageOperationError
    mock_files.delete.return_value.execute.side_effect = RuntimeError("Filesystem corrupt")
    with pytest.raises(StorageOperationError) as exc_info:
        await storage.delete(artifact)
    assert "Filesystem corrupt" in str(exc_info.value)
