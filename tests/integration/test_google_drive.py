import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.config.settings import Settings
from app.storage.google_drive import GoogleDriveStorage

RUN_EXTERNAL = os.getenv("VIDEOGEN_RUN_EXTERNAL_TESTS", "false").lower() in (
    "true",
    "1",
)


@pytest.mark.skipif(
    not RUN_EXTERNAL,
    reason="External integration tests are disabled by default. Set VIDEOGEN_RUN_EXTERNAL_TESTS=true to enable.",
)
@pytest.mark.asyncio
async def test_live_google_drive_storage(tmp_path: Path):
    settings = Settings()
    storage = GoogleDriveStorage.from_settings(settings)

    test_file = tmp_path / "live_test.txt"
    test_file.write_text("Integration test artifact content", encoding="utf-8")

    wf_id = uuid4()
    # 1. Upload
    artifact = await storage.upload(
        test_file,
        destination="live_test.txt",
        workflow_id=wf_id,
        artifact_type="test",
        overwrite=True,
    )
    assert artifact.drive_file_id

    # 2. Download
    download_dest = tmp_path / "downloaded_live.txt"
    await storage.download(artifact, download_dest)
    assert download_dest.is_file()
    assert download_dest.read_text(encoding="utf-8") == "Integration test artifact content"

    # 3. Delete
    await storage.delete(artifact)
