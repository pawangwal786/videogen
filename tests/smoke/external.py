"""Controlled external smoke test utility.

Executes real external API calls against configured services (Gemini, OpenRouter,
and Google Drive) when credentials are provided in `.env`.

Usage:
    python -m tests.smoke.external
"""

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from app.config.settings import Settings
from app.models.errors import ModelConfigurationError, ModelError
from app.models.gemini import GeminiTextModel
from app.models.openrouter import OpenRouterTextModel
from app.storage.errors import StorageConfigurationError, StorageError
from app.storage.google_drive import GoogleDriveStorage


async def smoke_gemini(settings: Settings) -> bool:
    print("\n--- Testing Gemini Provider ---")
    try:
        api_key = settings.require_gemini()
    except ModelConfigurationError as e:
        print(f"[SKIP] Gemini credentials missing: {e}")
        return True

    try:
        model = GeminiTextModel(
            api_key=api_key,
            model_name=settings.gemini_model,
            timeout_seconds=30.0,
        )
        print(f"Sending prompt to Gemini ({settings.gemini_model})...")
        response = await model.generate("Say 'PONG' and nothing else.")
        print(f"[PASS] Gemini response received: {response!r}")
        return True
    except ModelError as e:
        print(f"[FAIL] Gemini error: {e}")
        return False


async def smoke_openrouter(settings: Settings) -> bool:
    print("\n--- Testing OpenRouter Provider ---")
    try:
        api_key, base_url, model_name = settings.require_openrouter()
    except ModelConfigurationError as e:
        print(f"[SKIP] OpenRouter credentials missing: {e}")
        return True

    try:
        model = OpenRouterTextModel(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            timeout_seconds=30.0,
        )
        print(f"Sending prompt to OpenRouter ({model_name})...")
        response = await model.generate("Say 'PONG' and nothing else.")
        print(f"[PASS] OpenRouter response received: {response!r}")
        return True
    except ModelError as e:
        print(f"[FAIL] OpenRouter error: {e}")
        return False


async def smoke_google_drive(settings: Settings) -> bool:
    print("\n--- Testing Google Drive Storage ---")
    try:
        settings.require_google_drive()
    except StorageConfigurationError as e:
        print(f"[SKIP] Google Drive credentials missing: {e}")
        return True

    temp_dir = Path(tempfile.mkdtemp())
    try:
        storage = GoogleDriveStorage.from_settings(settings)
        sample_file = temp_dir / "smoke_artifact.txt"
        test_content = f"VideoGen smoke test payload - {uuid4()}"
        sample_file.write_text(test_content, encoding="utf-8")

        wf_id = uuid4()
        print(f"1. Uploading smoke artifact to workflow {wf_id}...")
        artifact = await storage.upload(
            sample_file,
            destination="smoke_artifact.txt",
            workflow_id=wf_id,
            artifact_type="smoke",
            overwrite=True,
        )
        print(
            f"   Uploaded! Drive file ID: {artifact.drive_file_id}, Size: {artifact.size_bytes} bytes"
        )

        print("2. Downloading smoke artifact...")
        download_dest = temp_dir / "smoke_downloaded.txt"
        await storage.download(artifact, download_dest)
        downloaded_content = download_dest.read_text(encoding="utf-8")
        assert downloaded_content == test_content, "Downloaded content mismatch!"
        print("   Verified downloaded content matches uploaded payload.")

        print("3. Deleting smoke artifact...")
        await storage.delete(artifact)
        print("   Deleted from Google Drive.")

        print("[PASS] Google Drive end-to-end operation successful.")
        return True
    except (StorageError, AssertionError) as e:
        print(f"[FAIL] Google Drive error: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main() -> int:
    print("==================================================")
    print("VideoGen Controlled External Smoke Test")
    print("==================================================")
    settings = Settings()

    gemini_ok = await smoke_gemini(settings)
    openrouter_ok = await smoke_openrouter(settings)
    drive_ok = await smoke_google_drive(settings)

    print("\n==================================================")
    if gemini_ok and openrouter_ok and drive_ok:
        print("Overall Smoke Status: ALL ATTEMPTED CHECKS PASSED")
        return 0
    else:
        print("Overall Smoke Status: ONE OR MORE CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
