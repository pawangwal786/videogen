import pytest
from pydantic import SecretStr

from app.config.settings import Settings
from app.models.errors import ModelConfigurationError
from app.models.router import ModelProvider
from app.storage.errors import StorageConfigurationError


def test_settings_defaults():
    settings = Settings(
        _env_file=None,  # ignore any local .env during testing
    )
    assert settings.videogen_env == "development"
    assert settings.videogen_log_level == "INFO"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.videogen_primary_text_provider == ModelProvider.GEMINI
    assert settings.videogen_fallback_text_provider == ModelProvider.OPENROUTER
    assert settings.videogen_run_external_tests is False


def test_require_gemini_success():
    settings = Settings(
        _env_file=None,
        gemini_api_key=SecretStr("test-gemini-key"),
    )
    assert settings.require_gemini() == "test-gemini-key"


def test_require_gemini_missing():
    settings = Settings(_env_file=None, gemini_api_key=None)
    with pytest.raises(ModelConfigurationError) as exc_info:
        settings.require_gemini()
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.retryable is False


def test_require_openrouter_success():
    settings = Settings(
        _env_file=None,
        openrouter_api_key=SecretStr("test-openrouter-key"),
        openrouter_base_url="https://custom.openrouter.ai/api/v1",
        openrouter_model="custom/model",
    )
    key, url, model = settings.require_openrouter()
    assert key == "test-openrouter-key"
    assert url == "https://custom.openrouter.ai/api/v1"
    assert model == "custom/model"


def test_require_openrouter_missing():
    settings = Settings(_env_file=None, openrouter_api_key=None)
    with pytest.raises(ModelConfigurationError) as exc_info:
        settings.require_openrouter()
    assert exc_info.value.provider == "openrouter"


def test_require_google_drive_success():
    settings = Settings(
        _env_file=None,
        google_drive_client_id=SecretStr("client-id-123"),
        google_drive_client_secret=SecretStr("client-secret-456"),
        google_drive_refresh_token=SecretStr("refresh-token-789"),
        google_drive_root_folder_id="folder-root-000",
    )
    cid, sec, tok, root = settings.require_google_drive()
    assert cid == "client-id-123"
    assert sec == "client-secret-456"
    assert tok == "refresh-token-789"
    assert root == "folder-root-000"


def test_require_google_drive_missing():
    settings = Settings(
        _env_file=None,
        google_drive_client_id=SecretStr("client-id-123"),
        google_drive_client_secret=None,
        google_drive_refresh_token=None,
        google_drive_root_folder_id=None,
    )
    with pytest.raises(StorageConfigurationError) as exc_info:
        settings.require_google_drive()
    assert "GOOGLE_DRIVE_CLIENT_SECRET" in str(exc_info.value)
    assert "GOOGLE_DRIVE_REFRESH_TOKEN" in str(exc_info.value)
    assert "GOOGLE_DRIVE_ROOT_FOLDER_ID" in str(exc_info.value)
