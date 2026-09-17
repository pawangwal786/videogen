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
    assert settings.ffmpeg_binary == "ffmpeg"
    assert settings.ffprobe_binary == "ffprobe"
    assert settings.media_max_concurrency == 2
    assert settings.media_assembly_timeout_seconds == 300.0
    assert settings.media_target_fps == 30
    assert settings.media_video_codec == "libx264"
    assert settings.media_pixel_format == "yuv420p"
    assert settings.media_audio_codec == "aac"
    assert settings.media_duration_tolerance_seconds == 0.5


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


def test_media_settings_validation_errors():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="media_max_concurrency"):
        Settings(_env_file=None, media_max_concurrency=0)

    with pytest.raises(ValidationError, match="media_assembly_timeout_seconds"):
        Settings(_env_file=None, media_assembly_timeout_seconds=0.0)

    with pytest.raises(ValidationError, match="media_assembly_timeout_seconds"):
        Settings(_env_file=None, media_assembly_timeout_seconds=-10.0)

    with pytest.raises(ValidationError, match="media_target_fps"):
        Settings(_env_file=None, media_target_fps=0)

    with pytest.raises(ValidationError, match="empty or whitespace only"):
        Settings(_env_file=None, ffmpeg_binary="   ")

    with pytest.raises(ValidationError, match="media_video_codec"):
        Settings(_env_file=None, media_video_codec="")

    with pytest.raises(ValidationError, match="empty or whitespace only"):
        Settings(_env_file=None, media_video_codec="   ")


def test_database_settings_defaults_and_methods():
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.test_database_url is None
    assert settings.database_pool_size == 5
    assert settings.database_max_overflow == 10
    assert settings.database_pool_timeout == 30.0
    assert settings.database_pool_recycle == 1800
    assert settings.orchestrator_worker_lease_seconds == 60.0
    assert settings.orchestrator_heartbeat_interval_seconds == 15.0
    assert settings.orchestrator_max_retries == 3

    # get_database_url without test url
    assert settings.get_database_url(for_test=False) == settings.database_url
    assert settings.get_database_url(for_test=True) == settings.database_url

    # with test url
    test_settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://user:pass@localhost:5432/main",
        test_database_url="postgresql+asyncpg://user:pass@localhost:5432/test",
    )
    assert (
        test_settings.get_database_url(for_test=False)
        == "postgresql+asyncpg://user:pass@localhost:5432/main"
    )
    assert (
        test_settings.get_database_url(for_test=True)
        == "postgresql+asyncpg://user:pass@localhost:5432/test"
    )


def test_database_settings_validation_errors():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None, database_url="")

    with pytest.raises(ValidationError, match="empty or whitespace only"):
        Settings(_env_file=None, database_url="   ")

    with pytest.raises(ValidationError, match="database_pool_size"):
        Settings(_env_file=None, database_pool_size=0)

    with pytest.raises(ValidationError, match="database_max_overflow"):
        Settings(_env_file=None, database_max_overflow=-1)

    with pytest.raises(ValidationError, match="database_pool_timeout"):
        Settings(_env_file=None, database_pool_timeout=0.0)

    with pytest.raises(ValidationError, match="database_pool_recycle"):
        Settings(_env_file=None, database_pool_recycle=30)


def test_production_auth_validation_fail_closed():
    """Verify that VIDEOGEN_ENV=production fails closed when API auth token is missing or whitespace."""
    from pydantic import ValidationError

    # Production with missing token
    with pytest.raises(
        ValidationError, match="VIDEOGEN_API_AUTH_TOKEN is required when VIDEOGEN_ENV=production"
    ):
        Settings(_env_file=None, videogen_env="production", api_auth_token=None)

    # Production with whitespace token
    with pytest.raises(
        ValidationError, match="VIDEOGEN_API_AUTH_TOKEN is required when VIDEOGEN_ENV=production"
    ):
        Settings(_env_file=None, videogen_env="production", api_auth_token=SecretStr("   "))

    # Production with valid token succeeds
    prod_settings = Settings(
        _env_file=None,
        videogen_env="production",
        api_auth_token=SecretStr("prod-secret-token-12345"),
    )
    assert prod_settings.api_auth_token is not None
    assert prod_settings.api_auth_token.get_secret_value() == "prod-secret-token-12345"

    # Development and test environments remain permissive when token is omitted
    dev_settings = Settings(_env_file=None, videogen_env="development", api_auth_token=None)
    assert dev_settings.api_auth_token is None

    test_settings = Settings(_env_file=None, videogen_env="test", api_auth_token=None)
    assert test_settings.api_auth_token is None


def test_production_auth_via_environment_variables(monkeypatch: pytest.MonkeyPatch):
    """Verify production fail-closed behavior when loaded via OS environment variables."""
    from pydantic import ValidationError

    monkeypatch.setenv("VIDEOGEN_ENV", "production")
    monkeypatch.delenv("VIDEOGEN_API_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("API_AUTH_TOKEN", raising=False)

    with pytest.raises(
        ValidationError, match="VIDEOGEN_API_AUTH_TOKEN is required when VIDEOGEN_ENV=production"
    ):
        Settings(_env_file=None)

    monkeypatch.setenv("VIDEOGEN_API_AUTH_TOKEN", "valid-env-secret")
    s = Settings(_env_file=None)
    assert s.api_auth_token is not None
    assert s.api_auth_token.get_secret_value() == "valid-env-secret"
