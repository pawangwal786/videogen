from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.models.errors import ModelConfigurationError
from app.models.router import ModelProvider
from app.storage.errors import StorageConfigurationError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    videogen_env: str = "development"
    videogen_log_level: str = "INFO"

    # Gemini
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    veo_model: str = "veo-2.0-generate-001"

    # OpenRouter
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-2.5-flash"

    # Google Drive
    google_drive_client_id: SecretStr | None = None
    google_drive_client_secret: SecretStr | None = None
    google_drive_refresh_token: SecretStr | None = None
    google_drive_root_folder_id: str | None = None

    # Model Routing
    videogen_primary_text_provider: ModelProvider = ModelProvider.GEMINI
    videogen_fallback_text_provider: ModelProvider | None = ModelProvider.OPENROUTER

    # Media Processing (FFmpeg / MPT)
    ffmpeg_binary: str = Field(default="ffmpeg", min_length=1)
    ffprobe_binary: str = Field(default="ffprobe", min_length=1)
    media_max_concurrency: int = Field(default=2, ge=1)
    media_assembly_timeout_seconds: float = Field(default=300.0, gt=0.0)
    media_target_fps: int = Field(default=30, ge=1)
    media_video_codec: str = Field(default="libx264", min_length=1)
    media_pixel_format: str = Field(default="yuv420p", min_length=1)
    media_audio_codec: str = Field(default="aac", min_length=1)
    media_duration_tolerance_seconds: float = Field(default=0.5, ge=0.0)

    @field_validator(
        "ffmpeg_binary",
        "ffprobe_binary",
        "media_video_codec",
        "media_pixel_format",
        "media_audio_codec",
        "database_url",
    )
    @classmethod
    def validate_non_whitespace_string(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Configuration string must not be empty or whitespace only")
        return s

    # Database & Persistence
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/videogen",
        min_length=1,
        validation_alias=AliasChoices(
            "VIDEOGEN_DATABASE_URL",
            "DATABASE_URL",
            "videogen_database_url",
            "database_url",
        ),
    )
    test_database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VIDEOGEN_TEST_DATABASE_URL",
            "TEST_DATABASE_URL",
            "videogen_test_database_url",
            "test_database_url",
        ),
    )
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)
    database_pool_timeout: float = Field(default=30.0, gt=0.0)
    database_pool_recycle: int = Field(default=1800, ge=60)

    # Orchestration & Worker Runtime
    orchestrator_worker_lease_seconds: float = Field(default=60.0, gt=0.0)
    orchestrator_heartbeat_interval_seconds: float = Field(default=15.0, gt=0.0)
    orchestrator_max_retries: int = Field(default=3, ge=0)

    # Integration testing safety
    videogen_run_external_tests: bool = False

    def get_database_url(self, for_test: bool = False) -> str:
        """Resolve the effective database URL, prioritizing test database URL if requested."""
        if for_test:
            return self.test_database_url or self.database_url
        return self.database_url

    def require_gemini(self) -> str:
        """Validate and return the Gemini API key, or raise ModelConfigurationError."""
        if not self.gemini_api_key or not self.gemini_api_key.get_secret_value().strip():
            raise ModelConfigurationError(
                "GEMINI_API_KEY is required when Gemini provider is enabled.",
                provider="gemini",
            )
        return self.gemini_api_key.get_secret_value().strip()

    def require_openrouter(self) -> tuple[str, str, str]:
        """Validate and return (api_key, base_url, model) or raise ModelConfigurationError."""
        if not self.openrouter_api_key or not self.openrouter_api_key.get_secret_value().strip():
            raise ModelConfigurationError(
                "OPENROUTER_API_KEY is required when OpenRouter provider is enabled.",
                provider="openrouter",
            )
        return (
            self.openrouter_api_key.get_secret_value().strip(),
            self.openrouter_base_url.strip(),
            self.openrouter_model.strip(),
        )

    def require_google_drive(self) -> tuple[str, str, str, str]:
        """Validate and return (client_id, client_secret, refresh_token, root_folder_id)

        or raise StorageConfigurationError.
        """
        missing: list[str] = []
        if (
            not self.google_drive_client_id
            or not self.google_drive_client_id.get_secret_value().strip()
        ):
            missing.append("GOOGLE_DRIVE_CLIENT_ID")
        if (
            not self.google_drive_client_secret
            or not self.google_drive_client_secret.get_secret_value().strip()
        ):
            missing.append("GOOGLE_DRIVE_CLIENT_SECRET")
        if (
            not self.google_drive_refresh_token
            or not self.google_drive_refresh_token.get_secret_value().strip()
        ):
            missing.append("GOOGLE_DRIVE_REFRESH_TOKEN")
        if not self.google_drive_root_folder_id or not self.google_drive_root_folder_id.strip():
            missing.append("GOOGLE_DRIVE_ROOT_FOLDER_ID")

        if missing:
            raise StorageConfigurationError(
                f"Missing required Google Drive configuration: {', '.join(missing)}."
            )

        return (
            self.google_drive_client_id.get_secret_value().strip(),  # type: ignore[union-attr]
            self.google_drive_client_secret.get_secret_value().strip(),  # type: ignore[union-attr]
            self.google_drive_refresh_token.get_secret_value().strip(),  # type: ignore[union-attr]
            self.google_drive_root_folder_id.strip(),  # type: ignore[union-attr]
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
