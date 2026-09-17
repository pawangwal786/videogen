from functools import lru_cache
from typing import Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url

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

    # API & Control Plane
    api_auth_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "VIDEOGEN_API_AUTH_TOKEN",
            "API_AUTH_TOKEN",
            "videogen_api_auth_token",
            "api_auth_token",
        ),
    )
    api_title: str = "VideoGen API"
    api_version: str = "0.1.0"
    api_rate_limit_per_minute: int = Field(default=60, ge=1)
    api_max_request_body_bytes: int = Field(default=1_048_576, ge=1024)

    @model_validator(mode="after")
    def validate_production_security(self) -> Self:
        allowed_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"}
        log_level = self.videogen_log_level.strip().upper()
        if log_level not in allowed_log_levels:
            raise ValueError(
                f"Invalid VIDEOGEN_LOG_LEVEL '{self.videogen_log_level}'. "
                f"Must be one of {sorted(allowed_log_levels)}."
            )

        if self.videogen_env.lower() == "production":
            # 1. API auth token enforcement
            if self.api_auth_token is None or not self.api_auth_token.get_secret_value().strip():
                raise ValueError(
                    "VIDEOGEN_API_AUTH_TOKEN is required when VIDEOGEN_ENV=production. "
                    "Production cannot run with authentication disabled."
                )

            # 2. Database URL: structural parsing
            try:
                db_url = make_url(self.database_url)
            except Exception as exc:
                raise ValueError(f"Invalid production database URL: {exc}") from exc

            host = (db_url.host or "").lower().strip("[]")
            if host in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError(
                    f"Production database URL cannot connect to loopback/local host '{db_url.host}'. "
                    "Configure a dedicated managed database host for production."
                )

            # 3. Google Drive storage tri-state validation in production (disabled, fully enabled, or rejected partial)
            drive_fields = [
                self.google_drive_client_id
                and self.google_drive_client_id.get_secret_value().strip(),
                self.google_drive_client_secret
                and self.google_drive_client_secret.get_secret_value().strip(),
                self.google_drive_refresh_token
                and self.google_drive_refresh_token.get_secret_value().strip(),
                self.google_drive_root_folder_id and self.google_drive_root_folder_id.strip(),
            ]
            configured_drive_count = sum(1 for f in drive_fields if bool(f))
            if 0 < configured_drive_count < 4:
                raise ValueError(
                    "Google Drive storage is partially configured. All four settings "
                    "(GOOGLE_DRIVE_CLIENT_ID, GOOGLE_DRIVE_CLIENT_SECRET, GOOGLE_DRIVE_REFRESH_TOKEN, "
                    "GOOGLE_DRIVE_ROOT_FOLDER_ID) must be provided together, or all omitted to disable."
                )

            user = (db_url.username or "").lower()
            password = (db_url.password or "").lower()
            if user == "postgres" and password == "postgres":
                raise ValueError(
                    "Production database URL cannot use default 'postgres:postgres' credentials."
                )

            # 3. Log level restrictions
            if log_level in {"DEBUG", "TRACE"}:
                raise ValueError(
                    f"VIDEOGEN_LOG_LEVEL cannot be set to '{log_level}' in production. "
                    "Use INFO, WARNING, ERROR, or CRITICAL."
                )

            # 4. Text provider validation for all enabled providers (primary + fallback)
            enabled_providers = {self.videogen_primary_text_provider}
            if self.videogen_fallback_text_provider is not None:
                enabled_providers.add(self.videogen_fallback_text_provider)

            if ModelProvider.GEMINI in enabled_providers and (
                not self.gemini_api_key or not self.gemini_api_key.get_secret_value().strip()
            ):
                raise ValueError(
                    "GEMINI_API_KEY is required in production when Gemini is configured as primary or fallback provider."
                )

            if ModelProvider.OPENROUTER in enabled_providers and (
                not self.openrouter_api_key
                or not self.openrouter_api_key.get_secret_value().strip()
            ):
                raise ValueError(
                    "OPENROUTER_API_KEY is required in production when OpenRouter is configured as primary or fallback provider."
                )

        return self

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
