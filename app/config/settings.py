from functools import lru_cache

from pydantic import SecretStr
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

    # Integration testing safety
    videogen_run_external_tests: bool = False

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
