class StorageError(Exception):
    """Base exception for all storage provider errors."""

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


class StorageConfigurationError(StorageError):
    """Configuration error for storage adapter (e.g. missing credentials or root folder ID)."""


class StorageAuthenticationError(StorageError):
    """Authentication or permission failure communicating with storage service."""


class StorageNotFoundError(StorageError):
    """Requested file or folder was not found in storage."""


class StorageDuplicateError(StorageError):
    """Attempted to upload a file that already exists without overwrite enabled."""


class StorageOperationError(StorageError):
    """General I/O or API failure during upload, download, or delete."""
