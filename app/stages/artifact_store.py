"""UUID-addressed, path-confined, immutable artifact storage.

This module implements the physical artifact storage layer with:
- Atomic race-free exclusive publication via os.link + os.unlink
- Complete-write loops guaranteeing all bytes are persisted
- Streaming write paths for bounded-memory handling of large media
- Chunked, bounded-memory SHA-256 integrity hashing
- Configurable fsync (file and directory) with strict error propagation
- Bounded thread pool executor combined with an in-flight capacity semaphore for backpressure
- Runtime UUID type validation and strict filesystem path confinement

Threat Model & Filesystem Security Assumptions:
------------------------------------------------
1. Service-Account Exclusivity: The artifact root directory is private infrastructure
   storage. Operating system permissions must restrict read/write access exclusively to the
   service daemon / worker OS account (e.g. chmod 700 / Windows restricted ACLs). Multi-tenant
   or untrusted OS users must NOT have write access to the directory tree.
2. Path Confinement: _validate_confinement() provides defense-in-depth against application-level
   path traversal vulnerabilities (traversal sequences '../', absolute path injection, or stage name
   tampering) by resolving canonical paths against the root.
3. Immutability & TOCTOU: Once published, artifact files are treated as immutable by the service.
   While high-level cross-platform APIs (Path.resolve, os.stat) cannot completely eliminate local
   symlink races (TOCTOU) if an attacker has concurrent local write privileges in the storage directory,
   the service-account exclusivity assumption ensures no untrusted local writer has access. On POSIX
   platforms with untrusted local actors, directory-descriptor-relative operations (openat with O_NOFOLLOW)
   would be required; here, isolation is enforced via OS-level directory ownership.
"""

import asyncio
import errno
import hashlib
import logging
import os
import sys
import tempfile
from collections.abc import AsyncIterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID

from app.stages.artifact_errors import ArtifactAlreadyExistsError, SecurityViolationError
from app.stages.dto import PipelineStage

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks for bounded-memory streaming


def _sync_dir_if_supported(dir_path: Path) -> None:
    """Sync directory entry to disk if supported by platform/filesystem.

    Durability guarantees and platform-specific error semantics:
    - On POSIX systems: directory metadata persistence requires calling fsync on
      the containing directory file descriptor. Genuine storage I/O errors
      (EIO, ENOSPC, EROFS, etc.) are strictly propagated. Filesystems explicitly
      reporting non-support (EINVAL, ENOTSUP, EOPNOTSUPP) are ignored as platform limits.
    - On Windows (NTFS): Directory file descriptors cannot be opened or fsynced
      via POSIX APIs (raising PermissionError or EINVAL). Therefore, directory fsync
      is omitted on Windows. Crash durability relies on file-level fsync and NTFS
      transaction/journal metadata handling. This provides weaker crash-consistency
      guarantees than a true POSIX directory fsync, as directory entry persistence
      remains subject to OS flush timing.
    """
    if sys.platform == "win32":
        return

    if not hasattr(os, "O_DIRECTORY"):
        return

    try:
        fd = os.open(str(dir_path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP), errno.EINVAL):
            return
        raise

    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno in (errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP), errno.EINVAL):
            return
        raise
    finally:
        os.close(fd)


class ArtifactStore:
    """UUID-addressed, path-confined, immutable artifact storage.

    Files are published atomically via same-directory tempfile + os.link,
    preventing race conditions and overwrites. Both thread execution and
    in-flight queued work are strictly bounded to prevent memory exhaustion.
    """

    ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"json", "mp4", "mp3", "wav", "png", "jpg"})
    DEFAULT_MAX_READ_BYTES: int = 32 * 1024 * 1024  # 32 MiB max for in-memory read_artifact

    def __init__(
        self,
        artifact_root: Path,
        *,
        fsync: bool = True,
        max_workers: int = 16,
        max_in_flight: int = 32,
    ) -> None:
        """Initialize with artifact root, durability, and concurrency policies.

        Args:
            artifact_root: Base directory for all artifacts. Must be an
                absolute path or will be resolved against CWD.
            fsync: If True, fsync the file and directory for crash durability.
                Set to False in test environments for performance.
            max_workers: Maximum threads in bounded pool for blocking I/O (must be > 0).
            max_in_flight: Maximum in-flight storage operations for backpressure (must be > 0).
        """
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got: {max_workers}")
        if max_in_flight <= 0:
            raise ValueError(f"max_in_flight must be positive, got: {max_in_flight}")

        self._artifact_root = self._normalize_confinement_path(artifact_root)
        self._fsync = fsync
        self._max_workers = max_workers
        self._max_in_flight = max_in_flight
        self._semaphore: asyncio.Semaphore | None = None
        self._cleanup_failure_count: int = 0
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="artifact-store",
        )

    @property
    def cleanup_failure_count(self) -> int:
        """Return total count of physical cleanup or file close failures encountered."""
        return self._cleanup_failure_count

    @property
    def artifact_root(self) -> Path:
        """Return the resolved artifact root path."""
        return self._artifact_root

    @property
    def in_flight_semaphore(self) -> asyncio.Semaphore:
        """Return lazily initialized semaphore for in-flight backpressure."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_in_flight)
        return self._semaphore

    def close(self) -> None:
        """Shutdown the underlying thread pool executor."""
        self._executor.shutdown(wait=False)

    def build_path(
        self,
        workflow_id: UUID,
        stage: PipelineStage,
        artifact_id: UUID,
        extension: str,
    ) -> Path:
        """Build and validate canonical artifact path.

        Path format: {artifact_root}/{workflow_id}/{stage}/{artifact_id}.{ext}
        """
        if not isinstance(workflow_id, UUID):
            raise TypeError(
                f"workflow_id must be a UUID instance, got: {type(workflow_id).__name__}"
            )

        if not isinstance(artifact_id, UUID):
            raise TypeError(
                f"artifact_id must be a UUID instance, got: {type(artifact_id).__name__}"
            )

        if extension not in self.ALLOWED_EXTENSIONS:
            raise SecurityViolationError(
                f"Extension '{extension}' is not in the allow-list: "
                f"{sorted(self.ALLOWED_EXTENSIONS)}"
            )

        if not isinstance(stage, PipelineStage):
            raise SecurityViolationError(
                f"Stage must be a PipelineStage enum member, got: {type(stage).__name__}"
            )

        target = self._artifact_root / str(workflow_id) / stage.value / f"{artifact_id}.{extension}"
        return self._validate_confinement(target)

    async def _run_in_executor_bounded(self, fn, *args):
        """Execute a blocking function in the executor with cancellation-safe backpressure.

        Acquires an in-flight semaphore permit before scheduling to the thread pool.
        The permit release is bound to the actual completion of the thread pool future,
        guaranteeing that task cancellation will not prematurely release capacity while
        the underlying filesystem thread is still executing.
        """
        await self.in_flight_semaphore.acquire()
        loop = asyncio.get_running_loop()
        try:
            fut = self._executor.submit(fn, *args)
        except Exception:
            self.in_flight_semaphore.release()
            raise
        fut.add_done_callback(lambda _: loop.call_soon_threadsafe(self.in_flight_semaphore.release))
        return await self._await_future_cancellation_safe(fut)

    async def _await_future_cancellation_safe(self, fut):
        """Await an executor future to completion, even if the awaiting coroutine is cancelled.

        Guarantees that cancellation cannot race with the underlying thread operation:
        if cancelled while the thread is executing, shields waiting for the thread to complete
        before re-raising CancelledError. Survives repeated cancellations.
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wrap_future(fut)
        except asyncio.CancelledError:
            if not fut.done():
                done_event = asyncio.Event()
                fut.add_done_callback(lambda _: loop.call_soon_threadsafe(done_event.set))
                while not fut.done():
                    try:
                        await asyncio.shield(done_event.wait())
                    except asyncio.CancelledError:
                        pass
            raise

    async def write_artifact(
        self,
        path: Path,
        content: bytes,
    ) -> tuple[str, int]:
        """Write content to path atomically with SHA-256 integrity hash.

        Acquires an in-flight semaphore permit to bound queued work, then runs
        blocking filesystem I/O in the thread pool.
        """
        resolved = self._validate_confinement(path)
        return await self._run_in_executor_bounded(self._write_artifact_sync, resolved, content)

    async def write_artifact_stream(
        self,
        path: Path,
        stream: AsyncIterable[bytes],
    ) -> tuple[str, int]:
        """Stream chunks into an artifact path atomically with bounded memory.

        Consumes chunks from an async byte iterator (e.g. video generator or HTTP stream)
        and writes incrementally to a temp file, hashing on-the-fly.

        All blocking filesystem operations (directory creation, mkstemp, writes, fsync,
        and cleanup) are performed inside the thread pool executor to avoid event loop stalls.
        Cancellation safety is strictly enforced: if cancelled while a worker thread is active,
        the method shields waiting for the thread operation to complete before initiating cleanup.
        This eliminates descriptor leaks during setup, concurrent write/close races, and premature
        permit release.
        """
        resolved = self._validate_confinement(path)
        await self.in_flight_semaphore.acquire()
        permit_released = False

        def _safe_release_permit():
            nonlocal permit_released
            if not permit_released:
                permit_released = True
                self.in_flight_semaphore.release()

        temp_fd: int | None = None
        temp_path: str | None = None
        hasher = hashlib.sha256()
        total_written = 0

        try:
            try:
                # 1. Setup temp file with cancellation safety
                setup_fut = self._executor.submit(self._setup_temp_file_sync, resolved)
                try:
                    temp_fd, temp_path = await self._await_future_cancellation_safe(setup_fut)
                except (asyncio.CancelledError, Exception):
                    if setup_fut.done() and not setup_fut.exception():
                        temp_fd, temp_path = setup_fut.result()
                    raise

                # 2. Write chunks incrementally with cancellation safety
                async for chunk in stream:
                    if not chunk:
                        continue
                    write_fut = self._executor.submit(self._write_chunk_sync, temp_fd, chunk)
                    written = await self._await_future_cancellation_safe(write_fut)
                    hasher.update(chunk)
                    total_written += written

                # 3. Atomically publish temp file
                publish_fut = self._executor.submit(
                    self._publish_temp_sync, temp_fd, temp_path, resolved
                )
                await self._await_future_cancellation_safe(publish_fut)
                return hasher.hexdigest(), total_written
            finally:
                if temp_fd is not None or temp_path is not None:
                    try:
                        cleanup_fut = self._executor.submit(
                            self._cleanup_temp_file_sync, temp_fd, temp_path
                        )
                        await self._await_future_cancellation_safe(cleanup_fut)
                    except (RuntimeError, OSError) as exc:
                        logger.warning(
                            "Executor unavailable during stream cleanup; executing synchronous cleanup: %s",
                            exc,
                        )
                        try:
                            self._cleanup_temp_file_sync(temp_fd, temp_path)
                        except Exception as sync_exc:  # noqa: BLE001
                            self._cleanup_failure_count += 1
                            logger.error("Synchronous cleanup failed: %s", sync_exc)
                    except Exception as unexpected:  # noqa: BLE001
                        self._cleanup_failure_count += 1
                        logger.error("Unexpected error during stream cleanup: %s", unexpected)
        finally:
            _safe_release_permit()

    async def write_artifact_from_file(
        self,
        path: Path,
        source_path: Path,
    ) -> tuple[str, int]:
        """Stream an existing local file into the artifact store in 1 MB chunks."""
        resolved = self._validate_confinement(path)
        return await self._run_in_executor_bounded(
            self._write_from_file_sync, resolved, source_path
        )

    def _setup_temp_file_sync(self, resolved: Path) -> tuple[int, str]:
        """Synchronously check existence, create directory, and allocate temp file."""
        if resolved.exists():
            raise ArtifactAlreadyExistsError(str(resolved))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(resolved.parent),
            prefix=".artifact_",
            suffix=".tmp",
        )
        return temp_fd, temp_path

    def _cleanup_temp_file_sync(self, temp_fd: int | None, temp_path: str | None) -> None:
        """Safely close temp file descriptor once and unlink temp path with failure tracking."""
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError as exc:
                self._cleanup_failure_count += 1
                logger.warning("Failed to close temp fd %s: %s", temp_fd, exc)
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except OSError as exc:
                self._cleanup_failure_count += 1
                logger.warning("Failed to unlink temp path %s: %s", temp_path, exc)

    def _write_chunk_sync(self, temp_fd: int, chunk: bytes) -> int:
        """Synchronously write a byte chunk using a complete-write loop."""
        mv = memoryview(chunk)
        written = 0
        while written < len(chunk):
            n = os.write(temp_fd, mv[written:])
            if n == 0:
                raise OSError("os.write returned 0 bytes during chunk write")
            written += n
        return written

    def _publish_temp_sync(self, temp_fd: int, temp_path: str, resolved: Path) -> None:
        """Sync temp file, atomically hard-link, and sync directory without closing temp_fd.

        temp_fd ownership remains with the caller's finally block to prevent double-closure.
        """
        if self._fsync:
            os.fsync(temp_fd)

        try:
            os.link(temp_path, str(resolved))
        except FileExistsError:
            raise ArtifactAlreadyExistsError(str(resolved)) from None

        if self._fsync:
            _sync_dir_if_supported(resolved.parent)

    def _write_from_file_sync(self, resolved: Path, source_path: Path) -> tuple[str, int]:
        """Stream source file into temp file in 1 MB chunks and publish atomically."""
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        temp_fd: int | None = None
        temp_path: str | None = None
        try:
            temp_fd, temp_path = self._setup_temp_file_sync(resolved)
            hasher = hashlib.sha256()
            total_written = 0
            with open(source_path, "rb") as sf:
                while True:
                    buf = sf.read(_HASH_CHUNK_SIZE)
                    if not buf:
                        break
                    self._write_chunk_sync(temp_fd, buf)
                    hasher.update(buf)
                    total_written += len(buf)

            self._publish_temp_sync(temp_fd, temp_path, resolved)
            return hasher.hexdigest(), total_written
        finally:
            self._cleanup_temp_file_sync(temp_fd, temp_path)

    def _write_artifact_sync(self, resolved: Path, content: bytes) -> tuple[str, int]:
        """Synchronous implementation of write_artifact running in thread pool."""
        temp_fd: int | None = None
        temp_path: str | None = None
        try:
            temp_fd, temp_path = self._setup_temp_file_sync(resolved)
            hasher = hashlib.sha256()
            mv = memoryview(content)
            total_written = 0

            while total_written < len(content):
                chunk = mv[total_written:]
                written = os.write(temp_fd, chunk)
                if written == 0:
                    raise OSError("os.write returned 0 bytes during artifact publication")
                hasher.update(mv[total_written : total_written + written])
                total_written += written

            if total_written != len(content):
                raise OSError(
                    f"Incomplete write: expected {len(content)} bytes, wrote {total_written}"
                )

            self._publish_temp_sync(temp_fd, temp_path, resolved)
            return hasher.hexdigest(), total_written
        finally:
            self._cleanup_temp_file_sync(temp_fd, temp_path)

    async def read_artifact(
        self, path: Path, *, max_bytes: int | None = DEFAULT_MAX_READ_BYTES
    ) -> bytes:
        """Read and return artifact content after verifying confinement and size limit.

        Args:
            path: Path to the artifact file.
            max_bytes: Maximum allowed size for in-memory read (default 32 MiB).
                Set to None to allow arbitrary sizes. Use read_artifact_stream()
                for large media to maintain bounded memory footprint.
        """
        resolved = self._validate_confinement(path)
        return await self._run_in_executor_bounded(self._read_artifact_sync, resolved, max_bytes)

    def _read_artifact_sync(self, resolved: Path, max_bytes: int | None = None) -> bytes:
        """Synchronous read running in thread pool with optional size guard."""
        if not resolved.exists():
            raise FileNotFoundError(f"Artifact not found: {resolved}")
        size = resolved.stat().st_size
        if max_bytes is not None and size > max_bytes:
            raise ValueError(
                f"Artifact size ({size} bytes) exceeds maximum in-memory read limit ({max_bytes} bytes). "
                f"Use read_artifact_stream() for large media artifacts."
            )
        return resolved.read_bytes()

    async def read_artifact_stream(
        self,
        path: Path,
        chunk_size: int = _HASH_CHUNK_SIZE,
    ) -> AsyncIterable[bytes]:
        """Stream artifact content in bounded chunks for large media artifacts."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got: {chunk_size}")

        resolved = self._validate_confinement(path)
        await self.in_flight_semaphore.acquire()
        permit_released = False

        def _safe_release():
            nonlocal permit_released
            if not permit_released:
                permit_released = True
                self.in_flight_semaphore.release()

        file_obj = None
        try:
            try:
                # 1. Open file with cancellation safety
                open_fut = self._executor.submit(self._open_file_read_sync, resolved)
                try:
                    file_obj = await self._await_future_cancellation_safe(open_fut)
                except (asyncio.CancelledError, Exception):
                    if open_fut.done() and not open_fut.exception():
                        file_obj = open_fut.result()
                    raise

                # 2. Stream chunks with cancellation safety
                while True:
                    read_fut = self._executor.submit(self._read_file_sync, file_obj, chunk_size)
                    chunk = await self._await_future_cancellation_safe(read_fut)
                    if not chunk:
                        break
                    yield chunk
            finally:
                if file_obj is not None:
                    try:
                        close_fut = self._executor.submit(self._close_file_sync, file_obj)
                        await self._await_future_cancellation_safe(close_fut)
                    except (RuntimeError, OSError) as exc:
                        logger.warning(
                            "Executor unavailable during stream read close; executing synchronous close: %s",
                            exc,
                        )
                        try:
                            self._close_file_sync(file_obj)
                        except Exception as sync_exc:  # noqa: BLE001
                            self._cleanup_failure_count += 1
                            logger.error("Synchronous file close failed: %s", sync_exc)
                    except Exception as unexpected:  # noqa: BLE001
                        self._cleanup_failure_count += 1
                        logger.error("Unexpected error during stream read close: %s", unexpected)
        finally:
            _safe_release()

    def _open_file_read_sync(self, resolved: Path):
        """Open artifact file for reading in binary mode."""
        if not resolved.exists():
            raise FileNotFoundError(f"Artifact not found: {resolved}")
        return open(resolved, "rb")

    def _read_file_sync(self, file_obj, chunk_size: int) -> bytes:
        """Read a single chunk from file."""
        return file_obj.read(chunk_size)

    def _close_file_sync(self, file_obj) -> None:
        """Close opened file object with failure tracking."""
        try:
            file_obj.close()
        except Exception as exc:
            self._cleanup_failure_count += 1
            logger.warning("Failed to close file object: %s", exc)
            raise

    async def compute_checksum(self, path: Path) -> tuple[str, int]:
        """Independently compute SHA-256 checksum and size using chunked reads."""
        resolved = self._validate_confinement(path)
        return await self._run_in_executor_bounded(self._compute_checksum_sync, resolved)

    def _compute_checksum_sync(self, resolved: Path) -> tuple[str, int]:
        """Compute SHA-256 with bounded memory usage using chunked streaming."""
        if not resolved.exists():
            raise FileNotFoundError(f"Artifact not found: {resolved}")

        hasher = hashlib.sha256()
        total_size = 0
        with open(resolved, "rb") as f:
            while True:
                chunk = f.read(_HASH_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                total_size += len(chunk)

        return hasher.hexdigest(), total_size

    @staticmethod
    def _normalize_confinement_path(p: Path) -> Path:
        """Resolve path and strip Windows extended-length prefix (\\?\\) if present."""
        resolved = p.resolve()
        s = str(resolved)
        if s.startswith("\\\\?\\"):
            return Path(s[4:])
        return resolved

    def _validate_confinement(self, path: Path) -> Path:
        """Resolve path and verify it is within artifact_root."""
        resolved = self._normalize_confinement_path(path)
        if not resolved.is_relative_to(self._artifact_root):
            raise SecurityViolationError(
                f"Path '{resolved}' is outside artifact root '{self._artifact_root}'"
            )
        return resolved


__all__ = ["ArtifactStore"]
