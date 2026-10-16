"""Unit tests for ArtifactStore."""

import asyncio
import errno
import hashlib
import os
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from app.stages.artifact_errors import ArtifactAlreadyExistsError, SecurityViolationError
from app.stages.artifact_store import ArtifactStore, _sync_dir_if_supported
from app.stages.dto import PipelineStage


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    return root


@pytest.fixture
def store(artifact_root: Path) -> ArtifactStore:
    s = ArtifactStore(artifact_root, fsync=False)
    yield s
    s.close()


def test_init_config_validation(artifact_root: Path):
    """P2 test: max_workers and max_in_flight must be strictly positive integers."""
    with pytest.raises(ValueError, match="max_workers must be positive"):
        ArtifactStore(artifact_root, max_workers=0)

    with pytest.raises(ValueError, match="max_workers must be positive"):
        ArtifactStore(artifact_root, max_workers=-1)

    with pytest.raises(ValueError, match="max_in_flight must be positive"):
        ArtifactStore(artifact_root, max_in_flight=0)

    with pytest.raises(ValueError, match="max_in_flight must be positive"):
        ArtifactStore(artifact_root, max_in_flight=-10)


def test_build_path_valid(store: ArtifactStore, artifact_root: Path):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    stage = PipelineStage.RESEARCH

    path = store.build_path(wf_id, stage, art_id, "json")
    expected = artifact_root / str(wf_id) / "RESEARCH" / f"{art_id}.json"
    assert path == expected


def test_build_path_disallowed_extension(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    stage = PipelineStage.RESEARCH

    with pytest.raises(SecurityViolationError, match="not in the allow-list"):
        store.build_path(wf_id, stage, art_id, "exe")

    with pytest.raises(SecurityViolationError, match="not in the allow-list"):
        store.build_path(wf_id, stage, art_id, "JSON")  # strict matching, no normalization


def test_build_path_invalid_stage(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()

    with pytest.raises(SecurityViolationError, match="PipelineStage enum member"):
        store.build_path(wf_id, "RESEARCH", art_id, "json")  # type: ignore[arg-type]


def test_build_path_runtime_uuid_validation(store: ArtifactStore):
    """P2 test: runtime type enforcement for UUID arguments in build_path."""
    art_id = uuid.uuid4()
    wf_id = uuid.uuid4()

    # workflow_id as string
    with pytest.raises(TypeError, match="workflow_id must be a UUID instance"):
        store.build_path(str(wf_id), PipelineStage.RESEARCH, art_id, "json")  # type: ignore[arg-type]

    # artifact_id as string
    with pytest.raises(TypeError, match="artifact_id must be a UUID instance"):
        store.build_path(wf_id, PipelineStage.RESEARCH, str(art_id), "json")  # type: ignore[arg-type]

    # Traversal strings
    with pytest.raises(TypeError, match="workflow_id must be a UUID instance"):
        store.build_path("../../traversal", PipelineStage.RESEARCH, art_id, "json")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_write_and_read_artifact(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.SCRIPT, art_id, "json")

    content = b'{"title": "Test Video"}'
    expected_hash = hashlib.sha256(content).hexdigest()
    expected_size = len(content)

    checksum, size = await store.write_artifact(path, content)
    assert checksum == expected_hash
    assert size == expected_size
    assert path.exists()

    read_content = await store.read_artifact(path)
    assert read_content == content


@pytest.mark.asyncio
async def test_write_artifact_stream(store: ArtifactStore):
    """P1 test: streaming write consuming an async byte iterator with bounded memory."""
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    chunks = [b"chunk-1-" * 1000, b"chunk-2-" * 1000, b"chunk-3-" * 1000]
    expected_content = b"".join(chunks)
    expected_hash = hashlib.sha256(expected_content).hexdigest()
    expected_size = len(expected_content)

    async def chunk_generator():
        for c in chunks:
            yield c

    checksum, size = await store.write_artifact_stream(path, chunk_generator())
    assert checksum == expected_hash
    assert size == expected_size
    assert (await store.read_artifact(path)) == expected_content


@pytest.mark.asyncio
async def test_write_artifact_from_file(store: ArtifactStore, tmp_path: Path):
    """P1 test: streaming write from an existing local file in bounded chunks."""
    source_file = tmp_path / "rendered_output.mp4"
    data = b"video_data_block_" * 50000  # ~850 KB
    source_file.write_bytes(data)

    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    expected_hash = hashlib.sha256(data).hexdigest()
    checksum, size = await store.write_artifact_from_file(path, source_file)
    assert checksum == expected_hash
    assert size == len(data)
    assert (await store.read_artifact(path)) == data


@pytest.mark.asyncio
async def test_write_artifact_immutability(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    await store.write_artifact(path, b"video-frame-data")

    # Attempting to overwrite must raise ArtifactAlreadyExistsError
    with pytest.raises(ArtifactAlreadyExistsError, match="Artifact already exists"):
        await store.write_artifact(path, b"new-video-frame-data")


@pytest.mark.asyncio
async def test_concurrent_writes_same_path_race_safety(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    content_a = b"content-from-writer-A"
    content_b = b"content-from-writer-B"

    results = await asyncio.gather(
        store.write_artifact(path, content_a),
        store.write_artifact(path, content_b),
        return_exceptions=True,
    )

    successes = [r for r in results if isinstance(r, tuple)]
    conflicts = [r for r in results if isinstance(r, ArtifactAlreadyExistsError)]

    assert len(successes) == 1, f"Expected exactly 1 success, got: {results}"
    assert len(conflicts) == 1, f"Expected exactly 1 conflict, got: {results}"

    published = await store.read_artifact(path)
    assert published in (content_a, content_b)
    winner_hash = successes[0][0]
    assert hashlib.sha256(published).hexdigest() == winner_hash


@pytest.mark.asyncio
async def test_in_flight_backpressure_enforcement(artifact_root: Path):
    """P2 test: verify in-flight capacity semaphore bounds active queued operations under real load."""
    store = ArtifactStore(artifact_root, fsync=False, max_workers=4, max_in_flight=2)
    try:
        active_count = 0
        max_seen = 0
        orig_write = store._write_artifact_sync

        def instrumented_write(resolved: Path, content: bytes):
            nonlocal active_count, max_seen
            active_count += 1
            max_seen = max(max_seen, active_count)
            time.sleep(0.03)
            try:
                return orig_write(resolved, content)
            finally:
                active_count -= 1

        with patch.object(store, "_write_artifact_sync", side_effect=instrumented_write):

            async def worker(idx: int):
                art_id = uuid.uuid4()
                path = store.build_path(uuid.uuid4(), PipelineStage.RESEARCH, art_id, "json")
                return await store.write_artifact(path, f'{{"idx": {idx}}}'.encode())

            results = await asyncio.gather(*[worker(i) for i in range(8)])
            assert len(results) == 8
            assert max_seen <= 2, f"Backpressure exceeded max_in_flight=2, peaked at {max_seen}"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_in_flight_backpressure_cancellation_safety(artifact_root: Path):
    """P1 test: coroutine cancellation while thread is active keeps competing workers blocked until thread finishes."""
    store = ArtifactStore(artifact_root, fsync=False, max_workers=2, max_in_flight=1)
    thread_can_finish = threading.Event()
    try:
        thread_started = asyncio.Event()
        loop = asyncio.get_running_loop()

        def slow_blocking_write(resolved: Path, content: bytes):
            loop.call_soon_threadsafe(thread_started.set)
            unblocked = thread_can_finish.wait(timeout=10.0)
            assert unblocked, "Worker thread timed out instead of waiting for explicit test release"
            return ("dummy_hash", len(content))

        with patch.object(store, "_write_artifact_sync", side_effect=slow_blocking_write):
            p1 = store.build_path(uuid.uuid4(), PipelineStage.RESEARCH, uuid.uuid4(), "json")
            p2 = store.build_path(uuid.uuid4(), PipelineStage.RESEARCH, uuid.uuid4(), "json")

            task1 = asyncio.create_task(store.write_artifact(p1, b'{"k": "1"}'))
            await asyncio.wait_for(thread_started.wait(), timeout=5.0)

            # Start competing task2 - it must be blocked waiting for capacity
            task2 = asyncio.create_task(store.write_artifact(p2, b'{"k": "2"}'))
            await asyncio.sleep(0.05)
            assert not task2.done(), "Task 2 should be blocked waiting for permit"

            # Cancel task1 while thread 1 is still executing in thread pool
            task1.cancel()
            await asyncio.sleep(0.05)

            # While thread 1 is STILL BLOCKED, task1 has not completed and task2 is STILL BLOCKED!
            assert not task1.done(), "Task 1 completed before worker thread finished"
            assert not task2.done(), (
                "Task 2 acquired capacity prematurely while Task 1's thread was still running"
            )
            assert store.in_flight_semaphore._value == 0, (
                "Capacity permit leaked before thread finished"
            )

            # Now allow thread 1 to finish
            thread_can_finish.set()

            # Task 1 propagates CancelledError
            with pytest.raises(asyncio.CancelledError):
                await task1

            # Task 2 now acquires the permit, executes, and completes successfully
            await asyncio.wait_for(task2, timeout=5.0)
            assert task2.done()
            assert store.in_flight_semaphore._value == 1
    finally:
        thread_can_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_descriptor_ownership_single_close(artifact_root: Path):
    """P1 test: temp_fd is closed exactly once across success, link failure, and sync failure."""
    store = ArtifactStore(artifact_root, fsync=False)
    try:
        closed_fds: list[int] = []
        orig_close = os.close

        def tracking_close(fd: int):
            closed_fds.append(fd)
            return orig_close(fd)

        with patch("os.close", side_effect=tracking_close):
            # 1. Normal successful write
            p1 = store.build_path(uuid.uuid4(), PipelineStage.RESEARCH, uuid.uuid4(), "json")
            closed_fds.clear()
            await store.write_artifact(p1, b'{"status": "ok"}')
            assert len(closed_fds) == 1, f"Expected exactly 1 close on success, got: {closed_fds}"

            # 2. Duplicate link failure where link races (os.link raises FileExistsError)
            p2 = store.build_path(uuid.uuid4(), PipelineStage.RESEARCH, uuid.uuid4(), "json")
            closed_fds.clear()
            with (
                patch("os.link", side_effect=FileExistsError("Target exists")),
                pytest.raises(ArtifactAlreadyExistsError),
            ):
                await store.write_artifact(p2, b'{"status": "race_link"}')
            assert len(closed_fds) == 1, (
                f"Expected exactly 1 close on link error, got: {closed_fds}"
            )

            # 3. Directory sync failure
            p3 = store.build_path(uuid.uuid4(), PipelineStage.RESEARCH, uuid.uuid4(), "json")
            closed_fds.clear()
            with patch(
                "app.stages.artifact_store._sync_dir_if_supported",
                side_effect=OSError(errno.EIO, "Sync error"),
            ):
                store._fsync = True
                with pytest.raises(OSError, match="Sync error"):
                    await store.write_artifact(p3, b'{"status": "sync_fail"}')
                store._fsync = False
            assert len(closed_fds) == 1, (
                f"Expected exactly 1 close on sync error, got: {closed_fds}"
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_publication_failure_after_hard_link_leaves_artifact_for_reconciliation(
    artifact_root: Path,
):
    """P1 test: directory sync failure after link propagates error but preserves file on disk for reconciliation."""
    store = ArtifactStore(artifact_root, fsync=True)
    try:
        wf_id = uuid.uuid4()
        art_id = uuid.uuid4()
        path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")
        content = b"video_bytes_for_ambiguous_publication"

        with (
            patch(
                "app.stages.artifact_store._sync_dir_if_supported",
                side_effect=OSError(errno.EIO, "Disk sync failure"),
            ),
            pytest.raises(OSError, match="Disk sync failure"),
        ):
            await store.write_artifact(path, content)

        # File was hard-linked before sync failed, so it exists and is intact for reconciliation
        assert path.exists(), (
            "Hard-linked artifact must exist on disk for post-crash reconciliation"
        )
        read_back = await store.read_artifact(path)
        assert read_back == content

        # Checksum must match exactly
        checksum, size = await store.compute_checksum(path)
        assert checksum == hashlib.sha256(content).hexdigest()
        assert size == len(content)
    finally:
        store.close()


@pytest.mark.asyncio
async def test_read_artifact_size_limit_and_streaming_read(store: ArtifactStore):
    """P2 test: read_artifact enforces max_bytes, read_artifact_stream streams chunks with bounded memory."""
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    content = b"0123456789" * 10000  # 100 KB
    await store.write_artifact(path, content)

    # 1. Enforces max_bytes limit
    with pytest.raises(ValueError, match="exceeds maximum in-memory read limit"):
        await store.read_artifact(path, max_bytes=50 * 1024)

    # 2. Successful bounded read when within limit
    data = await store.read_artifact(path, max_bytes=200 * 1024)
    assert data == content

    # 3. Streaming read yields chunks
    streamed_chunks = []
    async for chunk in store.read_artifact_stream(path, chunk_size=20 * 1024):
        streamed_chunks.append(chunk)

    assert b"".join(streamed_chunks) == content
    assert len(streamed_chunks) == 5


@pytest.mark.asyncio
async def test_directory_sync_genuine_io_error_propagates(artifact_root: Path):
    """P1 test: genuine storage I/O errors during directory sync are not swallowed."""
    store = ArtifactStore(artifact_root, fsync=True)
    try:
        wf_id = uuid.uuid4()
        art_id = uuid.uuid4()
        path = store.build_path(wf_id, PipelineStage.RESEARCH, art_id, "json")

        with patch("app.stages.artifact_store._sync_dir_if_supported") as mock_sync:
            mock_sync.side_effect = OSError(errno.EIO, "I/O hardware error")
            with pytest.raises(OSError, match="I/O hardware error"):
                await store.write_artifact(path, b'{"test": "durability"}')
    finally:
        store.close()


def test_directory_sync_platform_toleration(tmp_path: Path):
    """P1 test: platform-specific unsupported limitations (EINVAL/ENOTSUP) are tolerated, while genuine EIO propagates."""
    with (
        patch("sys.platform", "linux"),
        patch.object(os, "O_DIRECTORY", 0x10000, create=True),
        patch("os.close"),
    ):
        # 1. os.open raises EINVAL -> tolerated
        with patch("os.open", side_effect=OSError(errno.EINVAL, "Invalid argument")):
            _sync_dir_if_supported(tmp_path)

        # 2. os.open raises ENOTSUP -> tolerated
        with patch("os.open", side_effect=OSError(errno.ENOTSUP, "Operation not supported")):
            _sync_dir_if_supported(tmp_path)

        # 3. os.open raises genuine EIO -> propagates
        with (
            patch("os.open", side_effect=OSError(errno.EIO, "I/O error")),
            pytest.raises(OSError, match="I/O error"),
        ):
            _sync_dir_if_supported(tmp_path)

        # 4. os.open succeeds, os.fsync raises EINVAL -> tolerated
        with (
            patch("os.open", return_value=42),
            patch("os.fsync", side_effect=OSError(errno.EINVAL, "Invalid argument")),
        ):
            _sync_dir_if_supported(tmp_path)

        # 5. os.open succeeds, os.fsync raises ENOTSUP -> tolerated
        with (
            patch("os.open", return_value=42),
            patch("os.fsync", side_effect=OSError(errno.ENOTSUP, "Operation not supported")),
        ):
            _sync_dir_if_supported(tmp_path)

        # 6. os.open succeeds, os.fsync raises genuine EIO -> propagates
        with (
            patch("os.open", return_value=42),
            patch("os.fsync", side_effect=OSError(errno.EIO, "Disk hardware failure")),
            pytest.raises(OSError, match="Disk hardware failure"),
        ):
            _sync_dir_if_supported(tmp_path)


@pytest.mark.asyncio
async def test_write_artifact_stream_cleanup_failure_safe_on_executor_shutdown(
    artifact_root: Path,
):
    """P1 test: stream cleanup falls back to synchronous execution on executor shutdown, releasing permit."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    try:
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")

        async def faulty_stream():
            yield b"first_chunk"
            # Shut down executor mid-stream to simulate worker termination / pool shutdown
            store.close()
            yield b"second_chunk"

        with pytest.raises(RuntimeError):
            await store.write_artifact_stream(path, faulty_stream())

        # Verify that despite executor shutdown, permit was released and is not leaked
        assert store.in_flight_semaphore._value == 2, (
            "Permit leaked after executor shutdown in write_artifact_stream"
        )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_large_payload_chunked_streaming(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    content = b"0123456789abcdef" * (256 * 1024)
    expected_hash = hashlib.sha256(content).hexdigest()
    expected_size = len(content)

    checksum, size = await store.write_artifact(path, content)
    assert checksum == expected_hash
    assert size == expected_size

    streamed_checksum, streamed_size = await store.compute_checksum(path)
    assert streamed_checksum == expected_hash
    assert streamed_size == expected_size


@pytest.mark.asyncio
async def test_non_blocking_event_loop(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.MEDIA_SYNTHESIS, art_id, "mp4")

    large_content = b"x" * (2 * 1024 * 1024)
    loop_ticks = []

    async def heartbeat():
        for _ in range(5):
            loop_ticks.append(time.monotonic())
            await asyncio.sleep(0.005)

    await asyncio.gather(
        store.write_artifact(path, large_content),
        heartbeat(),
    )

    assert len(loop_ticks) == 5, "Event loop coroutine was starved by blocking I/O"


@pytest.mark.asyncio
async def test_confinement_violation_write(store: ArtifactStore, tmp_path: Path):
    outside_path = tmp_path / "outside.json"

    with pytest.raises(SecurityViolationError, match="outside artifact root"):
        await store.write_artifact(outside_path, b"data")


@pytest.mark.asyncio
async def test_confinement_violation_read(store: ArtifactStore, tmp_path: Path):
    outside_file = tmp_path / "outside.json"
    outside_file.write_bytes(b"data")

    with pytest.raises(SecurityViolationError, match="outside artifact root"):
        await store.read_artifact(outside_file)


@pytest.mark.asyncio
async def test_read_nonexistent_artifact(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.AUDIO, art_id, "mp3")

    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        await store.read_artifact(path)


@pytest.mark.asyncio
async def test_compute_checksum(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.QC, art_id, "json")

    content = b'{"qc_status": "PASSED"}'
    expected_hash = hashlib.sha256(content).hexdigest()
    expected_size = len(content)

    await store.write_artifact(path, content)

    checksum, size = await store.compute_checksum(path)
    assert checksum == expected_hash
    assert size == expected_size


@pytest.mark.asyncio
async def test_compute_checksum_nonexistent(store: ArtifactStore):
    wf_id = uuid.uuid4()
    art_id = uuid.uuid4()
    path = store.build_path(wf_id, PipelineStage.QC, art_id, "json")

    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        await store.compute_checksum(path)


@pytest.mark.asyncio
async def test_write_artifact_stream_cancellation_during_setup(artifact_root: Path):
    """P1 test: cancellation during temporary-file setup waits for setup to finish and cleans up temp file."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    allow_setup_finish = threading.Event()
    try:
        loop = asyncio.get_running_loop()
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        setup_started = asyncio.Event()
        cleanup_started = threading.Event()
        orig_setup = store._setup_temp_file_sync
        allocated_temp_path = []

        def slow_setup(resolved: Path):
            res = orig_setup(resolved)
            allocated_temp_path.append(res[1])
            loop.call_soon_threadsafe(setup_started.set)
            unblocked = allow_setup_finish.wait(timeout=10.0)
            assert unblocked, "Setup thread timed out instead of waiting for explicit test release"
            return res

        orig_cleanup = store._cleanup_temp_file_sync

        def monitored_cleanup(temp_fd: int | None, temp_path: str | None):
            cleanup_started.set()
            return orig_cleanup(temp_fd, temp_path)

        async def dummy_stream():
            yield b"chunk_data"

        with (
            patch.object(store, "_setup_temp_file_sync", side_effect=slow_setup),
            patch.object(store, "_cleanup_temp_file_sync", side_effect=monitored_cleanup),
        ):
            task = asyncio.create_task(store.write_artifact_stream(path, dummy_stream()))
            await asyncio.wait_for(setup_started.wait(), timeout=5.0)

            # Cancel while setup thread is still in flight
            task.cancel()
            await asyncio.sleep(0.05)

            # Assert while worker thread is STILL BLOCKED:
            assert not task.done(), "Task completed before setup thread finished"
            assert not cleanup_started.is_set(), "Cleanup started before setup thread finished"
            assert store.in_flight_semaphore._value == 1, (
                "Permit released before setup thread finished"
            )

            # Release setup thread
            allow_setup_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert cleanup_started.is_set(), "Cleanup did not execute after setup finished"

        # Verify no orphaned temp files remain
        assert len(allocated_temp_path) == 1
        assert not os.path.exists(allocated_temp_path[0]), (
            "Temporary file leaked after cancellation during setup"
        )
        assert not any(artifact_root.rglob("*.tmp")), "Orphaned .tmp files left in storage tree"

        # Verify capacity permit was preserved and released
        assert store.in_flight_semaphore._value == 2
    finally:
        allow_setup_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_write_artifact_stream_cancellation_during_chunk_write(artifact_root: Path):
    """P1 test: cancellation during chunk write waits for write to finish before cleanup (no close/write race)."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    allow_write_finish = threading.Event()
    try:
        loop = asyncio.get_running_loop()
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        write_started = asyncio.Event()
        cleanup_started = threading.Event()
        orig_write = store._write_chunk_sync
        active_writes = 0
        max_concurrent_close_writes = 0
        lock = threading.Lock()

        def slow_write(temp_fd: int, chunk: bytes):
            nonlocal active_writes
            with lock:
                active_writes += 1
            try:
                loop.call_soon_threadsafe(write_started.set)
                unblocked = allow_write_finish.wait(timeout=10.0)
                assert unblocked, (
                    "Chunk write thread timed out instead of waiting for explicit test release"
                )
                return orig_write(temp_fd, chunk)
            finally:
                with lock:
                    active_writes -= 1

        orig_cleanup = store._cleanup_temp_file_sync

        def monitored_cleanup(temp_fd: int | None, temp_path: str | None):
            nonlocal max_concurrent_close_writes
            cleanup_started.set()
            with lock:
                if active_writes > 0:
                    max_concurrent_close_writes += 1
            return orig_cleanup(temp_fd, temp_path)

        async def two_chunk_stream():
            yield b"first_chunk"
            yield b"second_chunk"

        with (
            patch.object(store, "_write_chunk_sync", side_effect=slow_write),
            patch.object(store, "_cleanup_temp_file_sync", side_effect=monitored_cleanup),
        ):
            task = asyncio.create_task(store.write_artifact_stream(path, two_chunk_stream()))
            await asyncio.wait_for(write_started.wait(), timeout=5.0)

            # Cancel while chunk write is active on thread
            task.cancel()
            await asyncio.sleep(0.05)

            # Assert while worker thread is STILL BLOCKED:
            assert not task.done(), "Task completed before chunk write finished"
            assert not cleanup_started.is_set(), "Cleanup started before chunk write finished"
            assert store.in_flight_semaphore._value == 1, (
                "Permit released before chunk write finished"
            )

            # Release worker thread
            allow_write_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert cleanup_started.is_set(), "Cleanup did not execute after worker finished"

        # Ensure cleanup did NOT run concurrently with active write
        assert max_concurrent_close_writes == 0, (
            "Concurrent close/write race detected: cleanup executed while chunk write was active"
        )
        assert not any(artifact_root.rglob("*.tmp")), "Orphaned .tmp file left after cancellation"
        assert store.in_flight_semaphore._value == 2
    finally:
        allow_write_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_read_artifact_stream_cancellation_during_open(artifact_root: Path):
    """P1 test: cancellation during open file waits for open and closes file handle cleanly."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    allow_open_finish = threading.Event()
    try:
        loop = asyncio.get_running_loop()
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        await store.write_artifact(path, b"video_data_for_read_test")

        open_started = asyncio.Event()
        close_started = threading.Event()
        orig_open = store._open_file_read_sync
        opened_files = []

        def slow_open(resolved: Path):
            f = orig_open(resolved)
            opened_files.append(f)
            loop.call_soon_threadsafe(open_started.set)
            unblocked = allow_open_finish.wait(timeout=10.0)
            assert unblocked, (
                "Open file thread timed out instead of waiting for explicit test release"
            )
            return f

        orig_close = store._close_file_sync

        def monitored_close(file_obj):
            close_started.set()
            return orig_close(file_obj)

        async def consume():
            async for _ in store.read_artifact_stream(path):
                pass

        with (
            patch.object(store, "_open_file_read_sync", side_effect=slow_open),
            patch.object(store, "_close_file_sync", side_effect=monitored_close),
        ):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(open_started.wait(), timeout=5.0)

            task.cancel()
            await asyncio.sleep(0.05)

            # Assert while worker is STILL BLOCKED:
            assert not task.done(), "Task completed before open finished"
            assert not close_started.is_set(), "Close executed before open finished"
            assert store.in_flight_semaphore._value == 1, "Permit released before open finished"

            allow_open_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert close_started.is_set(), "Close did not execute after open finished"

        assert len(opened_files) == 1
        assert opened_files[0].closed, "File handle leaked after cancellation during open"
        assert store.in_flight_semaphore._value == 2
    finally:
        allow_open_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_read_artifact_stream_cancellation_during_chunk_read(artifact_root: Path):
    """P1 test: cancellation during chunk read waits for read to finish before closing file handle."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    allow_read_finish = threading.Event()
    try:
        loop = asyncio.get_running_loop()
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        await store.write_artifact(path, b"chunk_1_chunk_2_chunk_3")

        read_started = asyncio.Event()
        close_started = threading.Event()
        orig_read = store._read_file_sync
        active_reads = 0
        concurrent_close_reads = 0
        lock = threading.Lock()

        def slow_read(file_obj, chunk_size: int):
            nonlocal active_reads
            with lock:
                active_reads += 1
            try:
                loop.call_soon_threadsafe(read_started.set)
                unblocked = allow_read_finish.wait(timeout=10.0)
                assert unblocked, (
                    "Chunk read thread timed out instead of waiting for explicit test release"
                )
                return orig_read(file_obj, chunk_size)
            finally:
                with lock:
                    active_reads -= 1

        orig_close = store._close_file_sync

        def monitored_close(file_obj):
            nonlocal concurrent_close_reads
            close_started.set()
            with lock:
                if active_reads > 0:
                    concurrent_close_reads += 1
            return orig_close(file_obj)

        async def consume():
            async for _ in store.read_artifact_stream(path, chunk_size=5):
                pass

        with (
            patch.object(store, "_read_file_sync", side_effect=slow_read),
            patch.object(store, "_close_file_sync", side_effect=monitored_close),
        ):
            task = asyncio.create_task(consume())
            await asyncio.wait_for(read_started.wait(), timeout=5.0)

            task.cancel()
            await asyncio.sleep(0.05)

            # Assert while worker is STILL BLOCKED:
            assert not task.done(), "Task completed before read finished"
            assert not close_started.is_set(), "Close executed before read finished"
            assert store.in_flight_semaphore._value == 1, "Permit released before read finished"

            allow_read_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert close_started.is_set(), "Close did not execute after read finished"

        assert concurrent_close_reads == 0, (
            "Concurrent close/read race: file handle closed while read was active"
        )
        assert store.in_flight_semaphore._value == 2
    finally:
        allow_read_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_write_artifact_stream_repeated_cancellation(artifact_root: Path):
    """P1 test: repeated cancellation requests while worker is blocked are handled cleanly and permit released once."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    allow_finish = threading.Event()
    try:
        loop = asyncio.get_running_loop()
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        write_started = asyncio.Event()

        def slow_write(temp_fd: int, chunk: bytes):
            loop.call_soon_threadsafe(write_started.set)
            unblocked = allow_finish.wait(timeout=10.0)
            assert unblocked, (
                "Chunk write thread timed out instead of waiting for explicit test release"
            )
            return len(chunk)

        async def dummy_stream():
            yield b"payload_bytes"

        with patch.object(store, "_write_chunk_sync", side_effect=slow_write):
            task = asyncio.create_task(store.write_artifact_stream(path, dummy_stream()))
            await asyncio.wait_for(write_started.wait(), timeout=5.0)

            # Repeatedly cancel task
            for _ in range(5):
                task.cancel()
            await asyncio.sleep(0.05)

            assert not task.done(), "Task exited prematurely despite in-flight thread"
            assert store.in_flight_semaphore._value == 1

            allow_finish.set()

            with pytest.raises(asyncio.CancelledError):
                await task

            # Permit restored exactly once
            assert store.in_flight_semaphore._value == 2
    finally:
        allow_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_write_artifact_stream_cancellation_during_cleanup(artifact_root: Path):
    """P1 test: repeated cancellation requests while cleanup itself is in flight wait for cleanup and release permit once."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    allow_cleanup_finish = threading.Event()
    try:
        loop = asyncio.get_running_loop()
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        cleanup_started = asyncio.Event()
        orig_cleanup = store._cleanup_temp_file_sync
        cleanup_completed = False

        def slow_cleanup(temp_fd: int | None, temp_path: str | None):
            nonlocal cleanup_completed
            loop.call_soon_threadsafe(cleanup_started.set)
            unblocked = allow_cleanup_finish.wait(timeout=10.0)
            assert unblocked, (
                "Cleanup thread timed out instead of waiting for explicit test release"
            )
            res = orig_cleanup(temp_fd, temp_path)
            cleanup_completed = True
            return res

        async def failing_stream():
            yield b"chunk_1"
            raise ValueError("Stream failed to trigger cleanup")

        with patch.object(store, "_cleanup_temp_file_sync", side_effect=slow_cleanup):
            task = asyncio.create_task(store.write_artifact_stream(path, failing_stream()))
            await asyncio.wait_for(cleanup_started.wait(), timeout=5.0)

            # Deliver multiple repeated cancellations while cleanup thread is executing
            for _ in range(5):
                task.cancel()
            await asyncio.sleep(0.05)

            # While cleanup is still running, task is NOT done, permit is held, cleanup has not finished
            assert not task.done(), "Task completed before cleanup thread finished"
            assert store.in_flight_semaphore._value == 1, (
                "Permit released while cleanup was in-flight"
            )
            assert not cleanup_completed, "Cleanup reported completion prematurely"

            # Release cleanup thread
            allow_cleanup_finish.set()

            with pytest.raises((ValueError, asyncio.CancelledError)):
                await task

            assert cleanup_completed, "Cleanup did not complete after release"
            assert store.in_flight_semaphore._value == 2, (
                "Permit not released exactly once after cleanup"
            )
    finally:
        allow_cleanup_finish.set()
        store.close()


@pytest.mark.asyncio
async def test_write_artifact_stream_unexpected_cleanup_exception_releases_permit(
    artifact_root: Path,
):
    """P1 test: unexpected exception during cleanup still triggers exactly-once permit release in outer finally."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    try:
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")

        def exploding_cleanup(temp_fd, temp_path):
            raise RuntimeError("Catastrophic disk corruption in cleanup")

        async def failing_stream():
            yield b"some_chunk"
            raise ValueError("Stream failed midway")

        with (
            patch.object(store, "_cleanup_temp_file_sync", side_effect=exploding_cleanup),
            pytest.raises(ValueError, match="Stream failed midway"),
        ):
            await store.write_artifact_stream(path, failing_stream())

        # Unconditional outer finally guarantees permit release despite unexpected cleanup error
        assert store.in_flight_semaphore._value == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_read_artifact_stream_chunk_size_validation(artifact_root: Path):
    """P2 test: chunk_size <= 0 is rejected with ValueError before acquiring semaphore or opening file."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    try:
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        await store.write_artifact(path, b"test_payload_12345")

        # 1. chunk_size = 0 rejected
        with pytest.raises(ValueError, match="chunk_size must be positive, got: 0"):
            async for _ in store.read_artifact_stream(path, chunk_size=0):
                pass

        # 2. chunk_size = -1 rejected
        with pytest.raises(ValueError, match="chunk_size must be positive, got: -1"):
            async for _ in store.read_artifact_stream(path, chunk_size=-1):
                pass

        # Semaphore permit was NOT acquired or leaked
        assert store.in_flight_semaphore._value == 2

        # 3. Positive chunk size succeeds
        chunks = []
        async for c in store.read_artifact_stream(path, chunk_size=5):
            chunks.append(c)

        assert b"".join(chunks) == b"test_payload_12345"
        assert store.in_flight_semaphore._value == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_cleanup_failure_count_observability_on_temp_file_cleanup(artifact_root: Path):
    """P2 test: verifies that physical cleanup failures increment cleanup_failure_count for explicit observability."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    try:
        assert store.cleanup_failure_count == 0

        # Simulate os.close and os.unlink failure during _cleanup_temp_file_sync
        with (
            patch("os.close", side_effect=OSError("Simulated EBADF/EIO")),
            patch("os.unlink", side_effect=OSError("Simulated EPERM/EIO")),
        ):
            store._cleanup_temp_file_sync(1234, str(artifact_root / "fake.tmp"))

        # Both failures are tracked and observable
        assert store.cleanup_failure_count == 2
    finally:
        store.close()


@pytest.mark.asyncio
async def test_cleanup_failure_count_observability_on_stream_close(artifact_root: Path):
    """P2 test: verifies that file close failures in streaming increment cleanup_failure_count."""
    store = ArtifactStore(artifact_root, fsync=False, max_in_flight=2)
    try:
        path = store.build_path(uuid.uuid4(), PipelineStage.MEDIA_SYNTHESIS, uuid.uuid4(), "mp4")
        await store.write_artifact(path, b"stream_payload_data")
        assert store.cleanup_failure_count == 0

        class FailingCloseFile:
            def __init__(self, real_file):
                self._real = real_file

            def read(self, size):
                return self._real.read(size)

            def close(self):
                self._real.close()
                raise OSError("Simulated close failure")

        orig_open = store._open_file_read_sync

        def failing_open(resolved):
            return FailingCloseFile(orig_open(resolved))

        with patch.object(store, "_open_file_read_sync", side_effect=failing_open):
            async for _ in store.read_artifact_stream(path):
                pass

        # Close failure was tracked and permit was released
        assert store.cleanup_failure_count >= 1
        assert store.in_flight_semaphore._value == 2
    finally:
        store.close()
