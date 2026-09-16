"""Unit tests for retry and backoff calculations."""

from datetime import UTC, datetime

from app.orchestration.retry import compute_next_available_at, compute_retry_delay


def test_compute_retry_delay_bounds():
    # Deterministic without jitter
    d1 = compute_retry_delay(attempt=1, base_delay=2.0, max_delay=60.0, jitter=False)
    assert d1 == 2.0

    d2 = compute_retry_delay(attempt=2, base_delay=2.0, max_delay=60.0, jitter=False)
    assert d2 == 4.0

    d3 = compute_retry_delay(attempt=3, base_delay=2.0, max_delay=60.0, jitter=False)
    assert d3 == 8.0

    # Max delay clamping
    d_huge = compute_retry_delay(attempt=10, base_delay=2.0, max_delay=30.0, jitter=False)
    assert d_huge == 30.0

    # With jitter
    for att in range(1, 6):
        d_jit = compute_retry_delay(attempt=att, base_delay=2.0, max_delay=60.0, jitter=True)
        max_possible = min(60.0, 2.0 * (2 ** (att - 1)))
        assert 0.5 * max_possible <= d_jit <= max_possible


def test_compute_next_available_at():
    now = datetime.now(UTC)
    avail = compute_next_available_at(attempt=1, base_delay=2.0, max_delay=10.0, jitter=False)
    assert avail > now
    assert (avail - now).total_seconds() >= 1.9
