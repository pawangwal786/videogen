"""Retry policy and exponential backoff with jitter."""

import random
from datetime import UTC, datetime, timedelta


def compute_retry_delay(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> float:
    """Compute exponential backoff delay with jitter.

    Formula: min(max_delay, base_delay * 2^(attempt - 1)) with uniform jitter.
    """
    clamped_attempt = max(1, attempt)
    delay = min(max_delay, base_delay * (2 ** (clamped_attempt - 1)))
    if jitter:
        delay = random.uniform(0.5 * delay, delay)
    return delay


def compute_next_available_at(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> datetime:
    """Calculate the UTC timestamp at which a failed retryable job becomes available."""
    delay_seconds = compute_retry_delay(
        attempt,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
    )
    return datetime.now(UTC) + timedelta(seconds=delay_seconds)
