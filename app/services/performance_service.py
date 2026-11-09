"""Sanitized rolling performance measurements for ingestion and RAG."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from statistics import quantiles

logger = logging.getLogger(__name__)
_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))


def record_timing(stage: str, duration_ms: float) -> None:
    values = _samples[stage]
    values.append(round(duration_ms, 3))
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    p95 = quantiles(ordered, n=20, method="inclusive")[18] if len(ordered) >= 2 else p50
    logger.info(
        "performance stage=%s duration_ms=%.3f samples=%d p50_ms=%.3f p95_ms=%.3f",
        stage, duration_ms, len(ordered), p50, p95,
    )


@contextmanager
def measure(stage: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        record_timing(stage, (time.perf_counter() - started) * 1000)
