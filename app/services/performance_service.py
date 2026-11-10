"""Sanitized rolling performance measurements for ingestion and RAG."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from collections import defaultdict, deque
from contextlib import contextmanager
from statistics import quantiles
from app.services.observability.langsmith_service import safe_trace

logger = logging.getLogger(__name__)
_samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=500))
_timing_context: ContextVar[dict] = ContextVar("timing_context", default={})


@contextmanager
def performance_context(**fields):
    """Attach safe request/job/document identifiers to nested timings."""
    current = dict(_timing_context.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    token = _timing_context.set(current)
    try:
        yield
    finally:
        _timing_context.reset(token)


def record_timing(stage: str, duration_ms: float, **fields) -> None:
    values = _samples[stage]
    values.append(round(duration_ms, 3))
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    p95 = quantiles(ordered, n=20, method="inclusive")[18] if len(ordered) >= 2 else p50
    context = dict(_timing_context.get())
    context.update({key: value for key, value in fields.items() if value is not None})
    logger.info(
        "performance stage=%s duration_ms=%.3f samples=%d p50_ms=%.3f p95_ms=%.3f request_id=%s job_id=%s document_id=%s user_id=%s",
        stage, duration_ms, len(ordered), p50, p95,
        context.get("request_id", "-"), context.get("job_id", "-"),
        context.get("document_id", "-"), context.get("user_id", "-"),
    )


@contextmanager
def measure(stage: str, **fields):
    started = time.perf_counter()
    context = dict(_timing_context.get())
    context.update({key: value for key, value in fields.items() if value is not None})
    with safe_trace(
        stage,
        request_id=context.get("request_id"),
        job_id=context.get("job_id"),
        document_id=context.get("document_id"),
        retrieval_stage=stage if stage.startswith("qa.") else None,
    ):
        try:
            yield
        finally:
            record_timing(stage, (time.perf_counter() - started) * 1000, **fields)
