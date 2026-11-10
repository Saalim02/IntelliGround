"""Failure-safe, metadata-only LangSmith tracing foundation.

This module is intentionally isolated. Nothing imports it from the request
path yet; later phases can add narrowly scoped instrumentation after this
foundation is verified.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Iterator

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_ALLOWED_METADATA = {
    "operation", "environment", "route", "intent", "document_language",
    "query_language", "response_language", "retrieval_stage",
    "retrieved_chunk_count", "reranked_chunk_count", "duration_ms",
    "status", "error_category", "request_id", "pipeline_version",
    "page_count", "chunk_count", "dense_candidate_count",
    "sparse_candidate_count", "fused_candidate_count", "reranker_status",
    "model", "provider", "prompt_version", "token_usage_available",
    "guardrail_decision", "evaluation_decision", "voice_stage",
    "document_identifier_hash", "job_id",
}
_SENSITIVE_FIELD_NAMES = {
    "api_key", "password", "authorization", "cookie", "token",
    "audio", "audio_base64", "prompt", "question", "query", "answer",
    "response", "text", "content", "chunk_text", "pdf",
}


_metrics_lock = Lock()
_active_parent: ContextVar[str | None] = ContextVar("langsmith_active_parent", default=None)
_runtime_override: bool | None = None
_metrics = {
    "total_recorded_traces": 0,
    "successful_runs": 0,
    "failed_runs": 0,
    "total_duration_ms": 0.0,
    "durations_ms": [],
    "last_trace_timestamp": None,
    "operation_counts": {},
}


def _reset_runtime_state() -> None:
    """Test-only reset for process-local controls and counters."""
    global _runtime_override
    with _metrics_lock:
        _runtime_override = None
        _metrics.update({
            "total_recorded_traces": 0, "successful_runs": 0,
            "failed_runs": 0, "total_duration_ms": 0.0,
            "durations_ms": [], "last_trace_timestamp": None,
            "operation_counts": {},
        })


def set_tracing_enabled(enabled: bool | None) -> bool | None:
    """Set a process-local developer override; ``None`` returns to env config."""
    global _runtime_override
    if enabled is not None and not isinstance(enabled, bool):
        raise ValueError("Tracing enabled must be a boolean or null.")
    with _metrics_lock:
        _runtime_override = enabled
        _client.cache_clear()
        return _runtime_override


def tracing_status() -> dict:
    """Return configuration state without exposing values."""
    required = ("LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    with _metrics_lock:
        override = _runtime_override
    configured_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() in _TRUE_VALUES
    return {
        "enabled": configured_enabled if override is None else override,
        "configured_enabled": configured_enabled,
        "runtime_override": override,
        "configured": not missing,
        "missing": missing,
        "project_configured": bool(os.getenv("LANGSMITH_PROJECT", "").strip()),
        "endpoint_configured": bool(os.getenv("LANGSMITH_ENDPOINT", "").strip()),
    }


def _metadata(**values) -> dict:
    """Build the fixed operational metadata schema; reject sensitive fields."""
    for key in values:
        if key.lower() in _SENSITIVE_FIELD_NAMES:
            raise ValueError(f"Sensitive metadata field is not allowed: {key}")
        if key not in _ALLOWED_METADATA:
            raise ValueError(f"Unsupported metadata field: {key}")
    return {key: value for key, value in values.items() if value is not None}


def _document_identifier(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def _client():
    status = tracing_status()
    if not status["enabled"] or not status["configured"]:
        return None
    try:
        from langsmith import Client

        return Client(
            api_url=os.environ["LANGSMITH_ENDPOINT"].strip(),
            api_key=os.environ["LANGSMITH_API_KEY"].strip(),
        )
    except Exception as exc:  # observability must never affect requests
        logger.warning("LangSmith client initialization unavailable: %s", type(exc).__name__)
        return None


@dataclass
class TraceHandle:
    client: object | None
    run_id: str | None
    operation: str
    metadata: dict
    started: float
    parent_run_id: str | None = None

    def finish(self, *, status: str = "success", error_category: str | None = None) -> None:
        if self.client is None or self.run_id is None:
            return
        final_metadata = dict(self.metadata)
        final_metadata["duration_ms"] = round((time.perf_counter() - self.started) * 1000, 3)
        final_metadata["status"] = status
        if error_category:
            final_metadata["error_category"] = error_category
        try:
            self.client.update_run(
                self.run_id,
                end_time=int(time.time() * 1000),
                outputs={"status": status},
                extra={"metadata": final_metadata},
            )
            duration = final_metadata["duration_ms"]
            with _metrics_lock:
                _metrics["successful_runs" if status == "success" else "failed_runs"] += 1
                _metrics["total_duration_ms"] += duration
                _metrics["durations_ms"].append(duration)
                _metrics["last_trace_timestamp"] = time.time()
        except Exception as exc:
            logger.warning("LangSmith trace finalization unavailable: %s", type(exc).__name__)

    def child(self, operation: str, **kwargs) -> "TraceHandle":
        return start_trace(operation, parent_run_id=self.run_id, **kwargs)


def start_trace(
    operation: str,
    *,
    environment: str | None = None,
    route: str | None = None,
    intent: str | None = None,
    document_id: str | None = None,
    document_language: str | None = None,
    query_language: str | None = None,
    response_language: str | None = None,
    retrieval_stage: str | None = None,
    parent_run_id: str | None = None,
    request_id: str | None = None,
    pipeline_version: str | None = None,
    page_count: int | None = None,
    chunk_count: int | None = None,
    dense_candidate_count: int | None = None,
    sparse_candidate_count: int | None = None,
    fused_candidate_count: int | None = None,
    reranker_status: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    prompt_version: str | None = None,
    token_usage_available: bool | None = None,
    guardrail_decision: str | None = None,
    evaluation_decision: str | None = None,
    voice_stage: str | None = None,
    job_id: str | None = None,
) -> TraceHandle:
    metadata = _metadata(
        operation=operation,
        environment=environment,
        route=route,
        intent=intent,
        document_language=document_language,
        query_language=query_language,
        response_language=response_language,
        retrieval_stage=retrieval_stage,
        request_id=request_id,
        pipeline_version=pipeline_version,
        page_count=page_count,
        chunk_count=chunk_count,
        dense_candidate_count=dense_candidate_count,
        sparse_candidate_count=sparse_candidate_count,
        fused_candidate_count=fused_candidate_count,
        reranker_status=reranker_status,
        model=model,
        provider=provider,
        prompt_version=prompt_version,
        token_usage_available=token_usage_available,
        guardrail_decision=guardrail_decision,
        evaluation_decision=evaluation_decision,
        voice_stage=voice_stage,
        job_id=job_id,
    )
    parent_run_id = parent_run_id or _active_parent.get()
    try:
        client = _client()
    except Exception as exc:
        logger.warning("LangSmith client unavailable: %s", type(exc).__name__)
        client = None
    handle = TraceHandle(client, None, operation, metadata, time.perf_counter(), parent_run_id)
    if client is None:
        return handle
    run_id = str(uuid.uuid4())
    if document_id:
        metadata["document_identifier_hash"] = _document_identifier(document_id)
    try:
        client.create_run(
            id=run_id,
            name=operation,
            run_type="chain",
            inputs={"operation": operation},
            project_name=os.environ["LANGSMITH_PROJECT"].strip(),
            extra={"metadata": metadata},
            **({"parent_run_id": parent_run_id} if parent_run_id else {}),
        )
        handle.run_id = run_id
        with _metrics_lock:
            _metrics["total_recorded_traces"] += 1
            _metrics["operation_counts"][operation] = _metrics["operation_counts"].get(operation, 0) + 1
    except Exception as exc:
        logger.warning("LangSmith trace creation unavailable: %s", type(exc).__name__)
    return handle


@contextmanager
def safe_trace(operation: str, **kwargs) -> Iterator[TraceHandle]:
    """Trace a scoped operation without changing its exception behavior."""
    handle = start_trace(operation, **kwargs)
    token = _active_parent.set(handle.run_id or _active_parent.get())
    try:
        yield handle
    except Exception as exc:
        handle.finish(status="error", error_category=type(exc).__name__)
        raise
    else:
        handle.finish()
    finally:
        _active_parent.reset(token)


def observed(operation: str, **trace_kwargs):
    """Decorate an execution path with a failure-safe parent trace."""
    def decorator(function):
        from functools import wraps

        @wraps(function)
        def wrapped(*args, **kwargs):
            with safe_trace(operation, route=kwargs.get("endpoint"), **trace_kwargs):
                return function(*args, **kwargs)

        return wrapped
    return decorator


def create_connectivity_trace() -> bool:
    """Create a harmless connectivity trace using only fixed test metadata."""
    handle = start_trace("intelliground_connectivity_test", environment="local")
    if handle.run_id is None:
        return False
    handle.finish()
    return True


def observability_metrics() -> dict:
    with _metrics_lock:
        durations = sorted(_metrics["durations_ms"])
        total = _metrics["total_recorded_traces"]
        # Metrics are process-local and intentionally unavailable after restart.
        return {
            "available": bool(total),
            "scope": "process",
            "total_recorded_traces": total,
            "successful_runs": _metrics["successful_runs"],
            "failed_runs": _metrics["failed_runs"],
            "average_latency_ms": round(_metrics["total_duration_ms"] / total, 3) if total else None,
            "p50_latency_ms": _percentile(durations, 0.50),
            "p95_latency_ms": _percentile(durations, 0.95),
            "last_trace_timestamp": _metrics["last_trace_timestamp"],
            "evaluation_run_count": sum(value for name, value in _metrics["operation_counts"].items() if "evaluation" in name or "grounding" in name),
        }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return round(values[index], 3)


def instrumentation_coverage() -> dict:
    return {
        name: {"instrumented": enabled, "status": "instrumented" if enabled else "not_instrumented"}
        for name, enabled in {
            "ingestion": True, "intent_classification": True,
            "retrieval": True, "reranking": True, "generation": True,
            "evaluation": True, "guardrails": True, "voice": True,
        }.items()
    }


def shutdown() -> None:
    client = _client()
    if client is None:
        return
    try:
        client.close()
    except Exception as exc:
        logger.warning("LangSmith client shutdown unavailable: %s", type(exc).__name__)
    finally:
        _client.cache_clear()
