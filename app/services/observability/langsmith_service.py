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
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_ALLOWED_METADATA = {
    "operation", "environment", "route", "intent", "document_language",
    "query_language", "response_language", "retrieval_stage",
    "retrieved_chunk_count", "reranked_chunk_count", "duration_ms",
    "status", "error_category",
    "document_identifier_hash",
}
_SENSITIVE_FIELD_NAMES = {
    "api_key", "password", "authorization", "cookie", "token",
    "audio", "audio_base64", "prompt", "question", "query", "answer",
    "response", "text", "content", "chunk_text", "pdf",
}


def tracing_status() -> dict:
    """Return configuration state without exposing values."""
    required = ("LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    return {
        "enabled": os.getenv("LANGSMITH_TRACING", "").strip().lower() in _TRUE_VALUES,
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
        except Exception as exc:
            logger.warning("LangSmith trace finalization unavailable: %s", type(exc).__name__)


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
    )
    try:
        client = _client()
    except Exception as exc:
        logger.warning("LangSmith client unavailable: %s", type(exc).__name__)
        client = None
    handle = TraceHandle(client, None, operation, metadata, time.perf_counter())
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
        )
        handle.run_id = run_id
    except Exception as exc:
        logger.warning("LangSmith trace creation unavailable: %s", type(exc).__name__)
    return handle


@contextmanager
def safe_trace(operation: str, **kwargs) -> Iterator[TraceHandle]:
    """Trace a scoped operation without changing its exception behavior."""
    handle = start_trace(operation, **kwargs)
    try:
        yield handle
    except Exception as exc:
        handle.finish(status="error", error_category=type(exc).__name__)
        raise
    else:
        handle.finish()


def create_connectivity_trace() -> bool:
    """Create a harmless connectivity trace using only fixed test metadata."""
    handle = start_trace("intelliground_connectivity_test", environment="local")
    if handle.run_id is None:
        return False
    handle.finish()
    return True


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
