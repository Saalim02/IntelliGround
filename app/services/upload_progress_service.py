"""User-scoped, durable progress records for synchronous PDF ingestion."""

from __future__ import annotations

import json
import fcntl
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.services.user_storage_service import get_user_storage_dir


def normalize_job_id(value: str | None) -> str:
    if not value:
        return str(uuid.uuid4())
    try:
        return str(UUID(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Invalid upload job ID.") from exc


def _path(user_id: int, job_id: str) -> Path:
    return get_user_storage_dir(user_id) / "upload_jobs" / f"{normalize_job_id(job_id)}.json"


@contextmanager
def _job_lock(user_id: int, job_id: str):
    """Serialize a job's read-modify-write cycle across threads/processes."""
    destination = _path(user_id, job_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(".lock")
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write(user_id: int, job_id: str, payload: dict) -> dict:
    destination = _path(user_id, job_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".upload_progress_", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return payload


def _start_job_unlocked(user_id: int, normalized: str, filename: str | None = None) -> dict:
    existing = get_job(user_id, normalized)
    if existing is not None:
        return existing
    return _write(user_id, normalized, {
        "job_id": normalized, "document_id": None,
        "status": "processing", "percentage": 0,
        "stage": "Upload initiated", "error": None,
        "filename": filename or "", "completed_at": None,
    })


def start_job(user_id: int, job_id: str, filename: str | None = None) -> dict:
    normalized = normalize_job_id(job_id)
    with _job_lock(user_id, normalized):
        return _start_job_unlocked(user_id, normalized, filename)


def update_job(user_id: int, job_id: str, percentage: int, stage: str, *, document_id: str | None = None) -> dict:
    normalized = normalize_job_id(job_id)
    with _job_lock(user_id, normalized):
        current = get_job(user_id, normalized)
        if current is None:
            current = _start_job_unlocked(user_id, normalized)
        if current.get("status") in {"completed", "failed"}:
            return current
        requested = max(0, min(100, int(percentage)))
        payload = {
            **current,
            "percentage": max(int(current.get("percentage", 0)), requested),
            "stage": stage,
            "status": "processing",
        }
        if document_id is not None:
            payload["document_id"] = document_id
        return _write(user_id, normalized, payload)


def complete_job(user_id: int, job_id: str, *, document_id: str | None = None) -> dict:
    normalized = normalize_job_id(job_id)
    with _job_lock(user_id, normalized):
        current = get_job(user_id, normalized) or _start_job_unlocked(user_id, normalized)
        if current.get("status") == "failed":
            return current
        payload = {**current, "status": "completed", "percentage": 100,
                   "stage": "PDF processing completed", "error": None,
                   "completed_at": datetime.now(timezone.utc).isoformat()}
        if document_id is not None:
            payload["document_id"] = document_id
        return _write(user_id, normalized, payload)


def fail_job(user_id: int, job_id: str, error: str) -> dict:
    normalized = normalize_job_id(job_id)
    with _job_lock(user_id, normalized):
        current = get_job(user_id, normalized) or _start_job_unlocked(user_id, normalized)
        if current.get("status") == "completed":
            return current
        if current.get("status") == "failed":
            return current
        return _write(user_id, normalized, {
            **current, "status": "failed", "stage": "PDF processing failed",
            "error": str(error)[:300], "completed_at": datetime.now(timezone.utc).isoformat(),
        })


def get_job(user_id: int, job_id: str) -> dict | None:
    path = _path(user_id, job_id)
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
