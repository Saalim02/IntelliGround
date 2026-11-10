from app.services import upload_progress_service as progress
from app.api.routes import upload as upload_route
from fastapi import HTTPException
import pytest
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor


def test_progress_stage_transitions_and_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "get_user_storage_dir", lambda _user_id: tmp_path)
    job_id = progress.normalize_job_id(None)
    progress.start_job(1, job_id, "sample.pdf")
    progress.update_job(1, job_id, 20, "Text extraction completed", document_id="doc-1")
    current = progress.get_job(1, job_id)
    assert current["percentage"] == 20
    assert current["status"] == "processing"
    assert current["document_id"] == "doc-1"

    completed = progress.complete_job(1, job_id, document_id="doc-1")
    assert completed["percentage"] == 100
    assert completed["status"] == "completed"
    assert completed["completed_at"]


def test_progress_is_monotonic_when_late_stage_updates_arrive(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "get_user_storage_dir", lambda _user_id: tmp_path)
    job_id = progress.normalize_job_id(None)
    progress.start_job(1, job_id)
    progress.update_job(1, job_id, 70, "Embedding generation completed")
    late = progress.update_job(1, job_id, 30, "Late page callback")
    assert late["percentage"] == 70
    assert late["status"] == "processing"


def test_late_updates_cannot_reopen_completed_or_failed_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "get_user_storage_dir", lambda _user_id: tmp_path)
    completed_id = progress.normalize_job_id(None)
    failed_id = progress.normalize_job_id(None)
    progress.start_job(1, completed_id)
    progress.start_job(1, failed_id)
    progress.complete_job(1, completed_id, document_id="doc-complete")
    progress.fail_job(1, failed_id, "failed")
    assert progress.update_job(1, completed_id, 40, "late callback")["status"] == "completed"
    assert progress.update_job(1, failed_id, 100, "late callback")["status"] == "failed"
    assert progress.complete_job(1, failed_id)["status"] == "failed"
    assert progress.fail_job(1, completed_id, "late failure")["status"] == "completed"


def test_concurrent_updates_to_one_job_do_not_lose_highest_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "get_user_storage_dir", lambda _user_id: tmp_path)
    job_id = progress.normalize_job_id(None)
    progress.start_job(1, job_id)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda value: progress.update_job(1, job_id, value, f"stage-{value}"),
            range(1, 101),
        ))
    assert progress.get_job(1, job_id)["percentage"] == 100


def test_progress_failure_is_explicit_and_sanitized(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "get_user_storage_dir", lambda _user_id: tmp_path)
    job_id = progress.normalize_job_id(None)
    progress.start_job(1, job_id)
    failed = progress.fail_job(1, job_id, "internal stack trace and private content")
    assert failed["status"] == "failed"
    assert failed["percentage"] == 0
    assert failed["error"] == "internal stack trace and private content"


def test_progress_jobs_are_isolated_by_user_and_job(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "get_user_storage_dir", lambda user_id: tmp_path / str(user_id))
    first = progress.normalize_job_id(None)
    second = progress.normalize_job_id(None)
    progress.start_job(1, first)
    progress.start_job(2, second)
    progress.update_job(1, first, 75, "Indexing")
    assert progress.get_job(1, first)["percentage"] == 75
    assert progress.get_job(2, second)["percentage"] == 0
    assert progress.get_job(1, second) is None


def test_upload_progress_route_uses_current_user_and_exact_job_contract(monkeypatch):
    job_id = "12345678-1234-5678-1234-567812345678"
    calls = []
    monkeypatch.setattr(
        upload_route,
        "get_job",
        lambda user_id, requested_job_id: calls.append((user_id, requested_job_id)) or {
            "job_id": requested_job_id, "status": "processing", "percentage": 20,
        },
    )

    result = upload_route.upload_progress(job_id, SimpleNamespace(id=41))

    assert result["job_id"] == job_id
    assert calls == [(41, job_id)]


def test_upload_progress_route_returns_404_for_unknown_or_other_user_job(monkeypatch):
    job_id = "12345678-1234-5678-1234-567812345678"
    monkeypatch.setattr(upload_route, "get_job", lambda _user_id, _job_id: None)

    with pytest.raises(HTTPException) as exc_info:
        upload_route.upload_progress(job_id, SimpleNamespace(id=99))

    assert exc_info.value.status_code == 404
