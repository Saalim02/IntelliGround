import json

import pytest

from app.services import document_registry_service


def test_save_active_document_creates_registry(
    tmp_path,
    monkeypatch,
):
    registry_path = tmp_path / "document_registry.json"

    monkeypatch.setattr(
        document_registry_service,
        "_get_registry_path",
        lambda user_id=None: registry_path,
    )

    document_registry_service.save_active_document(
        "doc-123",
        "Kannada.pdf",
        user_id=28,
    )

    data = json.loads(
        registry_path.read_text(encoding="utf-8")
    )

    assert data == {
        "active_doc_id": "doc-123",
        "active_filename": "Kannada.pdf",
    }


def test_failed_registry_replacement_preserves_old_registry(
    tmp_path,
    monkeypatch,
):
    registry_path = tmp_path / "document_registry.json"

    old_data = {
        "active_doc_id": "old-doc",
        "active_filename": "old.pdf",
    }

    registry_path.write_text(
        json.dumps(old_data),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        document_registry_service,
        "_get_registry_path",
        lambda user_id=None: registry_path,
    )

    def fail_replace(source, destination):
        raise OSError("Simulated registry replacement failure")

    monkeypatch.setattr(
        document_registry_service.os,
        "replace",
        fail_replace,
    )

    with pytest.raises(
        OSError,
        match="Simulated registry replacement failure",
    ):
        document_registry_service.save_active_document(
            "new-doc",
            "new.pdf",
            user_id=28,
        )

    actual_data = json.loads(
        registry_path.read_text(encoding="utf-8")
    )

    assert actual_data == old_data

    # Temporary registry files must be cleaned up.
    assert list(
        tmp_path.glob(".registry_tmp_*.json")
    ) == []


def test_activate_document_requires_source_and_index_and_restores_registry(tmp_path, monkeypatch):
    registry_path = tmp_path / "document_registry.json"
    source = tmp_path / "document.pdf"
    source.write_bytes(b"pdf")
    monkeypatch.setattr(document_registry_service, "_get_registry_path", lambda user_id=None: registry_path)
    monkeypatch.setattr(document_registry_service, "document_artifacts_available", lambda document, user_id: (True, ""))
    document = {
        "active_doc_id": "doc-b", "active_filename": "document.pdf", "source_path": str(source),
        "evidence_language": "en", "response_language": "ta", "total_pages": 3,
    }
    available, reason = document_registry_service.activate_document(document, 9)
    assert available is True
    assert reason == ""
    restored = json.loads(registry_path.read_text(encoding="utf-8"))
    assert restored["active_doc_id"] == "doc-b"
    assert restored["source_path"] == str(source)
    assert restored["response_language"] == "ta"


def test_missing_historical_pdf_is_reported_without_activation(tmp_path, monkeypatch):
    source = tmp_path / "missing.pdf"
    from app.services import vector_store_service
    monkeypatch.setattr(vector_store_service, "document_vectorstore_available", lambda user_id, document_id: True)
    available, reason = document_registry_service.document_artifacts_available(
        {"active_doc_id": "doc-a", "active_filename": "a.pdf", "source_path": str(source)}, 9,
    )
    assert available is False
    assert reason == "The original PDF is unavailable."


def test_missing_document_index_is_reported(tmp_path, monkeypatch):
    source = tmp_path / "present.pdf"
    source.write_bytes(b"pdf")
    from app.services import vector_store_service
    monkeypatch.setattr(vector_store_service, "document_vectorstore_available", lambda user_id, document_id: False)
    available, reason = document_registry_service.document_artifacts_available(
        {"active_doc_id": "doc-a", "active_filename": "a.pdf", "source_path": str(source)}, 9,
    )
    assert available is False
    assert reason == "The retrieval index for this PDF is unavailable."


def test_mismatched_index_metadata_is_not_accepted_as_historical_document(tmp_path, monkeypatch):
    source = tmp_path / "present.pdf"
    source.write_bytes(b"pdf")
    from app.services import vector_store_service
    monkeypatch.setattr(vector_store_service, "document_vectorstore_available", lambda user_id, document_id: True)
    monkeypatch.setattr(vector_store_service, "document_vectorstore_matches_identity", lambda user_id, document_id, filename: False)
    available, reason = document_registry_service.document_artifacts_available(
        {"active_doc_id": "doc-a", "active_filename": "a.pdf", "source_path": str(source)}, 9,
    )
    assert available is False
    assert "metadata does not match" in reason
