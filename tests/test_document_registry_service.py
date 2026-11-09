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