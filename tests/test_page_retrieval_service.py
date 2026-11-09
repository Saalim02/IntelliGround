from types import SimpleNamespace

import pytest

from app.services import page_retrieval_service as service


def _record(doc_id, filename, page, chunk_id, page_chunk_id, text, **extra):
    metadata = {
        "doc_id": doc_id, "file_name": filename, "source": filename,
        "page_number": page, "chunk_id": chunk_id,
        "page_chunk_id": page_chunk_id, "start_char": extra.pop("start_char", chunk_id * 10),
        "end_char": extra.pop("end_char", chunk_id * 10 + len(text)),
        **extra,
    }
    return SimpleNamespace(text=text, metadata=metadata)


@pytest.fixture
def page_store(monkeypatch, tmp_path):
    active = {
        "active_doc_id": "doc-a", "active_filename": "report.pdf",
        "total_pages": 50,
        "evidence_language": "ta", "response_language": "kn",
    }
    records = [
        _record("doc-a", "report.pdf", 2, 3, 2, "page two second", language="ta"),
        _record("doc-a", "report.pdf", 1, 2, 2, "page one second", language="ta"),
        _record("doc-a", "report.pdf", 1, 1, 1, "page one first", language="ta"),
        _record("doc-b", "report.pdf", 1, 99, 1, "other document"),
    ]
    index = SimpleNamespace(records=records)
    monkeypatch.setattr(service, "get_active_document", lambda user_id: active if user_id == 7 else None)
    monkeypatch.setattr(service, "load_sparse_index", lambda path: index)
    monkeypatch.setattr(service, "get_user_vectorstore_dir", lambda user_id: tmp_path / "users" / str(user_id) / "vectorstore")
    monkeypatch.setattr(service, "get_user_debug_dir", lambda user_id: tmp_path / "debug" / str(user_id))
    return active, tmp_path


def test_single_page_retrieval_preserves_chunks_and_metadata(page_store):
    result = service.retrieve_pages(1, user_id=7)
    assert result["status"] == "success"
    assert [chunk["chunk_id"] for chunk in result["pages"][0]["chunks"]] == [1, 2]
    assert result["pages"][0]["text"] == "page one first\n\npage one second"
    assert result["evidence"][0]["page_number"] == 1
    assert result["evidence"][0]["image_path"].endswith("/page_1.png")


def test_multiple_pages_follow_requested_page_order(page_store):
    result = service.retrieve_pages([2, 1], user_id=7)
    assert [page["page_number"] for page in result["pages"]] == [2, 1]
    assert [chunk["page_number"] for chunk in result["retrieved_chunks"]] == [2, 1, 1]


def test_page_numbers_are_one_based(page_store):
    with pytest.raises(service.PageRetrievalError):
        service.retrieve_pages(0, user_id=7)


def test_out_of_range_page_and_missing_active_document(page_store):
    result = service.retrieve_pages(99, user_id=7)
    assert result["status"] == "error"
    assert result["pages"] == []
    assert "not present" in result["message"]

    no_document = service.retrieve_pages(1, user_id=8)
    assert no_document["status"] == "error"
    assert no_document["active_document"] is None


def test_physical_page_count_rejects_mixed_invalid_request_before_index_lookup(page_store, monkeypatch):
    def fail_if_loaded(_path):
        raise AssertionError("index lookup must not run for an impossible page")

    monkeypatch.setattr(service, "load_sparse_index", fail_if_loaded)
    result = service.retrieve_pages([45, 50, 52], user_id=7)
    assert result["status"] == "error"
    assert "50 pages" in result["message"]
    assert "Pages 45, 50 exist" in result["message"]
    assert "Page 52 is not present" in result["message"]


def test_cross_document_chunks_are_isolated(page_store):
    result = service.retrieve_pages(1, user_id=7)
    assert all(chunk["doc_id"] == "doc-a" for chunk in result["retrieved_chunks"])
    assert "other document" not in result["pages"][0]["text"]


def test_stored_empty_page_uses_ocr_fallback_and_preserves_languages(page_store, monkeypatch, tmp_path):
    active, root = page_store
    clean_dir = root / "debug" / "7" / "report" / "clean_pages"
    clean_dir.mkdir(parents=True)
    (clean_dir / "page_3.txt").write_text("", encoding="utf-8")
    pdf_path = root / "users" / "7" / "uploads" / "report.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"pdf")

    monkeypatch.setattr(
        service, "extract_text_from_pdf",
        lambda path, language_hint="auto": {
            "pages": [{"page_number": 3, "text": "தமிழ் பக்கம்", "extraction_method": "ocr"}]
        },
    )
    result = service.retrieve_pages(3, user_id=7, pdf_path=pdf_path)
    assert result["status"] == "success"
    assert result["pages"][0]["extraction_method"] == "ocr"
    assert result["pages"][0]["text"] == "தமிழ் பக்கம்"
    assert result["pages"][0]["evidence_language"] == "ta"
    assert result["pages"][0]["response_language"] == "kn"


def test_ocr_source_must_be_in_user_upload_directory(page_store, tmp_path):
    _, root = page_store
    clean_dir = root / "debug" / "7" / "report" / "clean_pages"
    clean_dir.mkdir(parents=True)
    (clean_dir / "page_3.txt").write_text("", encoding="utf-8")
    result = service.retrieve_pages(3, user_id=7, pdf_path=tmp_path / "outside.pdf")
    assert result["status"] == "error"
    assert result["pages"] == []
    assert "extraction is unavailable" in result["message"]


def test_ocr_source_must_match_active_document(page_store, tmp_path):
    _, root = page_store
    clean_dir = root / "debug" / "7" / "report" / "clean_pages"
    clean_dir.mkdir(parents=True)
    (clean_dir / "page_3.txt").write_text("", encoding="utf-8")
    other = tmp_path / "users" / "7" / "uploads" / "other.pdf"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"pdf")
    result = service.retrieve_pages(3, user_id=7, pdf_path=other)
    assert result["status"] == "error"
    assert "extraction is unavailable" in result["message"]


def test_missing_page_content_is_truthful(page_store):
    result = service.retrieve_pages(3, user_id=7)
    assert result["status"] == "error"
    assert "extraction is unavailable" in result["message"] or "out of range" in result["message"]
