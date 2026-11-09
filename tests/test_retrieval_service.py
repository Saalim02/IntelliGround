from types import SimpleNamespace

from app.services import retrieval_service


def test_retrieve_relevant_chunks_preserves_metadata(monkeypatch):
    metadata = {
        "chunk_id": 12,
        "file_name": "kannada.pdf",
        "doc_id": "doc-456",
        "page_number": 3,
        "page_chunk_id": 2,
        "paragraph_ids": [1, 2],
        "start_char": 120,
        "end_char": 380,
        "page_language": "kn",
        "page_language_confidence": 0.95,
        "language": "kn",
        "language_confidence": 0.98,
        "source": "kannada.pdf",
    }

    document = SimpleNamespace(
        page_content="ಕನ್ನಡ ದಾಖಲೆ ವಿಷಯ",
        metadata=metadata,
    )

    class FakeVectorStore:
        docstore = SimpleNamespace(
            _dict={"document-1": document}
        )

        def similarity_search_with_score(self, query, k):
            assert query == "ಪ್ರಶ್ನೆ"
            assert k == 3
            return [(document, 0.5)]

    monkeypatch.setattr(
        retrieval_service,
        "get_active_document",
        lambda user_id=None: {
            "active_doc_id": "doc-456",
            "active_filename": "kannada.pdf",
        },
    )

    monkeypatch.setattr(
        retrieval_service,
        "load_vectorstore",
        lambda user_id=None: FakeVectorStore(),
    )

    result = retrieval_service.retrieve_relevant_chunks(
        "ಪ್ರಶ್ನೆ",
        user_id=123,
    )

    assert result["status"] == "success"
    assert len(result["retrieved_chunks"]) == 1

    chunk = result["retrieved_chunks"][0]

    assert chunk["doc_id"] == "doc-456"
    assert chunk["paragraph_ids"] == [1, 2]
    assert chunk["start_char"] == 120
    assert chunk["end_char"] == 380
    assert chunk["page_language"] == "kn"
    assert chunk["language"] == "kn"
    assert chunk["language_confidence"] == 0.98
    assert chunk["text"] == "ಕನ್ನಡ ದಾಖಲೆ ವಿಷಯ"


def test_active_document_requires_exact_document_identity():
    active_doc = {
        "active_doc_id": "doc-123",
        "active_filename": "kannada.pdf",
    }

    assert retrieval_service._active(
        {
            "metadata": {
                "doc_id": "doc-123",
                "file_name": "kannada.pdf",
            }
        },
        active_doc,
    )


def test_active_document_rejects_missing_doc_id():
    active_doc = {
        "active_doc_id": "doc-123",
        "active_filename": "kannada.pdf",
    }

    assert not retrieval_service._active(
        {
            "metadata": {
                "file_name": "kannada.pdf",
            }
        },
        active_doc,
    )


def test_active_document_rejects_wrong_document():
    active_doc = {
        "active_doc_id": "doc-123",
        "active_filename": "kannada.pdf",
    }

    assert not retrieval_service._active(
        {
            "metadata": {
                "doc_id": "doc-456",
                "file_name": "kannada.pdf",
            }
        },
        active_doc,
    )


def test_active_document_rejects_wrong_filename():
    active_doc = {
        "active_doc_id": "doc-123",
        "active_filename": "kannada.pdf",
    }

    assert not retrieval_service._active(
        {
            "metadata": {
                "doc_id": "doc-123",
                "file_name": "hindi.pdf",
            }
        },
        active_doc,
    )


def test_active_document_rejects_missing_active_registry_fields():
    assert not retrieval_service._active(
        {"metadata": {"doc_id": "doc-123", "file_name": "kannada.pdf"}},
        {},
    )
