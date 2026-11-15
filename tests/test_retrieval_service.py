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


def test_query_variants_search_original_index_and_deduplicate(monkeypatch):
    metadata = {"chunk_id": 1, "file_name": "malayalam.pdf", "doc_id": "doc-ml", "page_number": 1}
    document = SimpleNamespace(page_content="മലയാളം രേഖ", metadata=metadata)
    queries = []

    class FakeVectorStore:
        docstore = SimpleNamespace(_dict={"document-1": document})

        def similarity_search_with_score(self, query, k):
            queries.append(query)
            return [(document, 0.5)]

    monkeypatch.setattr(retrieval_service, "get_active_document", lambda user_id=None: {
        "active_doc_id": "doc-ml", "active_filename": "malayalam.pdf",
    })
    monkeypatch.setattr(retrieval_service, "load_vectorstore", lambda user_id=None: FakeVectorStore())
    monkeypatch.setattr(retrieval_service, "load_sparse_vectorstore", lambda user_id=None: None)

    result = retrieval_service.retrieve_relevant_chunks(
        "یہ سوال ہے؟", user_id=123, query_variants=["ഈ ചോദ്യം എന്താണ്?"],
    )

    assert queries == ["یہ سوال ہے؟", "ഈ ചോദ്യം എന്താണ്?"]
    assert [chunk["chunk_id"] for chunk in result["retrieved_chunks"]] == [1]


def test_retrieval_loads_the_active_document_index_and_filters_cross_document_chunks(monkeypatch):
    requested = []
    active_document = {"active_doc_id": "doc-b", "active_filename": "same-name.pdf"}
    matching = SimpleNamespace(page_content="B content", metadata={
        "chunk_id": 2, "file_name": "same-name.pdf", "doc_id": "doc-b", "page_number": 1,
    })
    wrong = SimpleNamespace(page_content="A content", metadata={
        "chunk_id": 1, "file_name": "same-name.pdf", "doc_id": "doc-a", "page_number": 1,
    })

    class FakeVectorStore:
        docstore = SimpleNamespace(_dict={"b": matching, "a": wrong})
        def similarity_search_with_score(self, query, k):
            return [(matching, 0.2), (wrong, 0.1)]

    monkeypatch.setattr(retrieval_service, "get_active_document", lambda user_id=None: active_document)
    monkeypatch.setattr(retrieval_service, "load_vectorstore", lambda user_id=None, document_id=None: requested.append(document_id) or FakeVectorStore())
    monkeypatch.setattr(retrieval_service, "load_sparse_vectorstore", lambda user_id=None, document_id=None: None)
    result = retrieval_service.retrieve_relevant_chunks("Which document?", user_id=7)
    assert requested == ["doc-b"]
    assert [chunk["doc_id"] for chunk in result["retrieved_chunks"]] == ["doc-b"]


def test_missing_document_index_does_not_fallback_to_shared_legacy_index(monkeypatch):
    active = {"active_doc_id": "doc-b", "active_filename": "b.pdf"}
    legacy = SimpleNamespace(docstore=SimpleNamespace(_dict={"legacy": object()}))
    monkeypatch.setattr(retrieval_service, "get_active_document", lambda user_id=None: active)
    monkeypatch.setattr(retrieval_service, "load_vectorstore", lambda user_id=None, document_id=None: None if document_id else legacy)
    monkeypatch.setattr(retrieval_service, "load_sparse_vectorstore", lambda user_id=None, document_id=None: None)
    result = retrieval_service.retrieve_relevant_chunks("question", user_id=7)
    assert result["status"] == "error"
    assert result["retrieved_chunks"] == []


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
