
import json
from pathlib import Path

from app.services import vector_store_service


# --------------------------------------------------
# VECTORSTORE DIRECTORY TESTS
# --------------------------------------------------

def test_vectorstore_directory_is_user_scoped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        vector_store_service,
        "get_user_vectorstore_dir",
        lambda user_id: tmp_path / "users" / str(user_id) / "vectorstore",
    )

    user_dir = vector_store_service._get_vectorstore_dir(123)

    assert user_dir == (
        tmp_path / "users" / "123" / "vectorstore"
    )


def test_vectorstore_directory_uses_legacy_path_without_user(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        vector_store_service,
        "VECTORSTORE_DIR",
        tmp_path / "legacy-vectorstore",
    )

    result = vector_store_service._get_vectorstore_dir()

    assert result == tmp_path / "legacy-vectorstore"


# --------------------------------------------------
# CHUNK CONVERSION TESTS
# --------------------------------------------------

def test_convert_chunks_to_documents_preserves_doc_metadata():
    chunks = [
        {
            "text": "First chunk",
            "page_number": 2,
            "chunk_id": 7,
            "doc_id": "doc-123",
        }
    ]

    documents = vector_store_service.convert_chunks_to_documents(
        chunks,
        "example.pdf",
    )

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == "First chunk"
    assert document.metadata["page_number"] == 2
    assert document.metadata["chunk_id"] == 7
    assert document.metadata["doc_id"] == "doc-123"
    assert document.metadata["file_name"] == "example.pdf"


def test_convert_chunks_to_documents_preserves_chunk_metadata():
    chunks = [
        {
            "text": "Kannada document content",
            "page_number": 3,
            "chunk_id": 12,
            "doc_id": "doc-456",
            "page_chunk_id": 2,
            "paragraph_ids": [1, 2],
            "start_char": 120,
            "end_char": 380,
            "page_language": "kn",
            "page_language_confidence": 0.95,
            "language": "kn",
            "language_confidence": 0.98,
            "source": "example.pdf",
        }
    ]

    documents = vector_store_service.convert_chunks_to_documents(
        chunks,
        "example.pdf",
    )

    assert len(documents) == 1

    metadata = documents[0].metadata

    assert metadata["page_number"] == 3
    assert metadata["chunk_id"] == 12
    assert metadata["doc_id"] == "doc-456"
    assert metadata["file_name"] == "example.pdf"
    assert metadata["page_chunk_id"] == 2
    assert metadata["paragraph_ids"] == [1, 2]
    assert metadata["start_char"] == 120
    assert metadata["end_char"] == 380
    assert metadata["page_language"] == "kn"
    assert metadata["page_language_confidence"] == 0.95
    assert metadata["language"] == "kn"
    assert metadata["language_confidence"] == 0.98
    assert metadata["source"] == "example.pdf"


# --------------------------------------------------
# VECTORSTORE CREATION TESTS
# --------------------------------------------------

def test_create_and_save_vectorstore_writes_to_user_directory(
    tmp_path,
    monkeypatch,
):
    class FakeEmbeddings:
        pass

    class FakeVectorStore:
        def __init__(self):
            self.saved_to = None

        def save_local(self, path):
            self.saved_to = Path(path)
            self.saved_to.mkdir(
                parents=True,
                exist_ok=True,
            )

            (self.saved_to / "index.faiss").write_text("faiss")
            (self.saved_to / "index.pkl").write_text("pickle")

    fake_vectorstore = FakeVectorStore()

    monkeypatch.setattr(
        vector_store_service,
        "get_embedding_model",
        lambda: FakeEmbeddings(),
    )

    monkeypatch.setattr(
        vector_store_service.FAISS,
        "from_documents",
        lambda documents, embeddings: fake_vectorstore,
    )

    monkeypatch.setattr(
        vector_store_service,
        "get_user_vectorstore_dir",
        lambda user_id: tmp_path / "users" / str(user_id) / "vectorstore",
    )

    result = vector_store_service.create_and_save_vectorstore(
        [{"page_content": "test"}],
        user_id=123,
    )

    expected_dir = (
        tmp_path / "users" / "123" / "vectorstore"
    )

    assert result is fake_vectorstore
    assert fake_vectorstore.saved_to != expected_dir

    assert (expected_dir / "index.faiss").exists()
    assert (expected_dir / "index.pkl").exists()

    assert (
        (expected_dir / "index.faiss").read_text()
        == "faiss"
    )

    assert (
        (expected_dir / "index.pkl").read_text()
        == "pickle"
    )

    # Verify embedding manifest.
    manifest_path = (
        expected_dir
        / vector_store_service.EMBEDDING_MANIFEST_FILENAME
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest == vector_store_service._get_embedding_manifest()
    assert manifest["model_name"] == "intfloat/multilingual-e5-small"
    assert manifest["document_prompt"] == "passage: "
    assert manifest["query_prompt"] == "query: "
    assert manifest["normalize_embeddings"] is True


def test_create_and_save_vectorstore_preserves_existing_store_on_failure(
    tmp_path,
    monkeypatch,
):
    class FakeEmbeddings:
        pass

    class FakeVectorStore:
        def save_local(self, path):
            path = Path(path)
            path.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Simulate a partial write followed by a failure.
            (path / "index.faiss").write_text("partial-new-faiss")

            raise RuntimeError("simulated save failure")

    vectorstore_dir = (
        tmp_path / "users" / "123" / "vectorstore"
    )

    vectorstore_dir.mkdir(parents=True)

    # Existing working vectorstore.
    (vectorstore_dir / "index.faiss").write_text("old-faiss")
    (vectorstore_dir / "index.pkl").write_text("old-pickle")

    monkeypatch.setattr(
        vector_store_service,
        "get_embedding_model",
        lambda: FakeEmbeddings(),
    )

    monkeypatch.setattr(
        vector_store_service.FAISS,
        "from_documents",
        lambda documents, embeddings: FakeVectorStore(),
    )

    monkeypatch.setattr(
        vector_store_service,
        "get_user_vectorstore_dir",
        lambda user_id: vectorstore_dir,
    )

    try:
        vector_store_service.create_and_save_vectorstore(
            [{"page_content": "test"}],
            user_id=123,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected simulated save failure")

    assert (
        vectorstore_dir / "index.faiss"
    ).read_text() == "old-faiss"

    assert (
        vectorstore_dir / "index.pkl"
    ).read_text() == "old-pickle"


def test_vectorstore_transaction_restores_previous_store_on_rollback(tmp_path):
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()
    (vectorstore_dir / "marker").write_text("old")
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "marker").write_text("new")

    transaction = vector_store_service.VectorStoreTransaction(
        vectorstore_dir, staged_dir
    )
    with transaction:
        transaction.install()
        transaction.rollback()

    assert (vectorstore_dir / "marker").read_text() == "old"


def test_vectorstore_transaction_removes_new_store_on_rollback(tmp_path):
    vectorstore_dir = tmp_path / "vectorstore"
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "marker").write_text("new")

    transaction = vector_store_service.VectorStoreTransaction(
        vectorstore_dir, staged_dir
    )
    with transaction:
        transaction.install()
        transaction.rollback()

    assert not vectorstore_dir.exists()


def test_vectorstore_transaction_commit_removes_backup(tmp_path):
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()
    (vectorstore_dir / "marker").write_text("old")
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "marker").write_text("new")

    transaction = vector_store_service.VectorStoreTransaction(
        vectorstore_dir, staged_dir
    )
    with transaction:
        transaction.install()
        transaction.commit()

    assert (vectorstore_dir / "marker").read_text() == "new"
    assert not list(tmp_path.glob(".vectorstore_backup_*"))


def test_vectorstore_commit_mark_retains_backup_until_context_cleanup(tmp_path):
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()
    (vectorstore_dir / "marker").write_text("old")
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "marker").write_text("new")

    transaction = vector_store_service.VectorStoreTransaction(
        vectorstore_dir, staged_dir
    )
    with transaction:
        transaction.install()
        transaction.mark_committed()
        assert transaction.backup_dir.exists()

    assert not list(tmp_path.glob(".vectorstore_backup_*"))


def test_vectorstore_backup_names_are_unique(tmp_path):
    vectorstore_dir = tmp_path / "vectorstore"
    vectorstore_dir.mkdir()
    (vectorstore_dir / "marker").write_text("old")
    existing_backup = tmp_path / ".vectorstore_backup_existing"
    existing_backup.mkdir()
    (existing_backup / "marker").write_text("keep")
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()

    transaction = vector_store_service.VectorStoreTransaction(
        vectorstore_dir, staged_dir
    )
    with transaction:
        transaction.install()
        transaction.rollback()

    assert (existing_backup / "marker").read_text() == "keep"


# --------------------------------------------------
# EMBEDDING MANIFEST TESTS
# --------------------------------------------------

def test_embedding_manifest_matches_current_configuration(
    tmp_path,
):
    manifest = vector_store_service._get_embedding_manifest()

    manifest_path = (
        tmp_path
        / vector_store_service.EMBEDDING_MANIFEST_FILENAME
    )

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert vector_store_service._has_compatible_embedding_manifest(
        tmp_path
    )


def test_missing_embedding_manifest_is_rejected(tmp_path):
    assert not vector_store_service._has_compatible_embedding_manifest(
        tmp_path
    )


def test_mismatched_embedding_manifest_is_rejected(tmp_path):
    manifest = vector_store_service._get_embedding_manifest()

    manifest["model_name"] = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    manifest_path = (
        tmp_path
        / vector_store_service.EMBEDDING_MANIFEST_FILENAME
    )

    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    assert not vector_store_service._has_compatible_embedding_manifest(
        tmp_path
    )


def test_corrupt_embedding_manifest_is_rejected(tmp_path):
    manifest_path = (
        tmp_path
        / vector_store_service.EMBEDDING_MANIFEST_FILENAME
    )

    manifest_path.write_text(
        "{invalid json",
        encoding="utf-8",
    )

    assert not vector_store_service._has_compatible_embedding_manifest(
        tmp_path
    )


def test_sparse_index_loader_reuses_unchanged_committed_index(tmp_path, monkeypatch):
    manifest_path = tmp_path / vector_store_service.EMBEDDING_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(vector_store_service._get_embedding_manifest()),
        encoding="utf-8",
    )
    sparse_path = tmp_path / "sparse_index.json"
    sparse_path.write_text("{}", encoding="utf-8")
    loaded = object()
    calls = []

    monkeypatch.setattr(
        vector_store_service,
        "load_sparse_index",
        lambda directory: calls.append(directory) or loaded,
    )
    vector_store_service._load_sparse_index_cached.cache_clear()

    monkeypatch.setattr(
        vector_store_service,
        "_get_vectorstore_dir",
        lambda user_id=None: tmp_path,
    )

    assert vector_store_service.load_sparse_vectorstore(7) is loaded
    assert vector_store_service.load_sparse_vectorstore(7) is loaded
    assert calls == [tmp_path]
