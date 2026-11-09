from pathlib import Path

import pytest

from app.services import vector_store_service


def test_sparse_write_failure_leaves_existing_faiss_and_sparse_files(tmp_path, monkeypatch):
    old = tmp_path / "users" / "1" / "vectorstore"
    old.mkdir(parents=True)
    (old / "index.faiss").write_text("old")
    (old / "sparse_index.json").write_text('{"schema_version": 1, "records": []}')

    class FakeStore:
        def save_local(self, path):
            Path(path, "index.faiss").write_text("new")

    monkeypatch.setattr(vector_store_service, "get_user_vectorstore_dir", lambda user_id: old)
    monkeypatch.setattr(vector_store_service, "get_embedding_model", lambda: object())
    monkeypatch.setattr(vector_store_service.FAISS, "from_documents", lambda documents, embeddings: FakeStore())
    def fail(*args, **kwargs):
        raise RuntimeError("sparse failure")
    monkeypatch.setattr(vector_store_service, "save_sparse_index", fail)

    with pytest.raises(RuntimeError, match="sparse failure"):
        vector_store_service.create_and_save_vectorstore([{"page_content": "x"}], user_id=1)
    assert (old / "index.faiss").read_text() == "old"
    assert (old / "sparse_index.json").read_text().startswith('{"schema_version"')
