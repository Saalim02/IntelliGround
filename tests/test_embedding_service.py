
from app.services import embedding_service


def test_embedding_model_uses_multilingual_e5(monkeypatch):
    captured = {}

    class FakeHuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        embedding_service,
        "HuggingFaceEmbeddings",
        FakeHuggingFaceEmbeddings,
    )

    embedding_service.get_embedding_model()

    assert captured["model"] == "intfloat/multilingual-e5-small"

    assert captured["encode_kwargs"] == {
        "normalize_embeddings": True,
        "prompt": "passage: ",
    }

    assert captured["query_encode_kwargs"] == {
        "normalize_embeddings": True,
        "prompt": "query: ",
    }


def test_embedding_model_name():
    assert (
        embedding_service.EMBEDDING_MODEL_NAME
        == "intfloat/multilingual-e5-small"
    )


def test_query_cache_identity_changes_with_model_and_dimensions(monkeypatch):
    calls = []

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def embed_query(self, _query):
            return [0.0] * embedding_service.EMBEDDING_DIMENSIONS

    monkeypatch.setattr(embedding_service, "HuggingFaceEmbeddings", FakeEmbeddings)
    embedding_service._get_embedding_model_cached.cache_clear()
    embedding_service._embed_query_cached.cache_clear()
    monkeypatch.setattr(embedding_service, "EMBEDDING_MODEL_NAME", "model-a")
    monkeypatch.setattr(embedding_service, "EMBEDDING_DIMENSIONS", 3)
    assert len(embedding_service.embed_query_cached("same query")) == 3
    assert len(embedding_service.embed_query_cached("same query")) == 3
    monkeypatch.setattr(embedding_service, "EMBEDDING_MODEL_NAME", "model-b")
    monkeypatch.setattr(embedding_service, "EMBEDDING_DIMENSIONS", 4)
    assert len(embedding_service.embed_query_cached("same query")) == 4
    assert len(calls) == 2


def test_query_cache_rejects_incompatible_dimensions(monkeypatch):
    class FakeEmbeddings:
        def embed_query(self, _query):
            return [0.0, 0.0]

    monkeypatch.setattr(embedding_service, "_get_embedding_model_cached", lambda *_args: FakeEmbeddings())
    embedding_service._embed_query_cached.cache_clear()
    monkeypatch.setattr(embedding_service, "EMBEDDING_DIMENSIONS", 3)
    try:
        embedding_service.embed_query_cached("dimension mismatch")
    except ValueError as exc:
        assert "dimension mismatch" in str(exc)
    else:
        raise AssertionError("Expected incompatible embedding dimensions to fail")
