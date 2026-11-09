
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
