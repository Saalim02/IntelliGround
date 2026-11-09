from app.services.reranker_service import CrossEncoderReranker


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [len(document) for _, document in pairs]


def test_reranker_is_injected_batched_and_preserves_text():
    candidates = [
        {"identity": "a", "text": "short", "metadata": {"page": 1}},
        {"identity": "b", "text": "much longer", "metadata": {"page": 2}},
    ]
    service = CrossEncoderReranker(enabled=True, model=FakeCrossEncoder())
    ranked, status = service.rerank("query", candidates, limit=2)
    assert status == "enabled_success"
    assert ranked[0]["identity"] == "b"
    assert ranked[0]["text"] == "much longer"
    assert ranked[0]["metadata"] == {"page": 2}


def test_reranker_disabled_is_explicit_and_does_not_load_model():
    service = CrossEncoderReranker(enabled=False)
    ranked, status = service.rerank("q", [{"identity": "a", "text": "x"}])
    assert status == "disabled"
    assert service.model is None
    assert ranked[0]["identity"] == "a"
