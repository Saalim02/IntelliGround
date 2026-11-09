from app.services.rrf_service import reciprocal_rank_fusion


def item(identity):
    return {"identity": identity, "text": identity, "metadata": {"chunk_id": identity}}


def test_rrf_dense_sparse_overlap_and_duplicates():
    result = reciprocal_rank_fusion([item("a"), item("b"), item("a")], [item("b"), item("c")], k=0)
    assert [entry["identity"] for entry in result] == ["b", "a", "c"]
    assert result[0]["rrf_score"] == 1.0 + 1 / 2


def test_rrf_empty_and_deterministic_ties_with_limit():
    assert reciprocal_rank_fusion([], []) == []
    result = reciprocal_rank_fusion([item("z")], [item("a")], k=60, candidate_limit=1)
    assert result[0]["identity"] == "a"


def test_rrf_preserves_first_result_metadata():
    dense = {"identity": "same", "text": "original", "metadata": {"language": "ur"}}
    sparse = {"identity": "same", "text": "other", "metadata": {"language": "en"}}
    result = reciprocal_rank_fusion([dense], [sparse])
    assert result[0]["text"] == "original"
    assert result[0]["metadata"]["language"] == "ur"
