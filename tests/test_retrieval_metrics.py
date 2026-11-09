import math

import pytest

from app.services.evaluation.retrieval_metrics import (
    RetrievalCase,
    dcg_at_k,
    mean_latency_ms,
    mean_reciprocal_rank,
    mrr_at_k,
    median_latency_ms,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_supports_multiple_relevant_chunks_and_k():
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], 2) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)


def test_reciprocal_rank_and_mrr_at_k():
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == pytest.approx(0.5)
    cases = [
        RetrievalCase("en", "one", ["a", "x"], ["a"]),
        RetrievalCase("en", "two", ["b"], ["a"]),
    ]
    assert mean_reciprocal_rank(cases, k=1) == pytest.approx(0.5)
    assert mrr_at_k(cases, k=1) == pytest.approx(0.5)


def test_dcg_and_ndcg_use_graded_relevance():
    grades = {"a": 3, "b": 1}
    expected = 7 / math.log2(2) + 1 / math.log2(3)
    assert dcg_at_k(["a", "b"], grades, 2) == pytest.approx(expected)
    assert ndcg_at_k(["b", "a"], grades, 2) < 1.0
    assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)


def test_duplicates_do_not_inflate_metrics_and_are_first_seen():
    assert recall_at_k(["a", "a", "b"], ["a", "b"], 2) == 1.0
    assert reciprocal_rank(["x", "a", "a"], ["a"]) == pytest.approx(0.5)
    assert dcg_at_k(["a", "a"], {"a": 1}, 2) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [-1, True, 1.5])
def test_invalid_k_is_rejected(k):
    with pytest.raises(ValueError):
        recall_at_k([], [], k)


def test_empty_and_no_relevant_inputs_return_zero():
    assert recall_at_k([], ["a"], 5) == 0.0
    assert reciprocal_rank(["a"], [], 5) == 0.0
    assert mean_reciprocal_rank([], 5) == 0.0
    assert ndcg_at_k([], {}, 5) == 0.0
    assert recall_at_k(["a"], ["a"], 0) == 0.0


def test_invalid_ids_grades_and_latencies_are_rejected():
    with pytest.raises(ValueError):
        recall_at_k([["not", "hashable"]], ["a"], 1)
    with pytest.raises(ValueError):
        dcg_at_k(["a"], {"a": -1}, 1)
    with pytest.raises(ValueError):
        mean_latency_ms([1, float("nan")])


def test_latency_statistics_are_deterministic_and_handle_empty_input():
    values = [3, 1, 2, 10]
    assert mean_latency_ms(values) == 4.0
    assert median_latency_ms(values) == 2.5
    assert mean_latency_ms([]) == 0.0
    assert median_latency_ms([]) == 0.0
