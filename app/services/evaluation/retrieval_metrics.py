"""Ground-truth retrieval metrics based on ranked, stable chunk IDs.

This module is intentionally separate from :mod:`retrieval_evaluator`, whose
distance-based score is a heuristic used by the existing application flow.
Similarity or distance scores are never read by these functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Hashable, Iterable, Mapping, Sequence, TypeAlias

ChunkId: TypeAlias = Hashable


@dataclass(frozen=True)
class RetrievalCase:
    """One query's ranked output and ground-truth judgments.

    ``relevance_grades`` is optional. If supplied, it maps relevant chunk IDs
    to non-negative grades, where larger values are more relevant. The mapping
    may include every ID in ``relevant_chunk_ids`` or a subset of it; omitted
    IDs are treated as grade zero by graded metrics.
    """

    language: str
    query: str
    retrieved_chunk_ids: Sequence[ChunkId]
    relevant_chunk_ids: Sequence[ChunkId]
    relevance_grades: Mapping[ChunkId, float] | None = None
    latency_ms: float | None = None


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")


def _validate_chunk_id(chunk_id: ChunkId) -> None:
    if chunk_id is None:
        raise ValueError("chunk IDs must not be None")
    try:
        hash(chunk_id)
    except TypeError as exc:
        raise ValueError("chunk IDs must be hashable stable identifiers") from exc


def _unique_ids(chunk_ids: Iterable[ChunkId]) -> list[ChunkId]:
    """Return IDs in first-seen order, removing duplicate retrieval results."""

    unique: list[ChunkId] = []
    seen: set[ChunkId] = set()
    for chunk_id in chunk_ids:
        _validate_chunk_id(chunk_id)
        if chunk_id not in seen:
            seen.add(chunk_id)
            unique.append(chunk_id)
    return unique


def _relevant_set(relevant_chunk_ids: Iterable[ChunkId]) -> set[ChunkId]:
    return set(_unique_ids(relevant_chunk_ids))


def _top_k(retrieved_chunk_ids: Iterable[ChunkId], k: int) -> list[ChunkId]:
    _validate_k(k)
    return _unique_ids(retrieved_chunk_ids)[:k]


def recall_at_k(
    retrieved_chunk_ids: Iterable[ChunkId],
    relevant_chunk_ids: Iterable[ChunkId],
    k: int,
) -> float:
    """Return the fraction of relevant chunks retrieved in the first ``k``.

    Returns ``0.0`` when there are no relevant chunks. Duplicate IDs are
    counted once and preserve their first-seen rank.
    """

    _validate_k(k)
    retrieved = _top_k(retrieved_chunk_ids, k)
    relevant = _relevant_set(relevant_chunk_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_chunk_ids: Iterable[ChunkId],
    relevant_chunk_ids: Iterable[ChunkId],
    k: int | None = None,
) -> float:
    """Return the reciprocal rank of the first relevant retrieved chunk.

    Rank is one-based. If ``k`` is provided, only the first ``k`` unique
    retrieved IDs are considered. No relevant document yields ``0.0``.
    """

    ranked = _unique_ids(retrieved_chunk_ids) if k is None else _top_k(retrieved_chunk_ids, k)
    relevant = _relevant_set(relevant_chunk_ids)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(cases: Iterable[RetrievalCase], k: int | None = None) -> float:
    """Return the mean reciprocal rank across query cases.

    The mean is over all supplied cases, including cases with no relevant
    retrieved chunk. An empty case collection returns ``0.0``.
    """

    if k is not None:
        _validate_k(k)
    case_list = list(cases)
    if not case_list:
        return 0.0
    return sum(
        reciprocal_rank(case.retrieved_chunk_ids, case.relevant_chunk_ids, k)
        for case in case_list
    ) / len(case_list)


def mrr_at_k(cases: Iterable[RetrievalCase], k: int) -> float:
    """Return MRR@K across query cases."""

    return mean_reciprocal_rank(cases, k)


def _validated_grades(relevance_grades: Mapping[ChunkId, float]) -> dict[ChunkId, float]:
    grades: dict[ChunkId, float] = {}
    for chunk_id, grade in relevance_grades.items():
        _validate_chunk_id(chunk_id)
        if isinstance(grade, bool) or not isinstance(grade, (int, float)):
            raise ValueError("relevance grades must be finite non-negative numbers")
        if not math.isfinite(float(grade)) or grade < 0:
            raise ValueError("relevance grades must be finite non-negative numbers")
        grades[chunk_id] = float(grade)
    return grades


def dcg_at_k(
    retrieved_chunk_ids: Iterable[ChunkId],
    relevance_grades: Mapping[ChunkId, float],
    k: int,
) -> float:
    """Return discounted cumulative gain using ``2**grade - 1`` gains."""

    grades = _validated_grades(relevance_grades)
    return sum(
        (2.0 ** grades.get(chunk_id, 0.0) - 1.0) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(_top_k(retrieved_chunk_ids, k), start=1)
    )


def ndcg_at_k(
    retrieved_chunk_ids: Iterable[ChunkId],
    relevance_grades: Mapping[ChunkId, float],
    k: int,
) -> float:
    """Return normalized DCG@K, or ``0.0`` when ideal gain is zero."""

    _validate_k(k)
    ranked = _top_k(retrieved_chunk_ids, k)
    grades = _validated_grades(relevance_grades)
    if not grades or k == 0:
        return 0.0
    ideal_ids = sorted(grades, key=lambda chunk_id: grades[chunk_id], reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_ids, grades, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked, grades, k) / ideal_dcg


def _validated_latencies(latencies_ms: Iterable[float]) -> list[float]:
    values: list[float] = []
    for latency in latencies_ms:
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise ValueError("latencies must be finite non-negative numbers")
        value = float(latency)
        if not math.isfinite(value) or value < 0:
            raise ValueError("latencies must be finite non-negative numbers")
        values.append(value)
    return values


def mean_latency_ms(latencies_ms: Iterable[float]) -> float:
    """Return arithmetic mean latency in milliseconds; empty input is ``0.0``."""

    values = _validated_latencies(latencies_ms)
    return sum(values) / len(values) if values else 0.0


def median_latency_ms(latencies_ms: Iterable[float]) -> float:
    """Return median latency in milliseconds; empty input is ``0.0``."""

    values = _validated_latencies(latencies_ms)
    return float(median(values)) if values else 0.0


def mean_latency(latencies_ms: Iterable[float]) -> float:
    """Return the arithmetic mean of latency measurements in milliseconds."""

    return mean_latency_ms(latencies_ms)


def median_latency(latencies_ms: Iterable[float]) -> float:
    """Return the median of latency measurements in milliseconds."""

    return median_latency_ms(latencies_ms)
