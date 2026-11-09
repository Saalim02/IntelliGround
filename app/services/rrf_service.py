"""Deterministic reciprocal-rank fusion."""

from __future__ import annotations

from typing import Any, Iterable

from app.services.bm25_service import document_identity


def reciprocal_rank_fusion(
    dense_results: Iterable[dict[str, Any]],
    sparse_results: Iterable[dict[str, Any]],
    *, k: int = 60,
    candidate_limit: int | None = None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for results in (dense_results, sparse_results):
        seen_in_list: set[str] = set()
        rank = 0
        for result in results:
            item = dict(result)
            metadata = dict(item.get("metadata") or {})
            identity = str(item.get("identity") or document_identity(metadata, str(item.get("text", ""))))
            if identity in seen_in_list:
                continue
            seen_in_list.add(identity)
            rank += 1
            item["identity"] = identity
            item["metadata"] = metadata
            scores[identity] = scores.get(identity, 0.0) + 1.0 / (k + rank)
            if identity not in merged:
                merged[identity] = item
    ranked = sorted(merged.values(), key=lambda item: (-scores[item["identity"]], item["identity"]))
    for item in ranked:
        item["rrf_score"] = scores[item["identity"]]
    return ranked[:candidate_limit] if candidate_limit is not None else ranked
