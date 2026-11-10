"""User-scoped dense + sparse retrieval with optional reranking."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

from app.services.bm25_service import document_identity
from app.services.document_registry_service import get_active_document
from app.services.rrf_service import reciprocal_rank_fusion
from app.services.reranker_service import get_reranker
from app.services.vector_store_service import load_sparse_vectorstore, load_vectorstore, vectorstore_read_lock
from app.services.embedding_service import embed_query_cached
from app.services.performance_service import measure

DENSE_CANDIDATE_COUNT = int(os.getenv("DENSE_CANDIDATE_COUNT", "20"))
SPARSE_CANDIDATE_COUNT = int(os.getenv("SPARSE_CANDIDATE_COUNT", "20"))
RRF_CONSTANT = int(os.getenv("RRF_CONSTANT", "60"))
RRF_CANDIDATE_LIMIT = int(os.getenv("RRF_CANDIDATE_LIMIT", "30"))
RERANKER_CANDIDATE_LIMIT = int(os.getenv("RERANKER_CANDIDATE_LIMIT", "20"))
FINAL_EVIDENCE_COUNT = int(os.getenv("FINAL_EVIDENCE_COUNT", "6"))


def get_dynamic_top_k(total_chunks: int) -> int:
    if total_chunks < 50:
        return 3
    if total_chunks < 200:
        return 4
    if total_chunks < 500:
        return 5
    return 6


def get_confidence_label(best_score: float) -> str:
    """Legacy FAISS-distance policy retained only for dense-only indexes."""
    if best_score < 1.30:
        return "high"
    if best_score < 1.60:
        return "medium"
    return "low"




def _active(result: dict, active_doc: dict) -> bool:
    """Accept only chunks belonging to the user's active document."""

    metadata = result.get("metadata") or {}

    expected_doc_id = active_doc.get("active_doc_id")
    expected_filename = active_doc.get("active_filename")

    if not expected_doc_id or not expected_filename:
        return False

    return (
        metadata.get("doc_id") == expected_doc_id
        and metadata.get("file_name") == expected_filename
    )


def _as_result(document, score: float, *, distance: bool = False) -> dict:
    metadata = dict(document.metadata)
    return {"identity": document_identity(metadata, document.page_content),
            "text": document.page_content, "metadata": metadata,
            "dense_score": None if distance else float(score),
            "distance": float(score) if distance else None}


def _chunk(result: dict, score: float) -> dict:
    metadata = result.get("metadata", {})
    chunk = {"chunk_id": metadata.get("chunk_id"), "file_name": metadata.get("file_name"),
             "page_number": metadata.get("page_number"), "text": result.get("text", ""),
             "score": float(score)}
    for field in ("doc_id", "source", "page_chunk_id", "paragraph_ids", "start_char", "end_char",
                  "page_language", "page_language_confidence", "language", "language_confidence"):
        chunk[field] = metadata.get(field)
    return chunk


def _empty(active_filename, message, *, timings=None):
    return {"status": "error", "active_document": active_filename, "retrieved_chunks": [],
            "confidence": "low", "best_score": None, "average_score": None,
            "message": message, "dense_candidate_count": 0, "sparse_candidate_count": 0,
            "fused_candidate_count": 0, "reranked_candidate_count": 0,
            "retrieval_stage_timings": timings or {}, "reranker_enabled": False,
            "reranker_status": "not_run"}


def retrieve_relevant_chunks(query: str, user_id: int | None = None):
    active_doc = get_active_document(user_id)
    if not active_doc:
        return _empty(None, "No active PDF uploaded.")
    active_filename = active_doc["active_filename"]
    with vectorstore_read_lock(user_id):
        with measure("qa.dense_index_load"):
            dense_store = load_vectorstore(user_id)
        if dense_store is None:
            return _empty(active_filename, "Vector store is empty.")

        total_chunks = len(dense_store.docstore._dict)
        legacy_top_k = get_dynamic_top_k(total_chunks)
        with measure("qa.bm25_index_load"):
            sparse_store = load_sparse_vectorstore(user_id)
    timings = {}
    dense_k = DENSE_CANDIDATE_COUNT if sparse_store else legacy_top_k
    if sparse_store is not None:
        stage_times = {}

        def dense_search():
            started = time.perf_counter()
            with measure("qa.dense_retrieval"):
                if hasattr(dense_store, "similarity_search_with_score_by_vector"):
                    result = dense_store.similarity_search_with_score_by_vector(
                        list(embed_query_cached(query)), dense_k,
                    )
                else:
                    result = dense_store.similarity_search_with_score(query, dense_k)
            stage_times["dense_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return result

        def sparse_search():
            started = time.perf_counter()
            with measure("qa.bm25_retrieval"):
                result = sparse_store.search(query, SPARSE_CANDIDATE_COUNT)
            stage_times["sparse_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return result

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid-retrieval") as pool:
            dense_future = pool.submit(dense_search)
            sparse_future = pool.submit(sparse_search)
            dense_pairs = dense_future.result()
            sparse_raw = sparse_future.result()
    else:
        started = time.perf_counter()
        with measure("qa.dense_retrieval"):
            if hasattr(dense_store, "similarity_search_with_score_by_vector"):
                dense_pairs = dense_store.similarity_search_with_score_by_vector(
                    list(embed_query_cached(query)), k=dense_k,
                )
            else:
                dense_pairs = dense_store.similarity_search_with_score(query, k=dense_k)
        dense_duration_ms = round((time.perf_counter() - started) * 1000, 3)
        sparse_raw = []
        stage_times = {"dense_ms": dense_duration_ms}
    timings.update(stage_times)
    timings.setdefault("sparse_ms", 0.0)
    dense_results = []
    for document, distance in dense_pairs:
        result = _as_result(document, distance, distance=True)
        if _active(result, active_doc):
            dense_results.append(result)

    if sparse_store is None:
        if not dense_results:
            return _empty(active_filename, "Answer not found in uploaded PDF.", timings=timings)
        chunks = [_chunk(result, result["distance"]) for result in dense_results[:legacy_top_k]]
        scores = [item["score"] for item in chunks]
        return {"status": "success", "active_document": active_filename, "top_k_used": legacy_top_k,
                "retrieved_chunks": chunks, "confidence": get_confidence_label(min(scores)),
                "best_score": round(min(scores), 4), "average_score": round(sum(scores) / len(scores), 4),
                "message": "Relevant chunks retrieved successfully.", "dense_candidate_count": len(dense_results),
                "sparse_candidate_count": 0, "fused_candidate_count": len(chunks),
                "reranked_candidate_count": 0, "retrieval_stage_timings": timings,
                "reranker_enabled": False, "reranker_status": "not_run"}

    sparse_results = [item for item in sparse_raw if _active(item, active_doc)]
    start = time.perf_counter()
    with measure("qa.rrf_fusion"):
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=RRF_CONSTANT, candidate_limit=RRF_CANDIDATE_LIMIT)
    timings["rrf_ms"] = round((time.perf_counter() - start) * 1000, 3)
    reranker = get_reranker()
    start = time.perf_counter()
    with measure("qa.reranking"):
        reranked, reranker_status = reranker.rerank(query, fused[:RERANKER_CANDIDATE_LIMIT], FINAL_EVIDENCE_COUNT)
    timings["reranker_ms"] = round((time.perf_counter() - start) * 1000, 3)
    if not reranked:
        return _empty(active_filename, "Answer not found in uploaded PDF.", timings=timings)
    score_key = "reranker_score" if reranker_status == "enabled_success" else "rrf_score"
    chunks = [_chunk(result, result[score_key]) for result in reranked[:FINAL_EVIDENCE_COUNT]]
    scores = [item["score"] for item in chunks]
    return {"status": "success", "active_document": active_filename, "top_k_used": FINAL_EVIDENCE_COUNT,
            "retrieved_chunks": chunks, "confidence": "medium", "best_score": round(max(scores), 4),
            "average_score": round(sum(scores) / len(scores), 4),
            "message": "Relevant chunks retrieved successfully.", "dense_candidate_count": len(dense_results),
            "sparse_candidate_count": len(sparse_results), "fused_candidate_count": len(fused),
            "reranked_candidate_count": len(reranked), "retrieval_stage_timings": timings,
            "reranker_enabled": reranker.enabled, "reranker_status": reranker_status}
