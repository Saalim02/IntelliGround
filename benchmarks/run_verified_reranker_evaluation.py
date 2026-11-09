"""Compare isolated hybrid RRF results with the existing reranker."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.services.bm25_service import load_sparse_index
from app.services.embedding_service import get_embedding_model
from app.services.reranker_service import CrossEncoderReranker
from app.services.rrf_service import reciprocal_rank_fusion


BASE = Path("storage/benchmarks/multilingual_20261105")
DATASET = BASE / "retrieval_evaluation_dataset.json"
INDEXES = BASE / "indexes"
RESULTS = BASE / "reranker_evaluation_results.json"
REPORT = BASE / "reranker_evaluation_report.md"


def identity(metadata: dict) -> str:
    return f"{metadata.get('doc_id')}:{metadata.get('chunk_id')}"


def metric(results: list[dict], relevant: set[int]) -> dict:
    ranks = [i for i, item in enumerate(results, 1) if item["metadata"].get("chunk_id") in relevant]
    rank = ranks[0] if ranks else None
    rank5 = rank if rank and rank <= 5 else None
    rank10 = rank if rank and rank <= 10 else None
    import math
    return {
        "recall_at_5": float(bool(rank5)),
        "recall_at_10": float(bool(rank10)),
        "reciprocal_rank": 1 / rank if rank else 0.0,
        "mrr_at_5": 1 / rank5 if rank5 else 0.0,
        "ndcg_at_5": 1 / math.log2(rank5 + 1) if rank5 else 0.0,
        "ndcg_at_10": 1 / math.log2(rank10 + 1) if rank10 else 0.0,
    }


def aggregate(cases: list[dict]) -> dict:
    keys = ["recall_at_5", "recall_at_10", "reciprocal_rank", "mrr_at_5", "ndcg_at_5", "ndcg_at_10"]
    out = {key: sum(item[key] for item in cases) / len(cases) for key in keys}
    latency = [item["latency_ms"] for item in cases]
    out["mean_latency_ms"] = statistics.mean(latency)
    out["median_latency_ms"] = statistics.median(latency)
    out["query_count"] = len(cases)
    return out


def evaluate() -> dict:
    if RESULTS.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite existing reranker evaluation artifacts")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    verified = [item for item in dataset if item["judgment_status"] == "VERIFIED"]
    if not verified:
        raise ValueError("No VERIFIED judgments available")
    dense = FAISS.load_local(str(INDEXES / "dense_faiss"), get_embedding_model(), allow_dangerous_deserialization=True)
    sparse = load_sparse_index(INDEXES / "bm25")
    reranker = CrossEncoderReranker(enabled=True, model_name="BAAI/bge-reranker-v2-m3")
    cases = []
    failures = []
    load_started = time.perf_counter()
    for item in verified:
        try:
            dense_results = [
                {"identity": identity(doc.metadata), "text": doc.page_content, "metadata": dict(doc.metadata), "score": float(score)}
                for doc, score in dense.similarity_search_with_score(item["query"], k=10)
            ]
            sparse_results = sparse.search(item["query"], limit=10)
            fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, candidate_limit=10)
            started = time.perf_counter()
            ranked, status = reranker.rerank(item["query"], fused, limit=10)
            latency = (time.perf_counter() - started) * 1000
            if status != "enabled_success":
                failures.append({"query_id": item["query_id"], "status": status})
                continue
            cases.append({"query_id": item["query_id"], "language": item["language"], "latency_ms": latency, **metric(ranked, set(item["relevant_chunk_ids"]))})
        except Exception as exc:
            failures.append({"query_id": item["query_id"], "error": f"{type(exc).__name__}: {exc}"})
    load_duration = time.perf_counter() - load_started
    result = {
        "schema_version": 1,
        "benchmark": "multilingual_20261105",
        "model": "BAAI/bge-reranker-v2-m3",
        "verified_query_count": len(verified),
        "successful_query_count": len(cases),
        "model_and_run_duration_seconds": round(load_duration, 6),
        "failures": failures,
        "status": "COMPLETED" if not failures else ("COMPLETED_WITH_FAILURES" if cases else "BLOCKED_MODEL_FAILURE"),
        "aggregate": aggregate(cases) if cases else None,
        "per_language": {language: aggregate([case for case in cases if case["language"] == language]) if any(case["language"] == language for case in cases) else None for language in sorted({item["language"] for item in verified})},
        "production_enabled": False,
    }
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Verified Reranker Evaluation", "", f"Status: **{result['status']}**.", "", f"Model: `BAAI/bge-reranker-v2-m3`; verified queries: {len(verified)}; successful queries: {len(cases)}; failures: {len(failures)}.", "", "The reranker was instantiated only for this benchmark and was not enabled in production.", ""]
    if result["aggregate"]:
        a = result["aggregate"]
        lines += ["| Recall@5 | Recall@10 | MRR@5 | nDCG@5 | nDCG@10 | Mean ms | Median ms |", "|---:|---:|---:|---:|---:|---:|---:|", f"| {a['recall_at_5']:.4f} | {a['recall_at_10']:.4f} | {a['mrr_at_5']:.4f} | {a['ndcg_at_5']:.4f} | {a['ndcg_at_10']:.4f} | {a['mean_latency_ms']:.3f} | {a['median_latency_ms']:.3f} |", ""]
    lines += ["## Failures", "", "None." if not failures else "```json\n" + json.dumps(failures, ensure_ascii=False, indent=2) + "\n```", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"status": result["status"], "successful": result["successful_query_count"], "failures": len(result["failures"])}))
