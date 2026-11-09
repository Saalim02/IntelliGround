"""Evaluate isolated dense, BM25, and RRF retrieval on verified judgments."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.services.bm25_service import load_sparse_index
from app.services.embedding_service import get_embedding_model
from app.services.rrf_service import reciprocal_rank_fusion


BASE = Path("storage/benchmarks/multilingual_20261105")
DATASET = BASE / "retrieval_evaluation_dataset.json"
INDEXES = BASE / "indexes"
RESULTS = BASE / "judgment_verified_retrieval_results.json"
REPORT = BASE / "judgment_verified_retrieval_report.md"


def identity(metadata: dict) -> str:
    return f"{metadata.get('doc_id')}:{metadata.get('chunk_id')}"


def dense_search(store, query: str, limit: int) -> list[dict]:
    return [
        {"identity": identity(document.metadata), "text": document.page_content, "metadata": dict(document.metadata), "score": float(score)}
        for document, score in store.similarity_search_with_score(query, k=limit)
    ]


def relevant_rank(results: list[dict], relevant: set[int], cutoff: int | None = None) -> int | None:
    for position, result in enumerate(results[:cutoff], start=1) if cutoff else enumerate(results, start=1):
        if result["metadata"].get("chunk_id") in relevant:
            return position
    return None


def metrics(results: list[dict], relevant: set[int]) -> dict:
    rank_5 = relevant_rank(results, relevant, 5)
    rank_10 = relevant_rank(results, relevant, 10)
    rank_all = relevant_rank(results, relevant)
    return {
        "recall_at_5": float(bool(rank_5)),
        "recall_at_10": float(bool(rank_10)),
        "reciprocal_rank": 1.0 / rank_all if rank_all else 0.0,
        "mrr_at_5": 1.0 / rank_5 if rank_5 else 0.0,
        "ndcg_at_5": 1.0 / (__import__("math").log2(rank_5 + 1)) if rank_5 else 0.0,
        "ndcg_at_10": 1.0 / (__import__("math").log2(rank_10 + 1)) if rank_10 else 0.0,
        "first_relevant_rank": rank_all,
    }


def aggregate(cases: list[dict]) -> dict:
    numeric = ["recall_at_5", "recall_at_10", "reciprocal_rank", "mrr_at_5", "ndcg_at_5", "ndcg_at_10"]
    output = {key: sum(case[key] for case in cases) / len(cases) for key in numeric}
    latencies = [case["latency_ms"] for case in cases]
    output["mean_latency_ms"] = statistics.mean(latencies)
    output["median_latency_ms"] = statistics.median(latencies)
    output["query_count"] = len(cases)
    return output


def evaluate() -> dict:
    if RESULTS.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite existing verified retrieval evaluation artifacts")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    verified = [item for item in dataset if item["judgment_status"] == "VERIFIED"]
    if not verified:
        raise ValueError("No VERIFIED judgments available")

    embeddings = get_embedding_model()
    dense = FAISS.load_local(str(INDEXES / "dense_faiss"), embeddings, allow_dangerous_deserialization=True)
    sparse = load_sparse_index(INDEXES / "bm25")
    if sparse is None:
        raise ValueError("Benchmark BM25 index could not be loaded")

    method_cases = {"dense": [], "bm25": [], "hybrid_rrf": []}
    failures = []
    for item in verified:
        relevant = set(item["relevant_chunk_ids"])
        query = item["query"]
        try:
            started = time.perf_counter()
            dense_results = dense_search(dense, query, 10)
            dense_latency = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            sparse_results = sparse.search(query, limit=10)
            sparse_latency = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, candidate_limit=10)
            hybrid_latency = (time.perf_counter() - started) * 1000 + dense_latency + sparse_latency
            for method, results, latency in (
                ("dense", dense_results, dense_latency),
                ("bm25", sparse_results, sparse_latency),
                ("hybrid_rrf", fused, hybrid_latency),
            ):
                method_cases[method].append({
                    "query_id": item["query_id"],
                    "language": item["language"],
                    "latency_ms": latency,
                    **metrics(results, relevant),
                })
        except Exception as exc:
            failures.append({"query_id": item["query_id"], "error": f"{type(exc).__name__}: {exc}"})

    by_language = {}
    for language in sorted({item["language"] for item in verified}):
        by_language[language] = {
            method: aggregate([case for case in cases if case["language"] == language])
            for method, cases in method_cases.items()
        }
    result = {
        "schema_version": 1,
        "benchmark": "multilingual_20261105",
        "status": "COMPLETED_VERIFIED_EVALUATION" if not failures else "COMPLETED_WITH_FAILURES",
        "dataset": str(DATASET),
        "verified_query_count": len(verified),
        "failures": failures,
        "aggregate": {method: aggregate(cases) for method, cases in method_cases.items()},
        "per_language": by_language,
        "methods": {
            "dense": "isolated FAISS index with configured multilingual E5 model",
            "bm25": "isolated BM25 index using app.services.bm25_service.tokenize",
            "hybrid_rrf": "app.services.rrf_service.reciprocal_rank_fusion(k=60, candidate_limit=10)",
        },
    }
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Verified Multilingual Retrieval Evaluation",
        "",
        f"Status: **{result['status']}**; verified queries: **{len(verified)}**; failures: **{len(failures)}**.",
        "",
        "## Aggregate metrics",
        "",
        "| Method | Recall@5 | Recall@10 | RR | MRR@5 | nDCG@5 | nDCG@10 | Mean ms | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in result["aggregate"].items():
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.3f} |".format(method, values["recall_at_5"], values["recall_at_10"], values["reciprocal_rank"], values["mrr_at_5"], values["ndcg_at_5"], values["ndcg_at_10"], values["mean_latency_ms"], values["median_latency_ms"]))
    lines += ["", "## Per-language metrics", ""]
    for language, methods in result["per_language"].items():
        lines.append(f"### {language}")
        lines.append("")
        for method, values in methods.items():
            lines.append(f"- {method}: Recall@5={values['recall_at_5']:.4f}, Recall@10={values['recall_at_10']:.4f}, MRR@5={values['mrr_at_5']:.4f}, nDCG@5={values['ndcg_at_5']:.4f}, nDCG@10={values['ndcg_at_10']:.4f}, mean={values['mean_latency_ms']:.3f}ms, median={values['median_latency_ms']:.3f}ms.")
        lines.append("")
    lines += ["## Failures", "", "None." if not failures else json.dumps(failures, ensure_ascii=False, indent=2), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps({"status": result["status"], "verified_query_count": result["verified_query_count"], "failures": len(result["failures"])}))
