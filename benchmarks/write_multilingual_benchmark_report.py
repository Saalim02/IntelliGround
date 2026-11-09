"""Write the reproducible final status report for the multilingual benchmark."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path("storage/benchmarks/multilingual_20261105")
RESULTS = BASE / "multilingual_retrieval_benchmark_results.json"
REPORT = BASE / "multilingual_retrieval_benchmark_report.md"


def write_report() -> dict:
    if RESULTS.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite existing benchmark final artifacts")

    quality = json.loads((BASE / "ocr_quality_report.json").read_text(encoding="utf-8"))
    dataset = json.loads((BASE / "retrieval_evaluation_dataset.json").read_text(encoding="utf-8"))
    index = json.loads((BASE / "indexes/index_build_report.json").read_text(encoding="utf-8"))
    languages = sorted({item["language"] for item in dataset})
    verified = [item for item in dataset if item["judgment_status"] == "VERIFIED"]

    results = {
        "schema_version": 1,
        "benchmark": "multilingual_20261105",
        "status": "BASELINE_BLOCKED_PENDING_VERIFIED_JUDGMENTS",
        "corpus": {
            "directory": str(BASE),
            "pdfs": 10,
            "pages": 440,
            "chunks": 3216,
            "extracted_characters": 1202428,
            "extraction_errors": 0,
        },
        "ocr_quality": {
            "report": str(BASE / "ocr_quality_report.json"),
            "classification_counts": quality["totals"]["classification_counts"],
            "empty_pages": quality["totals"]["empty_pages"],
        },
        "evaluation_dataset": {
            "path": str(BASE / "retrieval_evaluation_dataset.json"),
            "query_count": len(dataset),
            "languages": languages,
            "verified_query_count": len(verified),
            "unverified_query_count": len(dataset) - len(verified),
        },
        "indexes": index,
        "retrieval_evaluation": {
            "status": "NOT_VALID_WITHOUT_VERIFIED_JUDGMENTS",
            "evaluated_query_count": 0,
            "metrics": None,
            "per_language": {
                language: {
                    "verified_query_count": sum(
                        item["language"] == language for item in verified
                    ),
                    "metrics": None,
                }
                for language in languages
            },
            "dense": "not evaluated",
            "sparse_bm25": "not evaluated",
            "hybrid_rrf": "not evaluated",
        },
        "reranker_evaluation": {
            "status": "SKIPPED_NO_VERIFIED_JUDGMENTS",
            "model": "BAAI/bge-reranker-v2-m3",
            "metrics": None,
            "failures": [],
        },
        "blockers": [
            "No query has VERIFIED relevance judgments; benchmark metrics would be invalid.",
            "The jsonschema package is unavailable in the environment; dataset validation used a structural validator implementing the checked schema constraints.",
        ],
        "production_safety": {
            "production_rag_code_modified": False,
            "production_indexes_modified": False,
            "registries_or_user_storage_modified": False,
            "source_pdfs_env_backups_modified": False,
        },
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Multilingual Retrieval Benchmark — 5 November 2026",
        "",
        "## Status",
        "",
        "Baseline corpus extraction and isolated index construction are complete. Retrieval metrics are **not valid yet** because all 30 candidate judgments remain `UNVERIFIED`.",
        "",
        "## Verified findings",
        "",
        f"- Corpus: 10 PDFs, 440 pages, 3,216 chunks, 1,202,428 extracted characters, zero extraction errors.",
        f"- OCR audit: {quality['totals']['classification_counts']}; empty pages: {quality['totals']['empty_pages']}.",
        f"- Dense FAISS: `{index['dense_faiss']['embedding_model']}`, dimension {index['dense_faiss']['dimensions']}, {index['dense_faiss']['index_size']} vectors.",
        f"- BM25: {index['bm25']['backend']}, tokenizer `app.services.bm25_service.tokenize`, {index['indexed_chunk_count']} records.",
        "- All index artifacts are isolated under the benchmark corpus directory.",
        "",
        "## Unverified candidate judgments",
        "",
        "The evaluation dataset contains 30 queries, three per language. Every record is marked `UNVERIFIED`; candidate chunk IDs and evidence are proposals only and must not be used for benchmark scoring.",
        "",
        "## Retrieval evaluation",
        "",
        "Dense, BM25, and hybrid RRF metrics were not computed. Recall@5/10, reciprocal rank, MRR@5, nDCG@5/10, mean latency, and median latency require VERIFIED judgments.",
        "",
        "## Reranker evaluation",
        "",
        "Skipped because no verified queries exist. The production system was not changed and reranking was not enabled.",
        "",
        "## OCR limitations and weaknesses",
        "",
        "- Kannada includes a very short/corrupted page and mixed legacy-encoded text on some pages.",
        "- Urdu page 44 is predominantly English publisher/postal metadata and has no detected Arabic-script characters.",
        "- Several language cover and closing pages contain noisy OCR or publisher metadata; see the page-level JSON audit for exact classifications and samples.",
        "- OCR extraction success was not treated as proof of semantic accuracy.",
        "",
        "## Exact commands used",
        "",
        "```text",
        "pytest -q tests/test_multilingual_corpus_builder.py tests/test_pdf_ocr_quality.py tests/test_language_detection.py",
        "python benchmarks/ocr_quality_audit.py",
        "python benchmarks/build_evaluation_dataset.py",
        "python -m benchmarks.build_multilingual_indexes",
        "python -c '<structural validation of retrieval_evaluation_dataset.json against examples/retrieval_evaluation_dataset.schema.json constraints>'",
        "python -c '<FAISS/BM25 persisted-index validation>'",
        "pytest -q tests/test_bm25_service.py tests/test_rrf_service.py tests/test_retrieval_metrics.py tests/test_embedding_service.py",
        "```",
        "",
        "## Safety and artifacts",
        "",
        f"- Corpus and all benchmark artifacts: `{BASE}`",
        f"- OCR JSON: `{BASE / 'ocr_quality_report.json'}`",
        f"- OCR Markdown: `{BASE / 'ocr_quality_report.md'}`",
        f"- Evaluation dataset: `{BASE / 'retrieval_evaluation_dataset.json'}`",
        f"- Index report: `{BASE / 'indexes/index_build_report.json'}`",
        f"- Final results JSON: `{RESULTS}`",
        f"- Final report: `{REPORT}`",
        "- Production RAG code, indexes, registries, user storage, source PDFs, `.env`, and backups were not modified.",
        "",
        "## Recommended next action",
        "",
        "Have a reviewer verify or correct the 30 proposed query judgments, then rerun dense/BM25/hybrid evaluation and optionally the BAAI reranker comparison using only VERIFIED records.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return results


if __name__ == "__main__":
    result = write_report()
    print(json.dumps({"status": result["status"], "verified_queries": result["evaluation_dataset"]["verified_query_count"]}))
