"""Record the isolated reranker attempt when the model cannot be loaded."""

from __future__ import annotations

import json
from pathlib import Path


BASE = Path("storage/benchmarks/multilingual_20261105")
RESULTS = BASE / "reranker_evaluation_results.json"
REPORT = BASE / "reranker_evaluation_report.md"


def write() -> dict:
    if RESULTS.exists() or REPORT.exists():
        raise FileExistsError("Refusing to overwrite existing reranker report")
    result = {
        "schema_version": 1,
        "benchmark": "multilingual_20261105",
        "model": "BAAI/bge-reranker-v2-m3",
        "status": "BLOCKED_MODEL_UNAVAILABLE",
        "verified_query_count": 30,
        "attempted_query_count": 30,
        "successful_query_count": 0,
        "metrics": None,
        "failures": [
            {
                "scope": "model_load",
                "error": "Hugging Face model metadata unavailable: temporary failure in name resolution for huggingface.co",
                "action": "Stopped after the initial isolated attempt; no substitute model used.",
            }
        ],
        "production_enabled": False,
    }
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(
        "\n".join([
            "# Verified Reranker Evaluation",
            "",
            "Status: **BLOCKED_MODEL_UNAVAILABLE**.",
            "",
            "The isolated attempt used the existing `CrossEncoderReranker` API with `BAAI/bge-reranker-v2-m3`. The model could not be loaded because Hugging Face metadata requests failed with DNS resolution errors for `huggingface.co`.",
            "",
            "- Verified queries available: 30",
            "- Reranker queries completed: 0",
            "- Metrics: not computed",
            "- Production reranking: not enabled",
            "- No substitute model was used.",
            "",
        ]),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    print(json.dumps(write()))
