"""Create an explicitly unverified, corpus-grounded retrieval dataset."""

from __future__ import annotations

import json
from pathlib import Path


CORPUS_DIR = Path("storage/benchmarks/multilingual_20261105")
SCHEMA_PATH = Path("examples/retrieval_evaluation_dataset.schema.json")
OUTPUT_PATH = CORPUS_DIR / "retrieval_evaluation_dataset.json"
LANGUAGES = (
    "bengali", "gujarati", "hindi", "kannada", "malayalam",
    "marathi", "punjabi", "tamil", "telugu", "urdu",
)


def build_dataset() -> list[dict]:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {OUTPUT_PATH}")

    dataset = []
    for language in LANGUAGES:
        payload = json.loads(
            (CORPUS_DIR / f"chunks_{language}.json").read_text(encoding="utf-8")
        )
        candidates = [
            chunk for chunk in payload["chunks"]
            if chunk["page_number"] >= 22 and len(chunk["text"].strip()) >= 250
        ][:3]
        if len(candidates) < 3:
            raise ValueError(f"Insufficient usable chunks for {language}")
        for index, chunk in enumerate(candidates, start=1):
            evidence = " ".join(chunk["text"].split())
            phrase = evidence[:160].rstrip(" .,;:")
            dataset.append(
                {
                    "query_id": f"mlb-20261105-{language}-{index:02d}",
                    "language": language,
                    "query": f"Find the passage about: {phrase}",
                    "source_document_id": payload["doc_id"],
                    "relevant_chunk_ids": [chunk["chunk_id"]],
                    "evidence": [evidence],
                    "relevance_grade": 1,
                    "judgment_status": "UNVERIFIED",
                }
            )

    OUTPUT_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset


if __name__ == "__main__":
    rows = build_dataset()
    print(json.dumps({"queries": len(rows), "verified": 0, "unverified": len(rows)}))
