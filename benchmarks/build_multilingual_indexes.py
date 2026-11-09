"""Build isolated dense and sparse indexes for the multilingual benchmark."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.services.bm25_service import BM25Index, SPARSE_INDEX_SCHEMA_VERSION, save_sparse_index
from app.services.embedding_service import (
    DOCUMENT_PROMPT,
    EMBEDDING_MODEL_NAME,
    NORMALIZE_EMBEDDINGS,
    QUERY_PROMPT,
    get_embedding_model,
)
from app.services.vector_store_service import convert_chunks_to_documents


CORPUS_DIR = Path("storage/benchmarks/multilingual_20261105")
INDEX_DIR = CORPUS_DIR / "indexes"
OUTPUT_METADATA = INDEX_DIR / "index_build_report.json"
LANGUAGES = (
    "bengali", "gujarati", "hindi", "kannada", "malayalam",
    "marathi", "punjabi", "tamil", "telugu", "urdu",
)


def _load_documents() -> tuple[list, dict]:
    documents = []
    skipped: list[dict] = []
    for language in LANGUAGES:
        payload = json.loads((CORPUS_DIR / f"chunks_{language}.json").read_text(encoding="utf-8"))
        usable = [chunk for chunk in payload["chunks"] if chunk.get("text", "").strip()]
        skipped.extend(
            {"language": language, "chunk_id": chunk.get("chunk_id"), "reason": "empty text"}
            for chunk in payload["chunks"] if not chunk.get("text", "").strip()
        )
        documents.extend(convert_chunks_to_documents(usable, payload["source"]["filename"]))
    return documents, {"skipped_chunks": skipped, "input_chunk_count": len(documents) + len(skipped)}


def build_indexes() -> dict:
    if INDEX_DIR.exists() or OUTPUT_METADATA.exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark index output: {INDEX_DIR}")
    INDEX_DIR.mkdir(parents=True, exist_ok=False)
    dense_dir = INDEX_DIR / "dense_faiss"
    sparse_dir = INDEX_DIR / "bm25"
    dense_dir.mkdir(exist_ok=False)
    sparse_dir.mkdir(exist_ok=False)

    documents, counts = _load_documents()
    if not documents:
        raise ValueError("No nonempty benchmark chunks available for indexing")

    started = time.perf_counter()
    errors: list[str] = []
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(dense_dir))
    dense_duration = round(time.perf_counter() - started, 6)

    started_sparse = time.perf_counter()
    sparse = BM25Index.from_documents(documents)
    save_sparse_index(sparse, sparse_dir)
    sparse_duration = round(time.perf_counter() - started_sparse, 6)

    tokenizer_source = inspect.getsource(__import__("app.services.bm25_service", fromlist=["tokenize"]).tokenize)
    metadata = {
        "schema_version": 1,
        "benchmark": "multilingual_20261105",
        "corpus_directory": str(CORPUS_DIR),
        "index_directory": str(INDEX_DIR),
        "input_chunk_count": counts["input_chunk_count"],
        "indexed_chunk_count": len(documents),
        "skipped_chunks": counts["skipped_chunks"],
        "errors": errors,
        "dense_faiss": {
            "path": str(dense_dir),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "document_prompt": DOCUMENT_PROMPT,
            "query_prompt": QUERY_PROMPT,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "dimensions": int(vectorstore.index.d),
            "index_size": int(vectorstore.index.ntotal),
            "build_duration_seconds": dense_duration,
        },
        "bm25": {
            "path": str(sparse_dir),
            "tokenizer": "app.services.bm25_service.tokenize",
            "tokenizer_sha256": hashlib.sha256(tokenizer_source.encode()).hexdigest(),
            "sparse_index_schema_version": SPARSE_INDEX_SCHEMA_VERSION,
            "backend": type(sparse._index).__name__,
            "build_duration_seconds": sparse_duration,
        },
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = build_indexes()
    print(json.dumps({
        "indexed_chunk_count": result["indexed_chunk_count"],
        "dense_dimensions": result["dense_faiss"]["dimensions"],
        "errors": result["errors"],
    }))
