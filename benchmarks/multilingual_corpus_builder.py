"""Build an isolated, non-retrieval multilingual PDF benchmark corpus.

This module deliberately only writes beneath the benchmark output directory. It
does not create embeddings, update registries, or touch user-scoped storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.pdf_service import extract_text_from_pdf
from app.services.vector_store_service import convert_chunks_to_documents
from app.utils.metadata_builder import build_chunk_metadata
from app.services.language_detection_service import detect_text_language


BENCHMARK_NAME = "multilingual_20261105"
DEFAULT_SOURCE_DIR = Path("/mnt/d/Desktop/My Work/Books")
DEFAULT_OUTPUT_DIR = Path("storage/benchmarks") / BENCHMARK_NAME

PDFS = {
    "bengali": ("NIS Bengali July1-15.pdf", "ben"),
    "gujarati": ("NIS Gujarati July1-15.pdf", "guj"),
    "hindi": ("NIS Hindi July1-15.pdf", "hin"),
    "kannada": ("NIS Kannada July1-15.pdf", "kan"),
    "malayalam": ("NIS Malayalam July1-15.pdf", "mal"),
    "marathi": ("NIS Marathi July1-15.pdf", "mar"),
    "punjabi": ("NIS Punjabi July1-15.pdf", "pan"),
    "tamil": ("NIS Tamil July1-15.pdf", "tam"),
    "telugu": ("NIS Telugu July1-15.pdf", "tel"),
    "urdu": ("NIS Urdu July1-15.pdf", "urd"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document_record(document: Any) -> dict[str, Any]:
    return {
        "text": document.page_content,
        "metadata": dict(document.metadata),
    }


def _empty_stats() -> dict[str, Any]:
    return {
        "page_count": 0,
        "nonempty_page_count": 0,
        "extraction_method_counts": {},
        "chunk_count": 0,
        "character_count": 0,
        "detected_language_distribution": {},
    }


def _process_pdf(language: str, filename: str, language_hint: str, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = _utc_now()
    source_path = source_dir / filename
    source_info: dict[str, Any] = {
        "filename": filename,
        "path": str(source_path),
        "file_size_bytes": None,
        "sha256": None,
    }
    stats = _empty_stats()
    errors: list[str] = []
    pages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    doc_id = f"benchmark-{BENCHMARK_NAME}-{language}"

    if not source_path.is_file():
        errors.append(f"Source PDF does not exist: {source_path}")
    else:
        source_info["file_size_bytes"] = source_path.stat().st_size
        source_info["sha256"] = _file_sha256(source_path)
        doc_id = f"benchmark-{BENCHMARK_NAME}-{language}-{source_info['sha256'][:16]}"

        try:
            extracted = extract_text_from_pdf(
                str(source_path),
                language_hint=language_hint,
            )
            pages = list(extracted.get("pages", []))
            stats["page_count"] = extracted.get("total_pages", len(pages))
            stats["nonempty_page_count"] = sum(
                bool(page.get("text", "").strip()) for page in pages
            )
            stats["extraction_method_counts"] = dict(
                Counter(page.get("extraction_method", "unknown") for page in pages)
            )
            stats["character_count"] = sum(
                len(page.get("text", "")) for page in pages
            )
            stats["detected_language_distribution"] = dict(
                Counter(
                    detect_text_language(page.get("text", ""))["language"]
                    for page in pages
                    if page.get("text", "").strip()
                )
            )

            chunks, _chunk_configs = build_chunk_metadata(
                pages,
                filename,
                doc_id,
            )
            documents = [
                _document_record(document)
                for document in convert_chunks_to_documents(chunks, filename)
            ]
            stats["chunk_count"] = len(chunks)
        except Exception as exc:  # Record failure in the corpus, never hide it.
            errors.append(f"{type(exc).__name__}: {exc}")

    page_payload = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "language": language,
        "language_hint": language_hint,
        "doc_id": doc_id,
        "source": source_info,
        "statistics": stats,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "pages": pages,
        "errors": errors,
    }
    chunk_payload = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "language": language,
        "language_hint": language_hint,
        "doc_id": doc_id,
        "source": source_info,
        "statistics": stats,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "chunks": chunks,
        "documents": documents,
        "errors": errors,
    }
    _write_json(output_dir / f"pages_{language}.json", page_payload)
    _write_json(output_dir / f"chunks_{language}.json", chunk_payload)

    return {
        "language": language,
        "source": source_info,
        "doc_id": doc_id,
        "language_hint": language_hint,
        "statistics": stats,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.perf_counter() - started, 6),
        "errors": errors,
        "status": "error" if errors else "success",
    }


def build_corpus(source_dir: Path = DEFAULT_SOURCE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Build the corpus sequentially, refusing an existing output directory."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Benchmark output directory already exists; refusing to modify it: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    started_at = _utc_now()
    records = []
    for language, (filename, language_hint) in PDFS.items():
        record = _process_pdf(language, filename, language_hint, source_dir, output_dir)
        records.append(record)
        methods = ", ".join(
            f"{method}={count}"
            for method, count in sorted(record["statistics"]["extraction_method_counts"].items())
        ) or "none"
        print(
            f"[{record['language']}] status={record['status']} "
            f"pages={record['statistics']['page_count']} "
            f"chunks={record['statistics']['chunk_count']} "
            f"extraction_methods={methods} "
            f"elapsed_seconds={record['duration_seconds']:.3f}"
            + (f" errors={len(record['errors'])}" if record["errors"] else ""),
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "created_at": started_at,
        "finished_at": _utc_now(),
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "corpus_size": len(PDFS),
        "languages": [
            {"language": language, "filename": filename, "language_hint": hint}
            for language, (filename, hint) in PDFS.items()
        ],
        "pdfs": records,
    }
    _write_json(output_dir / "manifest.json", manifest)
    report = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "pdfs": records,
        "successful_pdf_count": sum(record["status"] == "success" for record in records),
        "error_pdf_count": sum(record["status"] == "error" for record in records),
    }
    _write_json(output_dir / "build_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        report = build_corpus(args.source_dir, args.output_dir)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"successful_pdf_count": report["successful_pdf_count"], "error_pdf_count": report["error_pdf_count"]}))
    return 0 if report["error_pdf_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
