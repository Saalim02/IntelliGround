"""Exact, user-scoped retrieval of indexed PDF pages.

This service intentionally does not call dense retrieval, BM25 search, RRF,
or the reranker.  A page number is an explicit document coordinate, so the
existing sparse index is used only as a persistent store of all chunk records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.document_registry_service import get_active_document
from app.services.pdf_service import extract_text_from_pdf
from app.services.bm25_service import load_sparse_index
from app.services.user_storage_service import get_user_debug_dir, get_user_vectorstore_dir


class PageRetrievalError(ValueError):
    """Raised for invalid page requests or unavailable active documents."""


def _page_sort_key(chunk: dict[str, Any]) -> tuple[int, int, int, int]:
    metadata = chunk.get("metadata") or {}
    return (
        int(metadata.get("page_number", 0)),
        int(metadata.get("page_chunk_id", 0)),
        int(metadata.get("start_char", 0)),
        int(metadata.get("chunk_id", 0)),
    )


def _chunk_from_record(record: Any, active: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(getattr(record, "metadata", None) or {})
    text = str(getattr(record, "text", "") or "")
    chunk = {
        "chunk_id": metadata.get("chunk_id"),
        "file_name": metadata.get("file_name", active["active_filename"]),
        "page_number": metadata.get("page_number"),
        "text": text,
        # Direct page matches are deterministic, not similarity scores.
        "score": 1.0,
    }
    for field in (
        "doc_id", "source", "page_chunk_id", "paragraph_ids", "start_char",
        "end_char", "page_language", "page_language_confidence", "language",
        "language_confidence",
    ):
        chunk[field] = metadata.get(field)
    return chunk


def _record_chunks(user_id: int, active: dict[str, Any]) -> list[dict[str, Any]]:
    index = load_sparse_index(get_user_vectorstore_dir(user_id))
    if index is None:
        return []

    expected_doc_id = active.get("active_doc_id")
    expected_filename = active.get("active_filename")
    records = []
    for record in index.records:
        metadata = getattr(record, "metadata", None) or {}
        if (
            metadata.get("doc_id") == expected_doc_id
            and metadata.get("file_name") == expected_filename
            and metadata.get("page_number") is not None
        ):
            records.append(_chunk_from_record(record, active))
    return sorted(records, key=lambda item: _page_sort_key({"metadata": item}))


def _debug_page_text(user_id: int, active: dict[str, Any], page_number: int) -> str | None:
    """Load already-extracted page text exported during ingestion."""
    document_dir = get_user_debug_dir(user_id) / Path(active["active_filename"]).stem
    page_path = document_dir / "clean_pages" / f"page_{page_number:03}.txt"
    if not page_path.is_file():
        return None
    return page_path.read_text(encoding="utf-8")


def _authorized_pdf_path(user_id: int, pdf_path: str | Path, active: dict[str, Any]) -> str:
    """Accept OCR input only from the authenticated user's upload directory."""
    upload_root = (get_user_vectorstore_dir(user_id).parent / "uploads").resolve()
    candidate = Path(pdf_path).resolve()
    try:
        candidate.relative_to(upload_root)
    except ValueError as exc:
        raise PageRetrievalError("OCR source is outside the user's upload directory.") from exc
    if not candidate.is_file():
        raise PageRetrievalError("OCR source PDF is unavailable.")
    # The current registry stores the original filename but not the generated
    # upload path.  Refuse a path whose filename cannot be tied to that active
    # document rather than risking OCR from another user's active/old upload.
    if candidate.name != Path(str(active["active_filename"])).name:
        raise PageRetrievalError("OCR source cannot be tied to the active document.")
    return str(candidate)


def _ocr_page_text(
    user_id: int,
    page_number: int,
    *,
    active: dict[str, Any],
    pdf_path: str | Path | None,
    language_hint: str,
) -> tuple[str | None, str]:
    if pdf_path is None:
        return None, "extraction_unavailable"
    try:
        authorized_path = _authorized_pdf_path(user_id, pdf_path, active)
        extracted = extract_text_from_pdf(authorized_path, language_hint=language_hint)
    except (PageRetrievalError, ValueError, OSError):
        return None, "extraction_unavailable"
    for page in extracted.get("pages", []):
        if page.get("page_number") == page_number and str(page.get("text") or "").strip():
            return str(page["text"]), str(page.get("extraction_method") or "ocr")
    return None, "extraction_unavailable"


def retrieve_pages(
    page_numbers: int | list[int] | tuple[int, ...],
    *,
    user_id: int,
    pdf_path: str | Path | None = None,
    language_hint: str = "auto",
) -> dict[str, Any]:
    """Return exact page chunks in requested page order.

    PDF page numbering is one-based.  ``pdf_path`` is optional and is only
    used for OCR fallback after ownership validation by upload-directory path.
    """
    if isinstance(page_numbers, int):
        requested = [page_numbers]
    else:
        requested = list(page_numbers)
    if not requested or any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in requested):
        raise PageRetrievalError("PDF page numbers are one-based and must be positive integers.")

    active = get_active_document(user_id)
    if not active or not active.get("active_doc_id") or not active.get("active_filename"):
        return {
            "status": "error", "active_document": None, "document_id": None,
            "requested_pages": requested, "pages": [], "retrieved_chunks": [],
            "evidence": [], "message": "No active PDF uploaded.",
        }

    total_pages = active.get("total_pages")
    if not isinstance(total_pages, int) or isinstance(total_pages, bool) or total_pages < 1:
        return {
            "status": "error", "active_document": active["active_filename"],
            "document_id": active["active_doc_id"], "requested_pages": requested,
            "total_pages": total_pages, "invalid_pages": invalid_pages,
            "pages": [], "retrieved_chunks": [], "evidence": [],
            "message": "The physical page count for the active PDF is unavailable.",
        }

    invalid_pages = [page for page in requested if page > total_pages]
    if invalid_pages:
        valid_pages = [page for page in requested if page <= total_pages]
        invalid_label = ", ".join(map(str, invalid_pages))
        invalid_noun = "Page" if len(invalid_pages) == 1 else "Pages"
        invalid_verb = "is" if len(invalid_pages) == 1 else "are"
        valid_clause = (
            f"Pages {', '.join(map(str, valid_pages))} exist, but "
            if valid_pages else ""
        )
        return {
            "status": "error", "active_document": active["active_filename"],
            "document_id": active["active_doc_id"], "requested_pages": requested,
            "pages": [], "retrieved_chunks": [], "evidence": [],
            "message": (
                f"This PDF contains only {total_pages} pages. "
                f"{valid_clause}{invalid_noun} {invalid_label} "
                f"{invalid_verb} not present in the current document."
            ),
        }

    records = _record_chunks(user_id, active)
    by_page: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_page.setdefault(int(record["page_number"]), []).append(record)

    known_pages = set(by_page)
    document_dir = get_user_debug_dir(user_id) / Path(active["active_filename"]).stem
    clean_pages_dir = document_dir / "clean_pages"
    if clean_pages_dir.is_dir():
        known_pages.update(
            int(path.stem.removeprefix("page_"))
            for path in clean_pages_dir.glob("page_*.txt")
            if path.stem.removeprefix("page_").isdigit()
        )

    pages = []
    all_chunks = []
    evidence = []
    unavailable_pages = []
    out_of_range_pages = []
    for page_number in requested:
        chunks = list(by_page.get(page_number, []))
        extraction_method = "indexed"
        if not chunks:
            page_text = _debug_page_text(user_id, active, page_number)
            if page_text and page_text.strip():
                extraction_method = "stored_page_text"
                chunks = [{
                    "chunk_id": None, "file_name": active["active_filename"],
                    "page_number": page_number, "text": page_text, "score": 1.0,
                    "doc_id": active["active_doc_id"], "source": active["active_filename"],
                    "page_chunk_id": 1, "paragraph_ids": None, "start_char": 0,
                    "end_char": len(page_text), "page_language": active.get("evidence_language"),
                    "page_language_confidence": None, "language": active.get("evidence_language"),
                    "language_confidence": None,
                }]
            else:
                ocr_text, extraction_method = _ocr_page_text(
                    user_id, page_number, active=active,
                    pdf_path=pdf_path, language_hint=language_hint,
                )
                if ocr_text:
                    chunks = [{
                        "chunk_id": None, "file_name": active["active_filename"],
                        "page_number": page_number, "text": ocr_text, "score": 1.0,
                        "doc_id": active["active_doc_id"], "source": active["active_filename"],
                        "page_chunk_id": 1, "paragraph_ids": None, "start_char": 0,
                        "end_char": len(ocr_text), "page_language": active.get("evidence_language"),
                        "page_language_confidence": None, "language": active.get("evidence_language"),
                        "language_confidence": None,
                    }]
                else:
                    if page_number in known_pages:
                        unavailable_pages.append(page_number)
                        extraction_method = "extraction_unavailable"
                    else:
                        out_of_range_pages.append(page_number)
                        extraction_method = "out_of_range"

        chunks.sort(key=lambda item: (
            int(item.get("page_chunk_id") or 0),
            int(item.get("start_char") or 0),
            int(item.get("chunk_id") or 0),
        ))
        if chunks:
            pages.append({
                "page_number": page_number,
                "extraction_method": extraction_method,
                "text": "\n\n".join(chunk["text"] for chunk in chunks),
                "chunks": chunks,
                "response_language": active.get("response_language"),
                "evidence_language": active.get("evidence_language"),
            })
            all_chunks.extend(chunks)
            evidence.extend({
                "page_number": page_number,
                "chunk_id": chunk.get("chunk_id"),
                "file_name": chunk.get("file_name"),
                "snippet": chunk["text"][:250],
                "image_path": f"/api/page-images/{active['active_doc_id']}/page_{page_number}.png",
            } for chunk in chunks)

    if unavailable_pages and out_of_range_pages:
        status = "partial" if pages else "error"
        message = (
            "Page text extraction is unavailable for: "
            + ", ".join(map(str, unavailable_pages))
            + "; requested pages are out of range: "
            + ", ".join(map(str, out_of_range_pages))
            + "."
        )
    elif unavailable_pages:
        status = "partial" if pages else "error"
        message = "Page text extraction is unavailable for: " + ", ".join(map(str, unavailable_pages)) + "."
    elif out_of_range_pages:
        status = "partial" if pages else "error"
        message = "Requested PDF pages are out of range: " + ", ".join(map(str, out_of_range_pages)) + "."
    elif pages:
        status = "success"
        message = "Requested PDF pages retrieved successfully."
    else:
        status = "error"
        message = "Requested PDF pages are out of range."

    return {
        "status": status,
        "active_document": active["active_filename"],
        "document_id": active["active_doc_id"],
        "requested_pages": requested,
        "pages": pages,
        "retrieved_chunks": all_chunks,
        "evidence": evidence,
        "message": message,
    }
