import asyncio
import logging
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
)

from app.services.validation_service import validate_pdf_file

from app.models.auth_models import User
from app.services.security.auth_dependencies import get_current_user

from app.services.document_cleanup_service import (
    cleanup_failed_upload,
)

from app.utils.file_handler import save_uploaded_file

from app.services.pdf_service import (
    extract_text_from_pdf,
    SUPPORTED_OCR_LANGUAGES,
)
from app.services.language_detection_service import detect_text_language

from app.models.response_models import UploadPDFResponse

from app.utils.metadata_builder import build_chunk_metadata

from app.utils.debug_exporter import export_debug_files

from app.services.document_registry_service import (
    save_active_document,
    get_active_document,
    clear_active_document,
)

from app.services.pdf_image_service import (
    generate_page_images,
)

from app.services.summary_storage_service import (
    save_summary_source,
)

from app.services.vector_store_service import (
    convert_chunks_to_documents,
    begin_vectorstore_transaction,
)
from app.services.upload_progress_service import (
    complete_job,
    fail_job,
    get_job,
    normalize_job_id,
    start_job,
    update_job,
)
from app.services.performance_service import measure

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload-pdf",
    response_model=UploadPDFResponse,
)
async def upload_pdf(
    file: UploadFile = File(...),
    language: str = Form(default="auto"),
    job_id: str = Form(default=""),
    current_user: User = Depends(get_current_user),
):
    saved_path = None
    original_filename = None
    doc_id = None
    vectorstore_transaction = None
    debug_transaction = None
    previous_registry = None
    registry_saved = False
    try:
        job_id = normalize_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if get_job(current_user.id, job_id) is not None:
        raise HTTPException(status_code=409, detail="Upload job already exists.")
    start_job(current_user.id, job_id, file.filename)

    try:
        # -----------------------------------
        # VALIDATE OCR LANGUAGE
        # -----------------------------------
        if language not in SUPPORTED_OCR_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported OCR language: {language}. "
                    f"Supported languages: "
                    f"{', '.join(SUPPORTED_OCR_LANGUAGES.keys())}"
                ),
            )

        # -----------------------------------
        # VALIDATE PDF
        # -----------------------------------
        with measure("ingestion.validation"):
            await validate_pdf_file(file)
        update_job(current_user.id, job_id, 10, "PDF received and validated")

        # -----------------------------------
        # SAVE PDF
        # -----------------------------------
        with measure("ingestion.upload_save"):
            (
                original_filename,
                saved_path,
                file_size_mb,
            ) = await save_uploaded_file(file, user_id=current_user.id)

        # -----------------------------------
        # EXTRACT PDF TEXT
        # -----------------------------------
        try:
            with measure("ingestion.extraction"):
                pdf_data = await asyncio.to_thread(
                    extract_text_from_pdf, saved_path, language_hint=language,
                )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        pages_data = pdf_data["pages"]
        total_pages = pdf_data["total_pages"]
        update_job(current_user.id, job_id, 20, "Text extraction completed")

        extracted_text = "\n".join(
            page["text"]
            for page in pages_data
        )
        detected_evidence_language = detect_text_language(extracted_text)["language"]
        if detected_evidence_language in {"und", "mixed"}:
            detected_evidence_language = {
                "hin": "hi", "kan": "kn", "tam": "ta", "tel": "te",
                "mal": "ml", "mar": "mr", "guj": "gu", "ben": "bn",
                "pan": "pa", "urd": "ur", "eng": "en", "auto": "en",
            }.get(language, "en")

        # -----------------------------------
        # LOW TEXT / EXTRACTION QUALITY CHECK
        # -----------------------------------
        if (
            not extracted_text.strip()
            or len(extracted_text.strip()) < 50
        ):
            cleanup_failed_upload(
                user_id=current_user.id,
                saved_path=saved_path,
                filename=original_filename,
            )

            saved_path = None
            fail_job(current_user.id, job_id, "Insufficient readable text was extracted from the PDF.")

            return UploadPDFResponse(
                status="error",
                filename=original_filename,
                saved_filename="",
                file_size_mb=file_size_mb,
                total_pages=total_pages,
                extracted_characters=len(extracted_text),
                num_chunks=0,
                selected_chunk_size=0,
                selected_chunk_overlap=0,
                preview_chunk=None,
                message=(
                    "Insufficient readable text was extracted "
                    "from this PDF, even after the selected "
                    "extraction/OCR process. Try selecting the "
                    "correct document language or uploading "
                    "a clearer PDF."
                ),
                job_id=job_id,
            )

        # -----------------------------------
        # DOCUMENT ID
        # -----------------------------------
        doc_id = str(uuid.uuid4())
        update_job(current_user.id, job_id, 35, "Text cleaned and chunked", document_id=doc_id)

        # -----------------------------------
        # GENERATE PAGE IMAGES
        # -----------------------------------
        with measure("ingestion.page_images"):
            await asyncio.to_thread(
                generate_page_images, saved_path, doc_id, user_id=current_user.id,
                progress_callback=lambda completed, total: update_job(
                    current_user.id, job_id, 20 + round(10 * completed / total),
                    f"Processing page images ({completed}/{total})", document_id=doc_id,
                ),
            )

        # -----------------------------------
        # SAVE SUMMARY SOURCE
        # -----------------------------------
        with measure("ingestion.summary_source_persistence"):
            await asyncio.to_thread(
                save_summary_source, doc_id, extracted_text, user_id=current_user.id,
            )

        # -----------------------------------
        # BUILD CHUNK METADATA
        # -----------------------------------
        with measure("ingestion.chunking_metadata"):
            metadata, chunk_configs = await asyncio.to_thread(
                build_chunk_metadata, pages_data, original_filename, doc_id,
            )

        # -----------------------------------
        # CONVERT TO LANGCHAIN DOCUMENTS
        # -----------------------------------
        documents = await asyncio.to_thread(
            convert_chunks_to_documents,
            metadata,
            original_filename,
        )

        if not documents:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable text chunks could be created "
                    "from this PDF."
                ),
            )

        # -----------------------------------
        # CREATE AND SAVE VECTOR STORE
        # -----------------------------------
        previous_registry = get_active_document(current_user.id)
        with measure("ingestion.embedding_and_index_build"):
            vectorstore_transaction = await asyncio.to_thread(
                begin_vectorstore_transaction, documents, user_id=current_user.id,
            )
        update_job(current_user.id, job_id, 50, "Embedding generation completed", document_id=doc_id)
        vectorstore_transaction.__enter__()
        vectorstore_transaction.install()
        update_job(current_user.id, job_id, 75, "Vector and BM25 indexes installed", document_id=doc_id)

        # -----------------------------------
        # EXPORT DEBUG FILES
        # -----------------------------------
        with measure("ingestion.debug_metadata_export"):
            debug_transaction = await asyncio.to_thread(
                export_debug_files, original_filename, pages_data, metadata,
                user_id=current_user.id, commit=False,
            )

        # Install both filesystem replacements before changing the registry.
        # Their backups remain available until every state change succeeds.
        debug_transaction.install()
        update_job(current_user.id, job_id, 90, "Metadata and page-image processing completed", document_id=doc_id)

        # -----------------------------------
        # ACTIVE DOCUMENT REGISTRY
        # -----------------------------------
        save_active_document(
            doc_id,
            original_filename,
            user_id=current_user.id,
            input_language=language,
            evidence_language=detected_evidence_language,
            response_language=(
                "en" if language in {"auto", "eng"} else {
                    "hin": "hi", "kan": "kn", "tam": "ta", "tel": "te",
                    "mal": "ml", "mar": "mr", "guj": "gu", "ben": "bn",
                    "pan": "pa", "urd": "ur",
                }.get(language, "en")
            ),
            total_pages=total_pages,
        )
        registry_saved = True
        # Installation is reversible.  Keep both backups until the registry
        # and both filesystem replacements have been marked committed.
        vectorstore_transaction.mark_committed()
        debug_transaction.mark_committed()
        vectorstore_transaction.cleanup_backup()
        debug_transaction.cleanup_backup()
        vectorstore_transaction.__exit__(None, None, None)
        vectorstore_transaction = None
        debug_transaction = None
        complete_job(current_user.id, job_id, document_id=doc_id)

        # -----------------------------------
        # DYNAMIC CHUNK STATISTICS
        # -----------------------------------
        avg_chunk_size = (
            sum(
                config["chunk_size"]
                for config in chunk_configs
            ) // len(chunk_configs)
            if chunk_configs
            else 0
        )

        avg_chunk_overlap = (
            sum(
                config["chunk_overlap"]
                for config in chunk_configs
            ) // len(chunk_configs)
            if chunk_configs
            else 0
        )

        # -----------------------------------
        # SUCCESS RESPONSE
        # -----------------------------------
        return UploadPDFResponse(
            status="success",
            filename=original_filename,
            saved_filename=saved_path.replace(
                "\\",
                "/",
            ).split("/")[-1],
            file_size_mb=file_size_mb,
            total_pages=total_pages,
            extracted_characters=len(extracted_text),
            num_chunks=len(metadata),
            selected_chunk_size=avg_chunk_size,
            selected_chunk_overlap=avg_chunk_overlap,
            preview_chunk=(
                metadata[0]["text"]
                if metadata
                else None
            ),
            input_language=language,
            evidence_language=detected_evidence_language,
            response_language=(
                "en" if language in {"auto", "eng"} else {
                    "hin": "hi", "kan": "kn", "tam": "ta", "tel": "te",
                    "mal": "ml", "mar": "mr", "guj": "gu", "ben": "bn",
                    "pan": "pa", "urd": "ur",
                }.get(language, "en")
            ),
            message=(
                "PDF uploaded, extracted, chunked, "
                "embedded, and stored successfully. "
                "Document metadata and page images "
                "were also generated."
            ),
            job_id=job_id,
        )

    except HTTPException:
        fail_job(current_user.id, job_id, "PDF processing could not be completed.")
        if debug_transaction:
            debug_transaction.rollback(force=True)
        if vectorstore_transaction:
            vectorstore_transaction.__exit__(HTTPException, None, None)
        if registry_saved:
            if previous_registry:
                save_active_document(
                    previous_registry["active_doc_id"],
                    previous_registry["active_filename"],
                    user_id=current_user.id,
                    input_language=previous_registry.get("input_language"),
                    evidence_language=previous_registry.get("evidence_language"),
                    response_language=previous_registry.get("response_language"),
                    total_pages=previous_registry.get("total_pages"),
                )
            else:
                clear_active_document(current_user.id)
        cleanup_failed_upload(
            user_id=current_user.id,
            doc_id=doc_id,
            saved_path=saved_path,
            filename=None,
        )
        raise

    except Exception as exc:
        fail_job(current_user.id, job_id, "PDF processing could not be completed.")
        logger.exception(
            "Unexpected error while processing PDF upload."
        )

        if debug_transaction:
            debug_transaction.rollback(force=True)
        if vectorstore_transaction:
            vectorstore_transaction.__exit__(Exception, exc, None)
        if registry_saved:
            if previous_registry:
                save_active_document(
                    previous_registry["active_doc_id"],
                    previous_registry["active_filename"],
                    user_id=current_user.id,
                    input_language=previous_registry.get("input_language"),
                    evidence_language=previous_registry.get("evidence_language"),
                    response_language=previous_registry.get("response_language"),
                    total_pages=previous_registry.get("total_pages"),
                )
            else:
                clear_active_document(current_user.id)

        cleanup_failed_upload(
            user_id=current_user.id,
            doc_id=doc_id,
            saved_path=saved_path,
            filename=None,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the uploaded PDF.",
        ) from exc


@router.get("/upload-pdf/{job_id}/progress")
def upload_progress(job_id: str, current_user: User = Depends(get_current_user)):
    try:
        normalized = normalize_job_id(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    progress = get_job(current_user.id, normalized)
    if progress is None:
        raise HTTPException(status_code=404, detail="Upload job not found.")
    return progress
