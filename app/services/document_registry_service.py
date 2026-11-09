
import json
import os
import tempfile
from pathlib import Path

from app.core.config import BASE_DIR
from app.services.user_storage_service import get_user_registry_path


LEGACY_REGISTRY_PATH = (
    BASE_DIR / "vectorstore" / "document_registry.json"
)


def _get_registry_path(user_id: int | None = None) -> Path:
    """
    Return the document registry path.

    user_id provided:
        Use isolated per-user registry.

    user_id omitted:
        Preserve legacy global registry for backwards compatibility.
    """

    if user_id is None:
        return LEGACY_REGISTRY_PATH

    return get_user_registry_path(user_id)


def save_active_document(
    doc_id: str,
    filename: str,
    user_id: int | None = None,
    *,
    input_language: str | None = None,
    evidence_language: str | None = None,
    response_language: str | None = None,
    total_pages: int | None = None,
):
    """
    Atomically save the active document registry.

    A temporary file is written in the same directory and then
    atomically replaces the existing registry.

    If writing or replacement fails, the previous registry remains
    intact.
    """

    registry_path = _get_registry_path(user_id)

    registry_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "active_doc_id": doc_id,
        "active_filename": filename,
    }
    for key, value in (
        ("input_language", input_language),
        ("evidence_language", evidence_language),
        ("response_language", response_language),
        ("total_pages", total_pages),
    ):
        if value is not None:
            data[key] = value

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=registry_path.parent,
            prefix=".registry_tmp_",
            suffix=".json",
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            json.dump(
                data,
                temp_file,
                indent=4,
            )

            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(
            temp_path,
            registry_path,
        )

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def update_document_languages(
    user_id: int,
    *,
    response_language: str | None = None,
    input_language: str | None = None,
    evidence_language: str | None = None,
):
    """Update language metadata without changing the active document identity."""
    active = get_active_document(user_id)
    if not active:
        return None
    return save_active_document(
        active["active_doc_id"],
        active["active_filename"],
        user_id=user_id,
        input_language=input_language if input_language is not None else active.get("input_language"),
        evidence_language=evidence_language if evidence_language is not None else active.get("evidence_language"),
        response_language=response_language if response_language is not None else active.get("response_language"),
    )


def get_active_document(
    user_id: int | None = None,
):
    """
    Get the active document for a user.

    Returns None if the registry does not exist.
    """

    registry_path = _get_registry_path(user_id)

    if not registry_path.exists():
        return None

    with registry_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def clear_active_document(
    user_id: int | None = None,
):
    """
    Clear the active document registry for a user.
    """

    registry_path = _get_registry_path(user_id)

    if registry_path.exists():
        registry_path.unlink()
