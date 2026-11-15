from app.services.document_intelligence_service import generate_document_questions
from app.services.document_registry_service import get_active_document


def generate_suggested_questions(
    api_key: str,
    user_id: int | None = None,
    response_language: str | None = None,
):
    """
    Generates useful suggested questions
    based on uploaded PDF content.
    """

    # -----------------------------
    # ACTIVE DOCUMENT
    # -----------------------------
    active_doc = get_active_document(user_id)

    if not active_doc:

        return {
            "status": "error",
            "active_document": None,
            "response_language": response_language,
            "suggested_questions": [],
            "message": "No active PDF uploaded."
        }

    active_filename = active_doc["active_filename"]
    try:
        grounded = generate_document_questions(
            user_id,
            api_key,
            response_language=response_language,
        )
        cleaned_questions = [item["question"] for item in grounded["questions"]]

        return {
            "status": "success",
            "active_document": active_filename,
            "response_language": grounded.get("response_language", response_language),
            "suggested_questions": cleaned_questions,
            "message": grounded.get("message", "Suggested questions generated successfully."),
        }
    except Exception as exc:

        return {
            "status": "error",
            "active_document": active_filename,
            "response_language": response_language,
            "suggested_questions": [],
            "message": str(exc) or "Failed to generate grounded suggested questions.",
        }
