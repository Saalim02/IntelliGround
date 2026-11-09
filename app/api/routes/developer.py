"""Developer-only evaluation, review, guardrail, and observability APIs."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.models.auth_models import User
from app.models.evaluation_models import HumanReview
from app.services.evaluation.evaluation_storage_service import (
    append_evaluation_dataset,
    load_evaluation,
    load_human_review,
    save_human_review,
)
from app.services.observability.langsmith_service import tracing_status
from app.services.security.developer_access import get_developer_user
from app.services.security.document_guardrails import inspect_retrieved_chunks
from app.services.security.guardrails import validate_question
from app.services.user_storage_service import get_user_evaluation_dir


router = APIRouter(prefix="/developer", tags=["Developer Settings"])


class GuardrailTestRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_text: str = Field(default="", max_length=12000)


def _evaluation_summary(evaluation, review):
    data = evaluation.model_dump(mode="json")
    data["review"] = review.model_dump(mode="json") if review else None
    data["review_status"] = "reviewed" if review else (
        "pending" if evaluation.decision.value != "AUTO_APPROVE" else "not_required"
    )
    return data


@router.get("/dashboard")
def developer_dashboard(user: User = Depends(get_developer_user)):
    evaluation_dir = get_user_evaluation_dir(user.id)
    records = []
    for path in sorted(evaluation_dir.glob("*.json")):
        evaluation = load_evaluation(path.stem, user_id=user.id)
        if evaluation:
            records.append(_evaluation_summary(evaluation, load_human_review(path.stem, user.id)))
    pending = [item for item in records if item["review_status"] == "pending"]
    return {
        "evaluations": records,
        "pending_reviews": pending,
        "counts": {"evaluations": len(records), "pending_reviews": len(pending)},
        "observability": {
            **tracing_status(),
            "trace_verified": False,
            "trace_note": "No real application trace has been verified yet.",
        },
    }


@router.post("/evaluations/{evaluation_id}/review")
def review_evaluation(
    evaluation_id: str,
    review: HumanReview,
    user: User = Depends(get_developer_user),
):
    evaluation = load_evaluation(evaluation_id, user_id=user.id)
    if evaluation is None or review.evaluation_id != evaluation_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    save_human_review(review, user_id=user.id)
    append_evaluation_dataset(evaluation, review, user_id=user.id)
    return {"status": "success", "evaluation": _evaluation_summary(evaluation, review)}


@router.post("/guardrails/test")
def test_guardrails(
    request: GuardrailTestRequest,
    user: User = Depends(get_developer_user),
):
    question_result = validate_question(request.question)
    document_result = inspect_retrieved_chunks(
        [{"chunk_id": 0, "page_number": 0, "text": request.document_text}]
    ) if request.document_text else {"is_suspicious": False, "suspicious_chunks": [], "clean_chunks": []}
    return {
        "question": {"decision": question_result.decision.value, "is_safe": question_result.is_safe, "risk_score": question_result.risk_score, "reasons": question_result.reasons},
        "document": document_result,
    }
