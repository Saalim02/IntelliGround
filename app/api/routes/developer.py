"""Developer-only evaluation, review, guardrail, and observability APIs."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.models.auth_models import User
from app.models.evaluation_models import HumanReview
from app.services.evaluation.evaluation_storage_service import (
    append_evaluation_dataset,
    apply_human_review,
    load_evaluation,
    load_human_review,
    list_evaluations,
    save_human_review,
)
from app.services.evaluation.confidence import configure_evaluation, evaluation_config
from app.services.emotion_state_service import get_settings, save_settings
from app.services.observability.langsmith_service import (
    create_connectivity_trace,
    instrumentation_coverage,
    observability_metrics,
    recent_traces,
    set_tracing_enabled,
    tracing_status,
)
from app.services.security.developer_access import get_developer_user
from app.services.llm.provider_factory import provider_status
from app.services.security.document_guardrails import inspect_retrieved_chunks
from app.services.security.guardrails import validate_question
from app.services.user_storage_service import get_user_evaluation_dir


router = APIRouter(prefix="/developer", tags=["Developer Settings"])


class GuardrailTestRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_text: str = Field(default="", max_length=12000)


class ObservabilityToggleRequest(BaseModel):
    enabled: bool | None = None


class EvaluationConfigRequest(BaseModel):
    auto_approve_threshold: float = Field(ge=0, le=1)
    review_recommended_threshold: float = Field(ge=0, le=1)
    critical_grounding_threshold: float = Field(ge=0, le=1)
    weights: dict[str, float] = Field(default_factory=dict)


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
    for path in sorted(evaluation_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        evaluation = load_evaluation(path.stem, user_id=user.id)
        if evaluation:
            records.append(_evaluation_summary(evaluation, load_human_review(path.stem, user.id)))
    pending = [item for item in records if item["review_status"] == "pending"]
    return {
        "evaluations": records,
        "pending_reviews": pending,
        "counts": {"evaluations": len(records), "pending_reviews": len(pending)},
        "evaluation_config": evaluation_config(),
        "observability": {
            **tracing_status(),
            "metrics": observability_metrics(),
            "coverage": instrumentation_coverage(),
            "trace_verified": observability_metrics()["total_recorded_traces"] > 0,
            "trace_note": "Metrics are process-local and unavailable after a backend restart." if observability_metrics()["available"] else "No trace has been recorded in this backend process.",
        },
    }


@router.get("/evaluation-config")
def get_evaluation_config(user: User = Depends(get_developer_user)):
    del user
    return evaluation_config()


@router.put("/evaluation-config")
def update_evaluation_config(
    request: EvaluationConfigRequest,
    user: User = Depends(get_developer_user),
):
    del user
    try:
        return configure_evaluation(request.model_dump())
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@router.get("/emotion-settings")
def emotion_settings(user: User = Depends(get_developer_user)):
    del user
    return get_settings()

@router.put("/emotion-settings")
def update_emotion_settings(payload: dict, user: User = Depends(get_developer_user)):
    del user
    try:
        return save_settings(payload)
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/observability")
def observability_dashboard(user: User = Depends(get_developer_user)):
    del user
    status = tracing_status()
    metrics = observability_metrics()
    return {
        "status": status,
        "metrics": metrics,
        "coverage": instrumentation_coverage(),
        "trace_verified": metrics["total_recorded_traces"] > 0,
    }


@router.get("/observability/traces")
def observability_traces(user: User = Depends(get_developer_user)):
    del user
    return recent_traces()


@router.get("/providers")
def provider_dashboard(user: User = Depends(get_developer_user)):
    del user
    return {"providers": provider_status()}


@router.post("/observability/test")
def test_observability(user: User = Depends(get_developer_user)):
    del user
    return {
        "status": "success" if create_connectivity_trace() else "unavailable",
        "configuration": tracing_status(),
        "metrics": observability_metrics(),
    }


@router.post("/observability/toggle")
def toggle_observability(
    request: ObservabilityToggleRequest,
    user: User = Depends(get_developer_user),
):
    del user
    set_tracing_enabled(request.enabled)
    return {"status": "success", "configuration": tracing_status()}


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
    review.reviewer_id = user.id
    from datetime import datetime, timezone
    review.reviewed_at = datetime.now(timezone.utc)
    review.original_decision = evaluation.decision
    review.original_answer = evaluation.answer
    save_human_review(review, user_id=user.id)
    evaluation = apply_human_review(evaluation, review, user_id=user.id)
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
