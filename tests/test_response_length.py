import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.models.query_models import QueryRequest
from app.models.auth_models import User
from app.main import app
from app.services.security.auth_dependencies import get_current_user
from app.services.response_length import (
    document_generation_plan,
    response_length_instruction,
    response_length_max_tokens,
)
from app.services import document_intelligence_service as intelligence


def test_document_generation_targets_follow_response_length_and_evidence_capacity():
    chunks = [{"page_number": index, "text": "A meaningful evidence passage with facts."} for index in range(24)]
    assert document_generation_plan(chunks, "short")["target_topics"] == 5
    assert document_generation_plan(chunks, "short")["target_questions"] == 5
    assert document_generation_plan(chunks, "medium")["target_topics"] == 10
    assert document_generation_plan(chunks, "large")["target_topics"] == 15
    assert document_generation_plan(chunks, "large")["max_questions"] == 20


def test_document_generation_plan_returns_fewer_for_sparse_evidence():
    plan = document_generation_plan([{"page_number": 1, "text": "One short but meaningful fact."}], "large")
    assert plan["target_topics"] < 15
    assert plan["target_questions"] < 15


@pytest.mark.parametrize(("length", "expected_max"), [("short", 5), ("medium", 10), ("large", 20)])
def test_question_generation_caps_grounded_questions_by_length(monkeypatch, length, expected_max):
    chunks = [
        {"chunk_id": index, "page_number": index, "text": f"Meaningful evidence passage {index} explains a distinct documented fact.", "doc_id": "doc"}
        for index in range(1, 25)
    ]
    monkeypatch.setattr(intelligence, "_base", lambda *_args: ({"active_filename": "source.pdf", "evidence_language": "en"}, chunks, "en"))
    captured = {}

    def invoke(_api_key, _language, _chunks, task, operation="document_intelligence"):
        captured["task"] = task
        return {
            "questions": [{
                "question": f"What documented fact is covered in passage {index}?",
                "question_type": "factual", "difficulty": "easy", "answerable": True,
                "evidence": [{"chunk_id": index}],
            } for index in range(1, 25)],
            "unsupported_items": [],
        }

    monkeypatch.setattr(intelligence, "_invoke", invoke)
    result = intelligence.generate_document_questions(1, "test-key", response_language="en", response_length=length)
    assert len(result["questions"]) == expected_max
    assert result["response_length"] == length
    assert f"target {min(expected_max, 5 if length == 'short' else 10 if length == 'medium' else 15)}" in captured["task"]


def test_response_length_defaults_to_medium():
    assert QueryRequest(question="What is this document about?").response_length == "medium"


def test_response_length_rejects_unknown_values():
    with pytest.raises(ValidationError):
        QueryRequest(question="Explain page 1", response_length="verbose")


@pytest.mark.parametrize("value", ["short", "medium", "large"])
def test_each_mode_has_distinct_generation_policy_and_budget(value):
    assert response_length_max_tokens(value) > 0
    assert response_length_instruction(value)


def test_length_policy_preserves_grounding_constraint():
    assert "evidence" in response_length_instruction("large").lower()
    assert "invent" in response_length_instruction("large").lower()


@pytest.mark.parametrize("endpoint", ["summary", "topics", "questions"])
@pytest.mark.parametrize("length", ["short", "medium", "large"])
def test_all_document_intelligence_endpoints_forward_response_length(monkeypatch, endpoint, length):
    user = User(id=901, email="length-contract@example.com", is_active=True)
    app.dependency_overrides[get_current_user] = lambda: user
    captured = {}
    try:
        monkeypatch.setattr(
            "app.api.routes.document_intelligence.resolve_authorized_api_key",
            lambda **_kwargs: "test-key",
        )

        def response(kind, response_language, response_length):
            captured[kind] = response_length
            base = {
                "status": "success", "active_document": "source.pdf",
                "response_language": response_language, "response_length": response_length,
                "unsupported_items": [], "message": "ok",
            }
            if kind == "summary":
                base.update({"short_summary": "A summary.", "detailed_summary": "A detailed summary.", "evidence": [], "main_conclusions": [], "topics": []})
            elif kind == "topics":
                base["topics"] = []
            else:
                base["questions"] = []
            return base

        monkeypatch.setattr(
            "app.api.routes.document_intelligence.generate_document_summary",
            lambda _user_id, _api_key, response_language=None, response_length="medium": response("summary", response_language, response_length),
        )
        monkeypatch.setattr(
            "app.api.routes.document_intelligence.generate_document_topics",
            lambda _user_id, _api_key, response_language=None, response_length="medium": response("topics", response_language, response_length),
        )
        monkeypatch.setattr(
            "app.api.routes.document_intelligence.generate_document_questions",
            lambda _user_id, _api_key, response_language=None, response_length="medium": response("questions", response_language, response_length),
        )

        result = TestClient(app).post(
            f"/api/document/{endpoint}",
            json={"document_id": "source-doc", "document_language": "te", "response_language": "ta", "response_length": length},
        )
        assert result.status_code == 200, result.text
        assert captured[endpoint] == length
        assert result.json()["response_length"] == length
    finally:
        app.dependency_overrides.pop(get_current_user, None)
