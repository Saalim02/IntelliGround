from types import SimpleNamespace

import pytest

from app.models.query_models import QueryRequest
from app.models.evaluation_models import EvaluationDecision
from app.models.security_models import SecurityCheckResult, SecurityDecision
from app.services.rag_orchestration_service import run_document_query
from app.services.conversation_service import NO_DOCUMENT_ID, NO_DOCUMENT_NAME, validate_conversation_document
from app.services.security.semantic_guardrail import SemanticSecurityDecision


LANGUAGES = ["en", "kn", "hi", "ta", "te", "ml", "mr", "gu", "bn", "pa", "ur"]


def _deps(captured=None, *, answer="Grounded answer"):
    safe = SecurityCheckResult(decision=SecurityDecision.ALLOW, is_safe=True, risk_score=0, reasons=[])
    semantic = SemanticSecurityDecision(decision=SecurityDecision.ALLOW, is_safe=True, risk_score=0, reasons=[])

    def retrieve(query, user_id):
        if captured is not None:
            captured["retrieval_query"] = query
        return {
            "status": "success", "active_document": "source.pdf", "confidence": "medium",
            "best_score": 0.8, "average_score": 0.7,
            "retrieved_chunks": [{"chunk_id": 1, "page_number": 2, "text": "Supported evidence", "file_name": "source.pdf"}],
            "dense_candidate_count": 1, "sparse_candidate_count": 1,
            "fused_candidate_count": 1, "reranked_candidate_count": 1,
            "retrieval_stage_timings": {}, "reranker_enabled": False, "reranker_status": "disabled",
        }

    def generate(**kwargs):
        if captured is not None:
            captured["answer_kwargs"] = kwargs
        return answer

    evaluation = SimpleNamespace(
        user_id=None,
        decision=EvaluationDecision.AUTO_APPROVE,
        model_dump=lambda: {"decision": "AUTO_APPROVE"},
    )
    return {
        "validate_question": lambda _question: safe,
        "classify_question_semantically": lambda **_kwargs: semantic,
        "resolve_authorized_api_key": lambda **_kwargs: "test-key",
        "retrieve_relevant_chunks": retrieve,
        "get_active_document": lambda _user_id: {"active_doc_id": "doc-1"},
        "generate_grounded_answer": generate,
        "evaluate_rag_response": lambda **_kwargs: evaluation,
        "save_evaluation": lambda *_args, **_kwargs: None,
    }


@pytest.mark.parametrize("language", LANGUAGES)
def test_all_languages_route_greetings_without_retrieval(language):
    captured = {}
    result = run_document_query(
        QueryRequest(question="hello", response_language=language),
        SimpleNamespace(id=1), endpoint="/api/voice", dependencies=_deps(captured),
    )
    assert result["status"] == "success"
    assert result["question_type"] == "CONVERSATIONAL"
    assert result["answer"]
    assert "retrieval_query" not in captured


def test_voice_pdf_question_keeps_grounded_rag_and_language():
    captured = {}
    result = run_document_query(
        QueryRequest(question="What does the uploaded document say?", response_language="hi"),
        SimpleNamespace(id=1), endpoint="/api/voice", dependencies=_deps(captured),
    )
    assert result["status"] == "success"
    assert captured["answer_kwargs"]["response_language"] == "hi"
    assert captured["answer_kwargs"]["retrieved_chunks"]


def test_voice_out_of_scope_is_not_sent_to_retrieval():
    captured = {}
    result = run_document_query(
        QueryRequest(question="What is the weather today?", response_language="bn"),
        SimpleNamespace(id=1), endpoint="/api/voice", dependencies=_deps(captured),
    )
    assert result["status"] == "out_of_scope"
    assert result["message"]
    assert "retrieval_query" not in captured


@pytest.mark.parametrize(
    ("question", "kind"),
    [("Hello", "CONVERSATIONAL"), ("How are you?", "GENERAL_CONVERSATION"), ("How do I upload a PDF?", "APPLICATION_HELP")],
)
def test_voice_without_document_avoids_all_document_services(question, kind):
    captured = {}
    dependencies = _deps(captured)
    dependencies["get_active_document"] = lambda _user_id: None
    dependencies["resolve_authorized_api_key"] = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("RAG authorization should not run"))
    dependencies["retrieve_relevant_chunks"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run"))
    dependencies["classify_question_semantically"] = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("semantic RAG guardrail should not run"))

    result = run_document_query(
        QueryRequest(question=question, response_language="hi"),
        SimpleNamespace(id=1), endpoint="/api/voice", dependencies=dependencies,
    )

    assert result["question_type"] == kind
    assert result["answer"]
    assert result["evidence"] == []
    assert result["evaluation"] is None
    assert "retrieval_query" not in captured


def test_no_document_general_conversation_is_classified_before_document_check():
    captured = {}
    dependencies = _deps(captured)
    dependencies["get_active_document"] = lambda _user_id: None
    dependencies["resolve_authorized_api_key"] = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("RAG authorization should not run"))
    dependencies["classify_question_semantically"] = lambda **_kwargs: (_ for _ in ()).throw(AssertionError("semantic RAG guardrail should not run"))
    dependencies["retrieve_relevant_chunks"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retrieval should not run"))

    result = run_document_query(
        QueryRequest(question="Can you hear me?", response_language="en"),
        SimpleNamespace(id=1), endpoint="/api/voice", dependencies=dependencies,
    )
    assert result["question_type"] == "AUDIO_CHECK"
    assert result["answer"]
    assert result["evidence"] == []
    assert result["confidence"] is None
    assert result["evaluation"] is None


def test_voice_document_question_without_document_requests_upload_without_confidence():
    dependencies = _deps({})
    dependencies["get_active_document"] = lambda _user_id: None
    result = run_document_query(
        QueryRequest(question="What is on page 4?", response_language="ta"),
        SimpleNamespace(id=1), endpoint="/api/voice", dependencies=dependencies,
    )
    assert result["status"] == "document_required"
    assert result["answer"] is None
    assert result["confidence"] is None
    assert result["evidence"] == []
    assert "PDF" in result["message"]


def test_no_document_conversation_is_allowed_only_without_active_document():
    conversation = SimpleNamespace(document_id=NO_DOCUMENT_ID, document_name=NO_DOCUMENT_NAME)
    result = validate_conversation_document(conversation, None)
    assert result["active_doc_id"] == NO_DOCUMENT_ID

    with pytest.raises(Exception, match="no active document"):
        validate_conversation_document(conversation, {"active_doc_id": "doc-a", "active_filename": "a.pdf"})
