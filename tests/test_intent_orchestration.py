from types import SimpleNamespace

import pytest

from app.models.evaluation_models import EvaluationDecision
from app.models.query_models import QueryRequest
from app.models.security_models import SecurityCheckResult, SecurityDecision
from app.services import rag_orchestration_service
from app.services.rag_orchestration_service import run_document_query
from app.services.voice_intent_service import IntentClassification, IntentType


def _dependencies(captured):
    safe = SecurityCheckResult(
        decision=SecurityDecision.ALLOW,
        is_safe=True,
        risk_score=0.0,
        reasons=[],
    )
    evaluation = SimpleNamespace(
        user_id=None,
        decision=EvaluationDecision.AUTO_APPROVE,
        model_dump=lambda: {"decision": "AUTO_APPROVE"},
    )

    def hybrid_retrieve(query, user_id):
        captured.setdefault("hybrid", []).append((query, user_id))
        return {
            "status": "success", "active_document": "source.pdf", "document_id": "doc-1",
            "confidence": "high", "best_score": 0.8, "average_score": 0.8,
            "retrieved_chunks": [{
                "chunk_id": 10, "file_name": "source.pdf", "page_number": 2,
                "text": "Hybrid evidence", "score": 0.8,
            }],
        }

    def page_retrieve(page_numbers, user_id):
        captured["pages"] = (list(page_numbers), user_id)
        chunks = [{
            "chunk_id": page * 10, "file_name": "source.pdf", "page_number": page,
            "text": f"Tamil evidence on page {page}", "score": 1.0,
            "doc_id": "doc-1", "source": "source.pdf", "page_chunk_id": 1,
            "language": "ta", "page_language": "ta",
        } for page in page_numbers]
        return {
            "status": "success", "active_document": "source.pdf", "document_id": "doc-1",
            "requested_pages": list(page_numbers), "retrieved_chunks": chunks,
            "evidence": [{
                "page_number": chunk["page_number"], "chunk_id": chunk["chunk_id"],
                "file_name": "source.pdf", "snippet": chunk["text"],
            } for chunk in chunks],
            "pages": [], "message": "Requested PDF pages retrieved successfully.",
        }

    def answer(**kwargs):
        captured["answer"] = kwargs
        samples = {
            "en": "Grounded answer",
            "kn": "ದಾಖಲೆಯ ಆಧಾರದ ಉತ್ತರ",
            "hi": "दस्तावेज़ पर आधारित उत्तर",
            "ta": "ஆவணத்தின் அடிப்படையிலான பதில்",
            "te": "పత్రం ఆధారిత సమాధానం",
            "ml": "രേഖയെ അടിസ്ഥാനമാക്കിയ മറുപടി",
            "mr": "दस्तऐवजावर आधारित उत्तर",
            "gu": "દસ્તાવેજ આધારિત જવાબ",
            "bn": "নথিভিত্তিক উত্তর",
            "pa": "ਦਸਤਾਵੇਜ਼ ਅਧਾਰਿਤ ਜਵਾਬ",
            "ur": "دستاویز پر مبنی جواب",
        }
        return samples.get(kwargs.get("response_language"), "Grounded answer")

    def evaluate(**kwargs):
        captured["evaluation"] = kwargs
        return evaluation

    return {
        "validate_question": lambda question: safe,
        "classify_question_semantically": lambda **kwargs: safe,
        "resolve_authorized_api_key": lambda **kwargs: "test-key",
        "retrieve_relevant_chunks": hybrid_retrieve,
        "retrieve_pages": page_retrieve,
        "get_active_document": lambda user_id: {"active_doc_id": "doc-1", "active_filename": "source.pdf"},
        "generate_grounded_answer": answer,
        "evaluate_rag_response": evaluate,
        "save_evaluation": lambda *args, **kwargs: None,
    }


def test_greeting_bypasses_all_retrieval():
    captured = {}
    result = run_document_query(
        QueryRequest(question="Hello", response_language="kn"),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert result["question_type"] == "GREETING"
    assert result["answer"]
    assert "hybrid" not in captured and "pages" not in captured


def test_general_conversation_with_pdf_never_invokes_judge_or_retrieval():
    captured = {}
    result = run_document_query(
        QueryRequest(question="Hey, hello, can you hear me?", response_language="en"),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert result["question_type"] == "GENERAL_CONVERSATION"
    assert result["confidence"] is None
    assert result["evaluation"] is None
    assert "hybrid" not in captured and "pages" not in captured and "evaluation" not in captured


def test_general_information_with_pdf_is_refused_without_retrieval():
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["classify_intent"] = lambda *_args, **_kwargs: IntentClassification(
        IntentType.GENERAL_INFORMATION, 0.99, (), False, "NONE", False,
        response_action="RESPOND_CONVERSATIONALLY", response_language="hi",
    )
    result = run_document_query(
        QueryRequest(question="What is today's weather?", response_language="hi"),
        SimpleNamespace(id=7), endpoint="/api/voice", dependencies=dependencies,
    )
    assert result["status"] == "out_of_scope"
    assert result["question_type"] == "OUT_OF_SCOPE"
    assert result["answer"] is None
    assert "hybrid" not in captured and "pages" not in captured


def test_application_help_with_pdf_bypasses_retrieval():
    captured = {}
    result = run_document_query(
        QueryRequest(question="How do I change the language?", response_language="en"),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert result["question_type"] == "APPLICATION_HELP"
    assert result["answer"]
    assert "hybrid" not in captured and "pages" not in captured


def test_follow_up_uses_prior_evidence_pages_for_exact_next_page():
    captured = {}
    result = run_document_query(
        QueryRequest(
            question="What about the next page?",
            response_language="en",
            conversation_context=(
                "User: What is on page 5?\n"
                "Assistant: The topic is introduced here.\n"
                "Assistant evidence pages: 5"
            ),
        ),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert captured["pages"] == ([6], 7)
    assert "hybrid" not in captured
    assert result["question_type"] == "FOLLOW_UP_QUERY"


@pytest.mark.parametrize("language", ["en", "hi", "kn", "ta", "te", "ml", "mr", "gu", "bn", "pa", "ur"])
def test_greeting_preserves_selected_response_language(language):
    result = run_document_query(
        QueryRequest(question="Hi", response_language=language),
        SimpleNamespace(id=7), dependencies=_dependencies({}),
    )
    assert result["status"] == "success"
    assert result["answer"]


def test_exact_page_query_uses_page_retrieval_only_and_preserves_evidence():
    captured = {}
    result = run_document_query(
        QueryRequest(question="What is on pages 5 and 2?", response_language="kn"),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert captured["pages"] == ([5, 2], 7)
    assert "hybrid" not in captured
    assert result["question_type"] == "PAGE_SPECIFIC_QUERY"
    assert result["evaluation"] is not None
    assert [item["page_number"] for item in captured["evaluation"]["evidence"]] == [5, 2]
    assert all(chunk["language"] == "ta" for chunk in captured["answer"]["retrieved_chunks"])
    assert captured["answer"]["response_language"] == "kn"


def test_page_retrieval_failure_is_truthful_and_skips_answer_generation():
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["retrieve_pages"] = lambda page_numbers, user_id: {
        "status": "error", "active_document": "source.pdf", "retrieved_chunks": [],
        "evidence": [], "message": "Requested PDF pages are out of range.",
    }
    result = run_document_query(
        QueryRequest(question="What is on page 999?"),
        SimpleNamespace(id=7), dependencies=dependencies,
    )
    assert result["status"] == "error"
    assert result["message"] == "Requested PDF pages are out of range."
    assert "answer" not in captured and "evaluation" not in captured


def test_voice_page_range_error_preserves_physical_count_and_selected_language():
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["retrieve_pages"] = lambda page_numbers, user_id: {
        "status": "error", "active_document": "source.pdf", "retrieved_chunks": [],
        "evidence": [], "message": "out of range", "total_pages": 50,
        "invalid_pages": [52],
    }
    result = run_document_query(
        QueryRequest(question="What is on page 52?", response_language="hi"),
        SimpleNamespace(id=7), endpoint="/api/voice", dependencies=dependencies,
    )
    assert result["status"] == "error"
    assert "50" in result["message"] and "52" in result["message"]
    assert "पृष्ठ" in result["message"]
    assert "answer" not in captured


def test_normal_document_query_keeps_existing_hybrid_pipeline():
    captured = {}
    result = run_document_query(
        QueryRequest(question="What does the document say?"),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert result["status"] == "success"
    assert captured["hybrid"] == [("What does the document say?", 7)]
    assert "pages" not in captured


def test_standalone_document_query_does_not_embed_conversation_history():
    captured = {}
    result = run_document_query(
        QueryRequest(
            question="What is the publication date?",
            conversation_context="User: Tell me about another section.\nAssistant: unrelated prior answer",
            response_language="en",
        ),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert result["status"] == "success"
    assert captured["hybrid"] == [("What is the publication date?", 7)]


def test_cross_lingual_translation_copy_targets_document_language(monkeypatch):
    class FakeLLM:
        def invoke(self, _prompt):
            return SimpleNamespace(content="മലയാളം പരിഭാഷ")

    captured = {}
    monkeypatch.setattr(rag_orchestration_service, "get_llm", lambda _api_key: FakeLLM())
    original = "یہ دستاویز کس بارے میں ہے؟"
    translated = rag_orchestration_service._translate_query_for_retrieval(original, "ml", "test-key")

    assert translated == "മലയാളം പരിഭാഷ"
    assert original != translated


def test_validated_suggestion_miss_triggers_normal_translation_fallback(monkeypatch):
    captured = {}
    dependencies = _dependencies(captured)
    base_retrieve = dependencies["retrieve_relevant_chunks"]
    calls = []

    def retrieve(query, user_id):
        calls.append(query)
        result = base_retrieve(query, user_id)
        if len(calls) == 1:
            result = {**result, "retrieved_chunks": [{**result["retrieved_chunks"][0], "chunk_id": 99}]}
        return result

    dependencies["retrieve_relevant_chunks"] = retrieve
    dependencies["get_active_document"] = lambda _user_id: {
        "active_doc_id": "ml-doc", "active_filename": "source.pdf", "evidence_language": "ml",
    }
    dependencies["classify_intent"] = lambda *_args, **_kwargs: IntentClassification(
        IntentType.DOCUMENT_QUERY, 1.0, (), True, "RUN_DOCUMENT_QUERY",
    )
    monkeypatch.setattr(rag_orchestration_service, "_translate_query_for_retrieval", lambda *_args: "മലയാളം ചോദ്യം")
    result = run_document_query(
        QueryRequest(
            question="What is this document about?", response_language="en",
            suggestion_provenance={
                "question": "What is this document about?", "document_id": "ml-doc",
                "answerability": "validated", "supporting_chunk_ids": [10],
            },
        ),
        SimpleNamespace(id=7), dependencies=dependencies,
    )
    assert result["status"] == "success"
    assert calls == ["What is this document about?", "മലയാളം ചോദ്യം"]
    assert result["retrieval_query_language"] == "ml"


def test_validated_cross_language_suggestion_always_adds_translation_variant(monkeypatch):
    captured = {}
    dependencies = _dependencies(captured)
    calls = []

    def retrieve(query, user_id, query_variants=None):
        calls.append((query, query_variants or []))
        return {
            "status": "success", "active_document": "source.pdf", "confidence": "high",
            "best_score": 0.8, "average_score": 0.8,
            "retrieved_chunks": [{
                "chunk_id": 10, "file_name": "source.pdf", "page_number": 2,
                "text": "മലയാളം രേഖയിലെ ഉത്തരം", "score": 0.8,
            }],
        }

    dependencies["retrieve_relevant_chunks"] = retrieve
    dependencies["get_active_document"] = lambda _user_id: {
        "active_doc_id": "ml-doc", "active_filename": "source.pdf", "evidence_language": "ml",
    }
    dependencies["classify_intent"] = lambda *_args, **_kwargs: IntentClassification(
        IntentType.DOCUMENT_QUERY, 1.0, (), True, "RUN_DOCUMENT_QUERY",
    )
    monkeypatch.setattr(rag_orchestration_service, "_translate_query_for_retrieval", lambda *_args: "മലയാളം ചോദ്യം")

    result = run_document_query(
        QueryRequest(
            question="What is the answer?", response_language="en",
            suggestion_provenance={
                "document_id": "ml-doc", "answerability": "validated",
                "supporting_chunk_ids": [10],
            },
        ),
        SimpleNamespace(id=7), dependencies=dependencies,
    )

    assert result["status"] == "success"
    assert calls == [
        ("What is the answer?", []),
        ("What is the answer?", ["മലയാളം ചോദ്യം"]),
    ]
    assert result["retrieval_query_language"] == "ml"


@pytest.mark.parametrize(
    ("question", "response_language"),
    [
        ("What does the document say?", "kn"),
        ("ದಾಖಲೆ ಏನು ಹೇಳುತ್ತದೆ?", "hi"),
        ("यह दस्तावेज़ क्या बताता है?", "ta"),
        ("இந்த ஆவணம் என்ன சொல்கிறது?", "te"),
        ("ఈ పత్రం ఏమి చెబుతోంది?", "ml"),
        ("ഈ രേഖ എന്താണ് പറയുന്നത്?", "mr"),
        ("हा दस्तऐवज काय सांगतो?", "gu"),
        ("આ દસ્તાવેજ શું કહે છે?", "bn"),
        ("এই নথিতে কী বলা হয়েছে?", "pa"),
        ("ਇਸ ਦਸਤਾਵੇਜ਼ ਵਿੱਚ ਕੀ ਕਿਹਾ ਗਿਆ ਹੈ?", "ur"),
        ("اس دستاویز میں کیا کہا گیا ہے؟", "en"),
    ],
)
def test_selected_language_reaches_normal_text_answer_for_all_languages(question, response_language):
    captured = {}
    result = run_document_query(
        QueryRequest(question=question, response_language=response_language),
        SimpleNamespace(id=7), dependencies=_dependencies(captured),
    )
    assert result["status"] == "success"
    assert captured["answer"]["response_language"] == response_language


def test_persisted_text_selector_is_used_when_request_omits_language():
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["get_active_document"] = lambda _user_id: {
        "active_doc_id": "doc-1", "active_filename": "source.pdf", "response_language": "ta",
    }
    result = run_document_query(
        QueryRequest(question="What does the document say?"),
        SimpleNamespace(id=7), dependencies=dependencies,
    )
    assert result["status"] == "success"
    assert captured["answer"]["response_language"] == "ta"


def test_text_wrong_language_answer_fails_closed_in_selected_language():
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["generate_grounded_answer"] = lambda **_kwargs: "This answer is English only."
    result = run_document_query(
        QueryRequest(question="What does the document say?", response_language="ta"),
        SimpleNamespace(id=7), dependencies=dependencies,
    )
    assert result["status"] == "error"
    assert result["answer"] is None
    assert result["message"]


def test_voice_answer_language_validation_is_not_changed_by_text_policy():
    dependencies = _dependencies({})
    dependencies["generate_grounded_answer"] = lambda **_kwargs: "This answer is English only."
    result = run_document_query(
        QueryRequest(question="What does the document say?", response_language="ta"),
        SimpleNamespace(id=7), endpoint="/api/voice", dependencies=dependencies,
    )
    assert result["status"] == "success"
    assert result["answer"] == "This answer is English only."


def test_out_of_scope_and_ambiguous_follow_up_do_not_retrieve():
    for question in ("What is the weather today?", "Tell me more."):
        captured = {}
        result = run_document_query(
            QueryRequest(question=question),
            SimpleNamespace(id=7), dependencies=_dependencies(captured),
        )
        assert "hybrid" not in captured and "pages" not in captured
        assert result["status"] in {"out_of_scope", "clarification_required"}


@pytest.mark.parametrize(
    ("question", "document_language"),
    [
        ("Covid संकट के दौरान डॉक्टरों की भूमिका क्या है?", "kn"),
        ("ಕೋವಿಡ್ ಸಂಕಷ್ಟದ ಸಮಯದಲ್ಲಿ ವೈದ್ಯರ ಪಾತ್ರವೇನು?", "kn"),
        ("Covid संकट के दौरान डॉक्टरों की भूमिका क्या है?", "en"),
    ],
)
def test_clear_multilingual_document_question_overrides_semantic_clarification(question, document_language):
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["get_active_document"] = lambda _user_id: {
        "active_doc_id": "doc-multilingual", "active_filename": "source.pdf",
        "evidence_language": document_language,
    }
    dependencies["classify_intent_semantically"] = lambda **_kwargs: IntentClassification(
        IntentType.DOCUMENT_CLARIFICATION, 0.91, (), False, "NONE", False,
        response_action="REQUEST_CLARIFICATION", response_language="hi",
    )

    result = run_document_query(
        QueryRequest(
            question=question, response_language="hi", use_semantic_intent=True,
            conversation_id="voice-session-1",
        ),
        SimpleNamespace(id=7), endpoint="/api/voice", dependencies=dependencies,
    )

    assert result["status"] == "success"
    assert result["response_language"] == "hi"
    assert result["document_id"] == "doc-1"
    assert captured["hybrid"] == [(question, 7)]
    assert result["evidence"][0]["page_number"] == 2


def test_valid_document_suggestion_is_not_clarified_by_semantic_router():
    captured = {}
    dependencies = _dependencies(captured)
    dependencies["get_active_document"] = lambda _user_id: {
        "active_doc_id": "kn-doc", "active_filename": "source.pdf", "evidence_language": "kn",
    }
    dependencies["classify_intent_semantically"] = lambda **_kwargs: IntentClassification(
        IntentType.AMBIGUOUS_QUERY, 0.88, (), False, "NONE", False,
        response_action="REQUEST_CLARIFICATION", response_language="hi",
    )
    result = run_document_query(
        QueryRequest(
            question="Covid ಸಂಕಷ್ಟದ ಸಮಯದಲ್ಲಿ ವೈದ್ಯರ ಪಾತ್ರವೇನು?", response_language="hi",
            use_semantic_intent=True,
            suggestion_provenance={
                "document_id": "kn-doc", "answerability": "validated",
                "supporting_chunk_ids": [10],
            },
        ),
        SimpleNamespace(id=7), endpoint="/api/voice", dependencies=dependencies,
    )
    assert result["status"] == "success"
    assert captured["hybrid"]


def test_text_and_voice_share_page_routing():
    for endpoint in ("/api/ask", "/api/voice"):
        captured = {}
        result = run_document_query(
            QueryRequest(question="What is on page 4?", response_language="ta"),
            SimpleNamespace(id=7), endpoint=endpoint, dependencies=_dependencies(captured),
        )
        assert result["question_type"] == "PAGE_SPECIFIC_QUERY"
        assert captured["pages"] == ([4], 7)
        assert "hybrid" not in captured
