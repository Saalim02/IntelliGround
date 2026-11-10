"""Shared document-question orchestration for text and voice interfaces."""

import inspect
import os
from uuid import uuid4

from app.models.auth_models import User
from app.models.evaluation_models import EvaluationDecision
from app.models.query_models import QueryRequest
from app.services.answer_service import generate_grounded_answer
from app.services.document_registry_service import get_active_document
from app.services.evaluation.evaluation_service import evaluate_rag_response
from app.services.evaluation.evaluation_storage_service import save_evaluation
from app.services.language_detection_service import detect_text_language
from app.services.language_registry import (
    is_language_supported_text,
    localized_text_message,
    validate_language,
)
from app.services.page_retrieval_service import retrieve_pages
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.security.audit_logger import log_security_event
from app.services.security.guardrails import validate_question
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.security.semantic_guardrail import classify_question_semantically
from app.services.voice_intent_service import (
    classify_intent,
    classify_intent_semantically,
    classify_voice_intent,
    generate_conversational_response,
)
from app.services.conversation_resolution_service import resolve_follow_up
from app.services.voice_response_templates import localized_voice_message
from app.services.performance_service import measure, performance_context
from app.services.observability.langsmith_service import observed


def classify_question(confidence: str) -> str:
    return "UNANSWERABLE" if confidence == "low" else "DOCUMENT"


def _retrieval_diagnostics(retrieval_output: dict) -> dict:
    return {
        key: retrieval_output.get(key)
        for key in (
            "dense_candidate_count",
            "sparse_candidate_count",
            "fused_candidate_count",
            "reranked_candidate_count",
            "retrieval_stage_timings",
            "reranker_enabled",
            "reranker_status",
        )
    }


def _unanswerable_response(
    retrieval_output: dict,
    question: str,
    *,
    confidence: str | None = None,
    response_language: str = "en",
    message: str | None = None,
) -> dict:
    return {
        "status": "error",
        "active_document": retrieval_output["active_document"],
        "question": question,
        "question_type": "UNANSWERABLE",
        "answer": None,
        "confidence": confidence or retrieval_output["confidence"],
        "best_score": retrieval_output["best_score"],
        "average_score": retrieval_output["average_score"],
        "evidence": [],
        "retrieved_chunks": [],
        "message": message or localized_voice_message(response_language, "not_found"),
        **_retrieval_diagnostics(retrieval_output),
    }


def _page_range_message(response_language: str, total_pages: int, invalid_pages: list[int]) -> str:
    invalid = ", ".join(map(str, invalid_pages))
    templates = {
        "hi": f"इस PDF में केवल {total_pages} पृष्ठ हैं। पृष्ठ {invalid} वर्तमान दस्तावेज़ में मौजूद नहीं हैं।",
        "kn": f"ಈ PDF ನಲ್ಲಿ ಕೇವಲ {total_pages} ಪುಟಗಳಿವೆ. ಪುಟ {invalid} ಪ್ರಸ್ತುತ ದಾಖಲೆಯಲ್ಲಿ ಇಲ್ಲ.",
        "ta": f"இந்த PDF-ல் {total_pages} பக்கங்கள் மட்டுமே உள்ளன. பக்கம் {invalid} தற்போதைய ஆவணத்தில் இல்லை.",
        "te": f"ఈ PDFలో {total_pages} పేజీలు మాత్రమే ఉన్నాయి. పేజీ {invalid} ప్రస్తుత పత్రంలో లేదు.",
        "mr": f"या PDF मध्ये फक्त {total_pages} पाने आहेत. पृष्ठ {invalid} सध्याच्या दस्तऐवजात नाही.",
    }
    return templates.get(
        response_language,
        f"This PDF contains only {total_pages} pages. Page(s) {invalid} are not present in the current document.",
    )


_CLARIFICATION_MESSAGES = {
    "en": "Could you clarify which topic, page, or part of the PDF you mean?",
    "hi": "कृपया स्पष्ट करें कि आप PDF के किस विषय, पृष्ठ या भाग की बात कर रहे हैं।",
    "kn": "ನೀವು PDF ನ ಯಾವ ವಿಷಯ, ಪುಟ ಅಥವಾ ಭಾಗವನ್ನು ಉಲ್ಲೇಖಿಸುತ್ತಿದ್ದೀರಿ ಎಂದು ಸ್ಪಷ್ಟಪಡಿಸಬಹುದೇ?",
    "ta": "PDF-இன் எந்தத் தலைப்பு, பக்கம் அல்லது பகுதியை நீங்கள் குறிப்பிடுகிறீர்கள் என்பதைத் தெளிவுபடுத்த முடியுமா?",
    "te": "మీరు PDFలో ఏ అంశం, పేజీ లేదా భాగాన్ని ఉద్దేశిస్తున్నారో స్పష్టీకరించగలరా?",
    "ml": "PDF-ലെ ഏത് വിഷയം, പേജ് അല്ലെങ്കിൽ ഭാഗമാണ് ഉദ്ദേശിക്കുന്നതെന്ന് വ്യക്തമാക്കാമോ?",
    "mr": "कृपया PDF मधील कोणता विषय, पृष्ठ किंवा भाग अभिप्रेत आहे ते स्पष्ट करा.",
    "gu": "તમે PDFના કયા વિષય, પાનું અથવા ભાગનો ઉલ્લેખ કરો છો તે સ્પષ્ટ કરશો?",
    "bn": "আপনি PDF-এর কোন বিষয়, পৃষ্ঠা বা অংশ বোঝাচ্ছেন তা কি স্পষ্ট করবেন?",
    "pa": "ਕਿਰਪਾ ਕਰਕੇ ਸਪਸ਼ਟ ਕਰੋ ਕਿ ਤੁਸੀਂ PDF ਦੇ ਕਿਹੜੇ ਵਿਸ਼ੇ, ਪੰਨੇ ਜਾਂ ਹਿੱਸੇ ਦੀ ਗੱਲ ਕਰ ਰਹੇ ਹੋ।",
    "ur": "براہِ کرم واضح کریں کہ آپ PDF کے کس موضوع، صفحے یا حصے کی بات کر رہے ہیں۔",
}


def _clarification_message(language: str) -> str:
    return _CLARIFICATION_MESSAGES.get(language, _CLARIFICATION_MESSAGES["en"])


def _response_question_type(intent_name: str) -> str:
    # Keep the established API labels stable while the classifier exposes the
    # richer intent vocabulary internally.
    return {
        "PAGE_QUERY": "PAGE_SPECIFIC_QUERY",
        "DOCUMENT_FOLLOWUP": "FOLLOW_UP_QUERY",
    }.get(intent_name, intent_name)


def _needs_semantic_intent(intent) -> bool:
    """Use the LLM only for ambiguous or low-confidence local routes."""
    return intent.intent.value in {"DOCUMENT_QUERY", "AMBIGUOUS_QUERY"} or intent.confidence < 0.80


def _last_assistant_answer(context: str | None) -> str | None:
    """Read only the last persisted answer for voice continuation/repetition."""
    lines = [line.strip() for line in (context or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("Assistant:"):
            answer = line.removeprefix("Assistant:").strip()
            if answer:
                return answer
    return None


def _no_document_voice_response(request: QueryRequest, intent=None, *, conversational_answer: str | None = None) -> dict:
    """Handle voice conversation before any document retrieval is possible."""
    if intent is not None and not intent.requires_retrieval:
        return _intent_response(request, intent, endpoint="/api/voice", conversational_answer=conversational_answer)
    kind = classify_voice_intent(request.question)
    intent_name = getattr(getattr(intent, "intent", None), "value", None)
    if intent_name == "GREETING" or kind == "greeting":
        answer = localized_voice_message(request.response_language, "greeting")
        status, question_type, confidence = "success", "CONVERSATIONAL", None
    elif intent_name == "GENERAL_CONVERSATION" or kind == "small_talk":
        answer = localized_voice_message(request.response_language, "no_document_small_talk")
        status, question_type, confidence = "success", "GENERAL_CONVERSATION", None
    elif intent_name == "APPLICATION_HELP" or kind == "application_help":
        answer = localized_voice_message(request.response_language, "application_help")
        status, question_type, confidence = "success", "APPLICATION_HELP", None
    else:
        answer = None
        status, question_type, confidence = "document_required", "DOCUMENT_QUERY", None
    return {
        "status": status,
        "active_document": None,
        "question": request.question,
        "question_type": question_type,
        "answer": answer,
        "confidence": confidence,
        "best_score": None,
        "average_score": None,
        "evidence": [],
        "retrieved_chunks": [],
        "evaluation": None,
        "message": answer or localized_voice_message(request.response_language, "document_required"),
    }


def _intent_response(request: QueryRequest, intent, *, endpoint: str, conversational_answer: str | None = None) -> dict:
    """Return a contract-compatible response for non-retrieval intents."""
    intent_name = intent.intent.value
    if intent_name == "GREETING":
        voice_kind = classify_voice_intent(request.question)
        response_kind = voice_kind if voice_kind in {"greeting", "small_talk"} else "greeting"
        return {
            "status": "success",
            "active_document": None,
            "question": request.question,
            "question_type": "CONVERSATIONAL" if endpoint == "/api/voice" else intent.intent.value,
            "answer": conversational_answer or localized_voice_message(request.response_language, response_kind),
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": (
                "Conversational response generated."
                if endpoint == "/api/voice"
                else localized_text_message(request.response_language, "conversation")
            ),
        }
    if intent_name in {"GENERAL_CONVERSATION", "CASUAL_CONVERSATION"}:
        return {
            "status": "success",
            "active_document": None,
            "question": request.question,
            "question_type": "GENERAL_CONVERSATION",
            "answer": conversational_answer or localized_voice_message(request.response_language, "small_talk"),
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": "Conversational response generated.",
        }
    if intent_name == "AUDIO_CHECK":
        return {
            "status": "success", "active_document": None, "question": request.question,
            "question_type": intent_name, "answer": localized_voice_message(request.response_language, "audio_check"),
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": "Audio check response generated.",
        }
    if intent_name in {"ACKNOWLEDGMENT", "CONTINUE_RESPONSE", "REPEAT_RESPONSE", "WAIT_FOR_USER", "READY_TO_CONTINUE"}:
        previous = _last_assistant_answer(request.conversation_context)
        if intent_name == "REPEAT_RESPONSE" and previous:
            answer = previous
        elif intent_name == "CONTINUE_RESPONSE" and previous:
            answer = localized_voice_message(request.response_language, "continue_response")
        elif intent_name == "WAIT_FOR_USER":
            answer = localized_voice_message(request.response_language, "wait_for_user")
        elif intent_name == "READY_TO_CONTINUE":
            answer = localized_voice_message(request.response_language, "ready_to_continue")
        else:
            answer = localized_voice_message(request.response_language, "acknowledgment")
        return {
            "status": "success", "active_document": None, "question": request.question,
            "question_type": intent_name, "answer": answer,
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": "Conversational response generated.",
        }
    if intent_name in {"CLARIFY_RESPONSE", "AMBIGUOUS_QUERY"}:
        return {
            "status": "clarification_required", "active_document": None, "question": request.question,
            "question_type": intent_name, "answer": None,
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": _clarification_message(request.response_language),
        }
    if intent.intent.value == "OUT_OF_SCOPE":
        return {
            "status": "out_of_scope",
            "active_document": None,
            "question": request.question,
            "question_type": intent.intent.value,
            "answer": None,
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": localized_voice_message(request.response_language, "out_of_scope"),
        }
    if intent.intent.value == "APPLICATION_HELP":
        return {
            "status": "success",
            "active_document": None,
            "question": request.question,
            "question_type": intent.intent.value,
            "answer": localized_voice_message(request.response_language, "application_help"),
            "confidence": None, "best_score": None, "average_score": None,
            "evidence": [], "retrieved_chunks": [], "evaluation": None,
            "message": localized_voice_message(request.response_language, "application_help"),
        }
    return {
        "status": "clarification_required",
        "active_document": None,
        "question": request.question,
        "question_type": intent.intent.value,
        "answer": None,
        "confidence": None, "best_score": None, "average_score": None,
        "evidence": [], "retrieved_chunks": [], "evaluation": None,
        "message": _clarification_message(request.response_language),
    }


def _direct_page_retrieval_output(page_output: dict) -> dict:
    """Adapt exact-page results to the existing evaluated QA shape."""
    chunks = page_output.get("retrieved_chunks", [])
    return {
        "status": page_output.get("status", "error"),
        "active_document": page_output.get("active_document"),
        "confidence": "high" if chunks else "low",
        "best_score": 1.0 if chunks else None,
        "average_score": 1.0 if chunks else None,
        "retrieved_chunks": chunks,
        "evidence": page_output.get("evidence", []),
        "message": page_output.get("message", "Page retrieval failed."),
        "total_pages": page_output.get("total_pages"),
        "invalid_pages": page_output.get("invalid_pages", []),
        "dense_candidate_count": 0,
        "sparse_candidate_count": 0,
        "fused_candidate_count": len(chunks),
        "reranked_candidate_count": len(chunks),
        "retrieval_stage_timings": {},
        "reranker_enabled": False,
        "reranker_status": "not_run",
    }


@observed("qa.request")
def run_document_query(
    request: QueryRequest,
    current_user: User,
    *,
    endpoint: str = "/api/ask",
    dependencies: dict | None = None,
) -> dict:
    """Run the existing guarded, evaluated document QA pipeline.

    Routes for text and voice should call this function rather than creating
    separate retrieval or answer-generation implementations.
    """
    dependencies = dependencies or {}
    validate = dependencies.get("validate_question", validate_question)
    semantic_classify = dependencies.get(
        "classify_question_semantically",
        classify_question_semantically,
    )
    resolve_api_key = dependencies.get(
        "resolve_authorized_api_key",
        resolve_authorized_api_key,
    )
    retrieve = dependencies.get("retrieve_relevant_chunks", retrieve_relevant_chunks)
    retrieve_pages_for_user = dependencies.get("retrieve_pages", retrieve_pages)
    classify = dependencies.get("classify_intent", classify_intent)
    get_active = dependencies.get("get_active_document", get_active_document)
    generate_answer = dependencies.get("generate_grounded_answer", generate_grounded_answer)
    evaluate = dependencies.get("evaluate_rag_response", evaluate_rag_response)
    persist_evaluation = dependencies.get("save_evaluation", save_evaluation)

    from fastapi import HTTPException

    # Text clients may omit the field when the selector is persisted on the
    # active document.  Resolve that persisted selection before validation so
    # the provider never silently falls back to English.  Voice keeps its
    # session/request language behavior unchanged.
    active_for_language = get_active(current_user.id) if endpoint != "/api/voice" else None
    requested_response_language = request.response_language or (
        (active_for_language or {}).get("response_language") if active_for_language else None
    ) or "en"
    try:
        request.response_language = validate_language(requested_response_language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Input language is detected independently for diagnostics and downstream
    # routing. It never overrides the explicitly selected response language.
    with measure("qa.language_detection"):
        request.query_language = detect_text_language(request.question)["language"]

    security_result = validate(request.question)
    if not security_result.is_safe:
        log_security_event(
            "input_guardrail_blocked",
            endpoint=endpoint,
            client_id=str(current_user.id),
            decision=security_result.decision.value,
            risk_score=security_result.risk_score,
            reasons=security_result.reasons,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Request blocked by security guardrail.",
                "question_type": "SECURITY_BLOCKED",
                "security": security_result.model_dump(),
            },
        )

    classify_parameters = inspect.signature(classify).parameters
    classify_kwargs = {}
    if "conversation_context" in classify_parameters:
        classify_kwargs["conversation_context"] = request.conversation_context
    if "response_language" in classify_parameters:
        classify_kwargs["response_language"] = request.response_language
    intent = classify(request.question, **classify_kwargs)
    semantic_intent_enabled = bool(request.use_semantic_intent and endpoint == "/api/voice")

    # Classify before checking document availability. Voice conversation and
    # application help do not need an API key, semantic RAG guardrail, or PDF.
    if endpoint == "/api/voice":
        active_document = get_active(current_user.id)
        if not active_document:
            if semantic_intent_enabled and intent.requires_retrieval and _needs_semantic_intent(intent):
                semantic = classify_intent_semantically(
                    request.question,
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    conversation_context=request.conversation_context,
                    response_language=request.response_language,
                )
                if semantic is not None:
                    intent = semantic
            conversational_answer = None
            if intent.intent.value in {"CASUAL_CONVERSATION", "GENERAL_CONVERSATION"}:
                conversational_answer = generate_conversational_response(
                    request.question,
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    conversation_context=request.conversation_context,
                    response_language=request.response_language,
                )
            return _no_document_voice_response(request, intent, conversational_answer=conversational_answer)
        if not intent.requires_retrieval:
            return _intent_response(request, intent, endpoint=endpoint)

    api_key = resolve_api_key(
        user=current_user,
        access_password=request.access_password,
        user_openai_api_key=request.user_openai_api_key,
    )

    if semantic_intent_enabled and intent.requires_retrieval and _needs_semantic_intent(intent):
        semantic = classify_intent_semantically(
            request.question,
            api_key=api_key,
            conversation_context=request.conversation_context,
            active_document_id=(get_active(current_user.id) or {}).get("active_doc_id"),
            response_language=request.response_language,
        )
        if semantic is not None:
            intent = semantic

    semantic_security_result = semantic_classify(
        question=request.question,
        api_key=api_key,
    )
    if not semantic_security_result.is_safe:
        log_security_event(
            "semantic_input_guardrail_blocked",
            endpoint=endpoint,
            client_id=str(current_user.id),
            decision=semantic_security_result.decision.value,
            risk_score=semantic_security_result.risk_score,
            reasons=semantic_security_result.reasons,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Request blocked by semantic security guardrail.",
                "question_type": "SECURITY_BLOCKED",
                "security": semantic_security_result.model_dump(),
            },
        )

    # A semantic decision may identify a conversational utterance that the
    # local fallback could not recognize. It bypasses RAG/evaluation only
    # after the existing semantic security guardrail has allowed it.
    if endpoint == "/api/voice" and not intent.requires_retrieval:
        conversational_answer = None
        if intent.intent.value in {"CASUAL_CONVERSATION", "GENERAL_CONVERSATION"} and semantic_intent_enabled:
            conversational_answer = generate_conversational_response(
                request.question,
                api_key=api_key,
                conversation_context=request.conversation_context,
                response_language=request.response_language,
            )
        return _intent_response(request, intent, endpoint=endpoint, conversational_answer=conversational_answer)

    if not intent.requires_retrieval:
        return _intent_response(request, intent, endpoint=endpoint)

    with measure("qa.query_preprocessing"):
        query_text = request.question
        resolved_pages: tuple[int, ...] = ()
        if intent.intent.value == "DOCUMENT_FOLLOWUP":
            resolved = resolve_follow_up(request.question, request.conversation_context)
            if resolved.ambiguous:
                return _intent_response(request, intent, endpoint=endpoint)
            query_text = resolved.query
            resolved_pages = resolved.page_numbers
        elif request.conversation_context:
            query_text = (
                "Conversation context:\n"
                f"{request.conversation_context}\n\n"
                f"Current user question:\n{request.question}"
            )

    active_doc = get_active(current_user.id)
    active_doc_id = active_doc["active_doc_id"] if active_doc else None
    timing_fields = {
        "request_id": request.turn_id or f"query-{uuid4().hex}",
        "user_id": current_user.id,
        "document_id": active_doc_id,
    }
    if intent.intent.value == "PAGE_QUERY" or resolved_pages:
        with performance_context(**timing_fields):
            with measure("qa.direct_page_retrieval"):
                page_output = retrieve_pages_for_user(
                    intent.page_numbers or resolved_pages,
                    user_id=current_user.id,
                )
        retrieval_output = _direct_page_retrieval_output(page_output)
    else:
        with performance_context(**timing_fields):
            with measure("qa.retrieval_pipeline"):
                retrieval_output = retrieve(query_text, user_id=current_user.id)
    if retrieval_output["status"] != "success":
        page_error_message = None
        if retrieval_output.get("invalid_pages") and retrieval_output.get("total_pages"):
            page_error_message = _page_range_message(
                request.response_language,
                retrieval_output["total_pages"],
                retrieval_output["invalid_pages"],
            )
        if endpoint == "/api/voice":
            return _unanswerable_response(
                retrieval_output, request.question,
                response_language=request.response_language,
                message=page_error_message,
            )
        if request.response_language != "en":
            retrieval_output["message"] = localized_voice_message(
                request.response_language, "not_found"
            )
        retrieval_output["question"] = request.question
        retrieval_output["question_type"] = intent.intent.value
        retrieval_output["answer"] = None
        retrieval_output["evidence"] = retrieval_output.get("evidence", [])
        retrieval_output["retrieved_chunks"] = retrieval_output.get("retrieved_chunks", [])
        retrieval_output["evaluation"] = None
        return retrieval_output

    if retrieval_output["confidence"] == "low":
        return _unanswerable_response(
            retrieval_output, request.question,
            response_language=request.response_language,
        )

    retrieved_chunks = retrieval_output["retrieved_chunks"]
    answer_kwargs = {
        "question": query_text,
        "retrieved_chunks": retrieved_chunks,
        "api_key": api_key,
    }
    # The selected text response language is always explicit at the answer
    # boundary, including follow-ups and requests with no supplied context.
    # Keep compatibility with older injected answer adapters that only accept
    # the original three arguments.
    answer_parameters = inspect.signature(generate_answer).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in answer_parameters.values()
    )
    if accepts_kwargs or "response_language" in answer_parameters:
        answer_kwargs["response_language"] = request.response_language
    if accepts_kwargs or "conversation_context" in answer_parameters:
        answer_kwargs["conversation_context"] = request.conversation_context
    with performance_context(**timing_fields):
        with measure("qa.answer_generation"):
            final_answer = generate_answer(**answer_kwargs)
    if not final_answer or final_answer.strip() == "Answer not found in uploaded PDF.":
        return _unanswerable_response(
            retrieval_output,
            request.question,
            confidence="low",
            response_language=request.response_language,
        )
    if endpoint != "/api/voice" and not is_language_supported_text(
        final_answer, request.response_language
    ):
        # Do not pass a grounded answer through merely to satisfy the response
        # contract.  The caller receives the existing truthful no-evidence
        # response in the selected language instead.
        return _unanswerable_response(
            retrieval_output,
            request.question,
            confidence="low",
            response_language=request.response_language,
        )

    with measure("qa.citation_construction"):
        evidence = retrieval_output.get("evidence", [])
        if not evidence:
            seen_pages = set()
            for chunk in retrieved_chunks:
                page_number = chunk["page_number"]
                if page_number in seen_pages:
                    continue
                seen_pages.add(page_number)
                evidence.append(
                    {
                        "page_number": page_number,
                        "snippet": chunk["text"][:250],
                        "image_path": (
                            f"/api/page-images/{active_doc_id}/page_{page_number}.png"
                        ),
                    }
                )

    with performance_context(**timing_fields):
        with measure("qa.grounding_and_citation_validation"):
            evaluation = evaluate(
                question=query_text, answer=final_answer,
                retrieved_chunks=retrieved_chunks, evidence=evidence,
                api_key=api_key, use_llm_judges=True,
            )
    evaluation.user_id = current_user.id
    persist_evaluation(evaluation, user_id=current_user.id)

    if evaluation.decision == EvaluationDecision.HUMAN_REVIEW:
        response_status = "human_review_required"
        response_answer = None
        response_question_type = "HUMAN_REVIEW_REQUIRED"
        response_message = localized_voice_message(request.response_language, "human_review") if endpoint == "/api/voice" else localized_text_message(request.response_language, "human_review")
    elif evaluation.decision == EvaluationDecision.REVIEW_RECOMMENDED:
        response_status = "review_recommended"
        response_answer = final_answer
        response_question_type = (
            _response_question_type(intent.intent.value)
            if intent.intent.value != "DOCUMENT_QUERY"
            else classify_question(retrieval_output["confidence"])
        )
        response_message = (
            "Answer generated, but additional human review is recommended."
            if endpoint == "/api/voice"
            else localized_text_message(request.response_language, "review")
        )
    elif evaluation.decision == EvaluationDecision.AUTO_APPROVE:
        response_status = "success"
        response_answer = final_answer
        response_question_type = (
            _response_question_type(intent.intent.value)
            if intent.intent.value != "DOCUMENT_QUERY"
            else classify_question(retrieval_output["confidence"])
        )
        response_message = (
            "Answer generated successfully."
            if endpoint == "/api/voice"
            else localized_text_message(request.response_language, "success")
        )
    else:
        response_status = "human_review_required"
        response_answer = None
        response_question_type = "HUMAN_REVIEW_REQUIRED"
        response_message = (
            "An unrecognized evaluation decision was returned. Human review is required."
            if endpoint == "/api/voice"
            else localized_text_message(request.response_language, "unknown")
        )

    return {
        "status": response_status,
        "active_document": retrieval_output["active_document"],
        "question": request.question,
        "question_type": response_question_type,
        "answer": response_answer,
        "confidence": retrieval_output["confidence"],
        "best_score": retrieval_output["best_score"],
        "average_score": retrieval_output["average_score"],
        "evidence": evidence,
        "retrieved_chunks": retrieved_chunks,
        "evaluation": evaluation.model_dump(),
        "message": response_message,
        **_retrieval_diagnostics(retrieval_output),
    }
