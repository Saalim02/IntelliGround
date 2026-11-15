import pytest

from app.models.voice_models import VoicePreferenceRequest, VoiceSessionCreateRequest
from app.services.emotion_state_service import choose_emotion
from app.services.gnani_timbre_service import _payload
from app.services.voice_intent_service import IntentType, classify_intent


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does this document say about financial provisions?", IntentType.DOCUMENT_QUERY),
        ("Who is the President of India?", IntentType.GENERAL_KNOWLEDGE),
        ("What's the weather today?", IntentType.CURRENT_INFORMATION),
        ("Hello", IntentType.GREETING),
    ],
)
def test_advanced_voice_intents_are_non_rag_when_out_of_scope(question, expected):
    result = classify_intent(question)
    assert result.intent is expected
    if expected is not IntentType.DOCUMENT_QUERY:
        assert result.requires_retrieval is False


@pytest.mark.parametrize("language_text", [
    "ನಮಸ್ಕಾರ",
    "नमस्ते",
    "வணக்கம்",
    "నమస్కారం",
    "നമസ്കാരം",
])
def test_multilingual_greetings_are_not_document_queries(language_text):
    result = classify_intent(language_text)
    assert result.intent is IntentType.GREETING
    assert result.requires_retrieval is False


def test_contextual_clarification_is_document_clarification():
    result = classify_intent("Can you clarify?", conversation_context="Assistant: The PDF discusses budgets.")
    assert result.intent is IntentType.DOCUMENT_CLARIFICATION
    assert result.requires_retrieval is False


@pytest.mark.parametrize("emotion", ["AUTO", "HAPPY", "SAD", "ANGRY", "CALM", "PROFESSIONAL", "FRIENDLY"])
def test_response_emotion_is_validated_and_manual_selection_wins(emotion):
    request = VoicePreferenceRequest(voice_id="Kaveri", response_emotion=emotion)
    session = VoiceSessionCreateRequest(response_emotion=emotion)
    assert request.response_emotion == emotion
    assert session.response_emotion == emotion


def test_manual_response_emotion_is_distinct_from_detected_user_emotion():
    state = choose_emotion(selected_emotion="PROFESSIONAL", intent="DOCUMENT_QUERY", evidence_count=2)
    assert state.emotion == "PROFESSIONAL"
    assert state.trigger == "user-selected response emotion"


def test_unsupported_tts_emotion_is_not_sent_to_provider_payload():
    payload = _payload("Grounded answer", "en", "wav", "Kaveri", "HAPPY")
    assert "emotion" not in payload
