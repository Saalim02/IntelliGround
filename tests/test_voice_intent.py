import pytest

from app.services.voice_intent_service import classify_voice_intent
from app.services.voice_response_templates import localized_voice_message


@pytest.mark.parametrize("language_text", [
    "Hello", "नमस्ते", "ನಮಸ್ಕಾರ", "வணக்கம்", "నమస్కారం",
    "നമസ്കാരം", "नमस्कार", "નમસ્તે", "নমস্কার", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ",
    "السلام علیکم",
])
def test_greetings_route_without_document_retrieval(language_text):
    assert classify_voice_intent(language_text) == "greeting"


@pytest.mark.parametrize("language", ["en", "kn", "hi", "ta", "te", "ml", "mr", "gu", "bn", "pa", "ur"])
def test_all_language_templates_are_present(language):
    for kind in ("greeting", "small_talk", "out_of_scope", "not_found", "human_review", "safe_refusal"):
        assert localized_voice_message(language, kind)


def test_unknown_factual_voice_request_remains_document_bound():
    assert classify_voice_intent("What does the uploaded document say about eligibility?") == "document"


def test_clear_unrelated_voice_request_is_out_of_scope():
    assert classify_voice_intent("What is the weather today?") == "out_of_scope"
