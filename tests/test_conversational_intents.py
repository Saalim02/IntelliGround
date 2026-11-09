from app.services.voice_intent_service import IntentType, classify_intent


def test_conversational_controls_do_not_require_rag_or_evaluation():
    cases = {
        "Hello": IntentType.GREETING,
        "How are you?": IntentType.CASUAL_CONVERSATION,
        "Can you hear me?": IntentType.AUDIO_CHECK,
        "Thank you": IntentType.ACKNOWLEDGMENT,
        "Continue speaking": IntentType.CONTINUE_RESPONSE,
        "Explain that again": IntentType.REPEAT_RESPONSE,
    }
    for text, expected in cases.items():
        result = classify_intent(text)
        assert result.intent is expected
        assert result.requires_rag is False
        assert result.requires_evaluation is False


def test_document_and_page_intents_are_evaluation_eligible():
    overview = classify_intent("What's in this PDF?")
    page = classify_intent("Explain page 40")
    assert overview.intent is IntentType.DOCUMENT_OVERVIEW
    assert overview.requires_rag is True
    assert overview.requires_evaluation is True
    assert page.intent is IntentType.PAGE_QUERY
    assert page.referenced_page == 40
    assert page.page_numbers == (40,)


def test_contextual_follow_up_and_ambiguous_question_are_distinct():
    assert classify_intent("What does that mean?").intent is IntentType.AMBIGUOUS_QUERY
    result = classify_intent(
        "What does that mean?",
        conversation_context="User: Explain page 40.\nAssistant: The section defines eligibility.",
    )
    assert result.intent is IntentType.DOCUMENT_FOLLOWUP
    assert result.requires_rag is True


def test_multilingual_intents_route_without_translating_evidence():
    assert classify_intent("ನಮಸ್ಕಾರ").intent is IntentType.GREETING
    assert classify_intent("இந்த ஆவணம் எதைப் பற்றி?").intent is IntentType.DOCUMENT_OVERVIEW
    assert classify_intent("కొనసాగించండి").intent is IntentType.CONTINUE_RESPONSE


def test_page_marker_extracts_requested_page_without_fabricating_content():
    result = classify_intent("What is written on page 999?")
    assert result.intent is IntentType.PAGE_QUERY
    assert result.page_numbers == (999,)
    assert result.as_dict()["referenced_page"] == 999
