from app.services.voice_intent_service import (
    IntentType,
    SemanticIntentDecision,
    classify_intent,
    classify_intent_semantically,
)


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


def test_wait_and_ready_are_conversation_actions_without_rag():
    waiting = classify_intent("Hey, give me a second.")
    ready = classify_intent("I'm ready now.")
    assert waiting.intent is IntentType.WAIT_FOR_USER
    assert waiting.requires_rag is False
    assert waiting.conversation_action == "WAIT"
    assert ready.intent is IntentType.READY_TO_CONTINUE
    assert ready.requires_evaluation is False
    assert ready.conversation_action == "RESUME"


def test_capability_questions_use_application_help_without_rag():
    result = classify_intent("What can you do?")
    assert result.intent is IntentType.APPLICATION_HELP
    assert result.requires_rag is False
    assert result.requires_evaluation is False


def test_semantic_intent_uses_structured_decision_without_answer_generation(monkeypatch):
    class FakeChain:
        def invoke(self, _payload):
            return SemanticIntentDecision(
                intent="CASUAL_CONVERSATION",
                intent_confidence=0.96,
                requires_document=False,
                requires_rag=False,
                requires_evaluation=False,
                conversation_action="RESPOND_CONVERSATIONALLY",
            )

    class FakePrompt:
        def __or__(self, _structured):
            return FakeChain()

    class FakeLlm:
        def with_structured_output(self, _schema):
            return self

    monkeypatch.setattr(
        "app.services.voice_intent_service.ChatPromptTemplate.from_template",
        lambda _template: FakePrompt(),
    )
    monkeypatch.setattr(
        "app.services.voice_intent_service.get_llm",
        lambda _api_key: FakeLlm(),
    )
    result = classify_intent_semantically("Could you stay with me for a moment?", api_key="test-key")
    assert result is not None
    assert result.intent is IntentType.CASUAL_CONVERSATION
    assert result.requires_rag is False
    assert result.routing_dict()["conversation_action"] == "RESPOND_CONVERSATIONALLY"
