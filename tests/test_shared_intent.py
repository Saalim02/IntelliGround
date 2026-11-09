from app.services.voice_intent_service import IntentType, classify_intent


def test_greetings_bypass_retrieval():
    for text in ("Hi.", "Hello."):
        result = classify_intent(text)
        assert result.intent is IntentType.GREETING
        assert result.requires_retrieval is False
        assert result.page_numbers == ()


def test_general_conversation_is_not_greeting_or_document_query():
    for text in ("Hey, hello, can you hear me?", "What are you doing?", "Tell me about yourself"):
        result = classify_intent(text)
        assert result.intent is IntentType.GENERAL_CONVERSATION
        assert result.requires_retrieval is False


def test_document_overview_and_summary_are_distinct():
    assert classify_intent("What is this PDF about?").intent is IntentType.DOCUMENT_OVERVIEW
    assert classify_intent("Explain the important points.").intent is IntentType.DOCUMENT_OVERVIEW
    assert classify_intent("Summarize this document.").intent is IntentType.DOCUMENT_SUMMARY


def test_application_help_does_not_require_document_retrieval():
    result = classify_intent("How do I upload a PDF?")
    assert result.intent is IntentType.APPLICATION_HELP
    assert result.requires_retrieval is False


def test_page_query_extracts_explicit_page_numbers():
    result = classify_intent("What is on page 43?")
    assert result.intent is IntentType.PAGE_SPECIFIC_QUERY
    assert result.page_numbers == (43,)
    assert result.retrieval_strategy == "DIRECT_PAGE"
    assert classify_intent("Read page no. 45").page_numbers == (45,)


def test_follow_up_requires_context_when_reference_is_ambiguous():
    assert classify_intent("Explain the second topic.").intent is IntentType.AMBIGUOUS_QUERY
    assert classify_intent("Tell me more.").intent is IntentType.AMBIGUOUS_QUERY
    assert classify_intent("Which page mentions it?").intent is IntentType.AMBIGUOUS_QUERY
    assert classify_intent("What about that?").intent is IntentType.AMBIGUOUS_QUERY

    context = "User: What are the main topics?\nAssistant: Topic one and topic two."
    for text in ("Explain the second topic.", "Tell me more.", "Which page mentions it?", "What about that?"):
        assert classify_intent(text, conversation_context=context).intent is IntentType.FOLLOW_UP_QUERY


def test_short_factual_question_uses_document_fallback():
    result = classify_intent("What does NREGA mean?")
    assert result.intent is IntentType.DOCUMENT_QUERY
    assert result.requires_retrieval is True
    assert result.retrieval_strategy == "HYBRID"


def test_known_external_question_is_out_of_scope_without_facts():
    result = classify_intent("What is the weather today?")
    assert result.intent is IntentType.OUT_OF_SCOPE
    assert result.requires_retrieval is False


def test_multilingual_greetings_document_questions_and_pages():
    assert classify_intent("नमस्ते").intent is IntentType.GREETING
    assert classify_intent("இந்த ஆவணம் எதைப் பற்றி?").intent is IntentType.DOCUMENT_OVERVIEW
    assert classify_intent("इस PDF का सारांश दें").intent is IntentType.DOCUMENT_SUMMARY
    result = classify_intent("ಪುಟ 12ರಲ್ಲಿ ಏನಿದೆ?")
    assert result.intent is IntentType.PAGE_SPECIFIC_QUERY
    assert result.page_numbers == (12,)


def test_result_contains_no_evidence_or_generated_document_facts():
    result = classify_intent("What is on page 43?")
    payload = result.as_dict()
    assert set(payload) == {
        "intent", "confidence", "page_numbers", "requires_retrieval", "requires_rag",
        "requires_evaluation", "referenced_page", "referenced_topic",
        "response_action", "response_language", "retrieval_strategy",
    }
    assert "evidence" not in payload
    assert "answer" not in payload
