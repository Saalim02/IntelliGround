import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.services import document_intelligence_service as intelligence
from app.services import document_registry_service
from app.services.language_registry import is_language_supported_text
from app.main import app
from app.models.auth_models import User
from app.services.security.auth_dependencies import get_current_user


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, _input):
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def _chunk(doc_id="doc-a", chunk_id=7, page=3):
    return {
        "chunk_id": chunk_id,
        "file_name": "source.pdf",
        "page_number": page,
        "text": "The document explains digital public services and their conclusions.",
        "doc_id": doc_id,
    }


def _configure(monkeypatch, language, payload, doc_id="doc-a"):
    chunk = _chunk(doc_id=doc_id)
    record = SimpleNamespace(metadata=chunk, text=chunk["text"])
    monkeypatch.setattr(
        intelligence,
        "get_active_document",
        lambda user_id: {
            "active_doc_id": doc_id,
            "active_filename": "source.pdf",
            "response_language": language,
        },
    )
    monkeypatch.setattr(intelligence, "load_sparse_index", lambda _path: SimpleNamespace(records=[record]))
    monkeypatch.setattr(intelligence, "get_llm", lambda _api_key: FakeLLM(payload))


@pytest.mark.parametrize(
    ("language", "text"),
    [("kn", "ಕನ್ನಡ ವಿಷಯದ ಸಂಕ್ಷಿಪ್ತ ವಿವರಣೆ"), ("en", "A concise document summary"), ("hi", "दस्तावेज़ का संक्षिप्त सारांश")],
)
def test_summary_enforces_requested_output_language(monkeypatch, language, text):
    payload = {
        "short_summary": text,
        "detailed_summary": text,
        "evidence": [{"chunk_id": 7}],
        "main_conclusions": [{"text": text, "evidence": [{"chunk_id": 7}]}],
        "topics": [{"title": text, "explanation": text, "evidence": [{"chunk_id": 7}]}],
        "unsupported_items": [],
    }
    _configure(monkeypatch, language, payload)
    result = intelligence.generate_document_summary(11, "test-key")
    assert result["response_language"] == language
    assert result["topics"][0]["evidence"] == [{
        "page_number": 3,
        "chunk_id": 7,
        "text": "The document explains digital public services and their conclusions.",
    }]


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "This document explains the main public service process."),
        ("kn", "ಈ ದಾಖಲೆ ಸಾರ್ವಜನಿಕ ಸೇವೆಯ ಮುಖ್ಯ ಪ್ರಕ್ರಿಯೆಯನ್ನು ವಿವರಿಸುತ್ತದೆ."),
        ("hi", "यह दस्तावेज़ सार्वजनिक सेवा की मुख्य प्रक्रिया समझाता है।"),
        ("ta", "இந்த ஆவணம் பொது சேவையின் முக்கிய செயல்முறையை விளக்குகிறது."),
        ("te", "ఈ పత్రం ప్రజా సేవ యొక్క ప్రధాన ప్రక్రియను వివరిస్తుంది."),
        ("ml", "ഈ രേഖ പൊതുസേവനത്തിന്റെ പ്രധാന പ്രക്രിയ വിശദീകരിക്കുന്നു."),
        ("mr", "हा दस्तऐवज सार्वजनिक सेवेची मुख्य प्रक्रिया स्पष्ट करतो."),
        ("gu", "આ દસ્તાવેજ જાહેર સેવાની મુખ્ય પ્રક્રિયા સમજાવે છે."),
        ("bn", "এই নথিটি জনসেবার প্রধান প্রক্রিয়া ব্যাখ্যা করে।"),
        ("pa", "ਇਹ ਦਸਤਾਵੇਜ਼ ਜਨਤਕ ਸੇਵਾ ਦੀ ਮੁੱਖ ਪ੍ਰਕਿਰਿਆ ਸਮਝਾਉਂਦਾ ਹੈ।"),
        ("ur", "یہ دستاویز عوامی خدمت کے بنیادی عمل کی وضاحت کرتی ہے۔"),
    ],
)
def test_language_validation_supports_all_registered_response_languages(language, text):
    assert is_language_supported_text(text, language)


def test_language_validation_allows_mixed_technical_terms_but_rejects_wrong_script():
    assert is_language_supported_text("ಕನ್ನಡದಲ್ಲಿ API ಮತ್ತು PDF ಕುರಿತು ವಿವರಣೆ.", "kn")
    assert is_language_supported_text("यह PDF API के बारे में बताता है।", "hi")
    assert is_language_supported_text("हा PDF API विषयी माहिती देतो.", "mr")
    assert is_language_supported_text(
        "यह दस्तावेज़ NREGA 2025-26, e-Governance, API और PDF के बारे में महत्वपूर्ण जानकारी देता है।",
        "hi",
    )
    assert not is_language_supported_text("This is an English-only answer.", "ur")
    assert not is_language_supported_text("यह केवल हिंदी उत्तर है।", "pa")


@pytest.mark.parametrize(
    ("language", "text"),
    [
        ("en", "This document explains the main process."),
        ("kn", "ಈ ದಾಖಲೆ ಮುಖ್ಯ ಪ್ರಕ್ರಿಯೆಯನ್ನು ವಿವರಿಸುತ್ತದೆ."),
        ("hi", "यह दस्तावेज़ मुख्य प्रक्रिया समझाता है।"),
        ("ta", "இந்த ஆவணம் முக்கிய செயல்முறையை விளக்குகிறது."),
        ("te", "ఈ పత్రం ప్రధాన ప్రక్రియను వివరిస్తుంది."),
        ("ml", "ഈ രേഖ പ്രധാന പ്രക്രിയ വിശദീകരിക്കുന്നു."),
        ("mr", "हा दस्तऐवज मुख्य प्रक्रिया स्पष्ट करतो."),
        ("gu", "આ દસ્તાવેજ મુખ્ય પ્રક્રિયા સમજાવે છે."),
        ("bn", "এই নথিটি প্রধান প্রক্রিয়া ব্যাখ্যা করে।"),
        ("pa", "ਇਹ ਦਸਤਾਵੇਜ਼ ਮੁੱਖ ਪ੍ਰਕਿਰਿਆ ਸਮਝਾਉਂਦਾ ਹੈ।"),
        ("ur", "یہ دستاویز بنیادی عمل کی وضاحت کرتی ہے۔"),
    ],
)
def test_mocked_summary_accepts_each_registered_response_language(monkeypatch, language, text):
    _configure(monkeypatch, language, {
        "short_summary": text,
        "detailed_summary": text,
        "main_conclusions": [{"text": text, "evidence": [{"chunk_id": 7}] }],
        "topics": [{"title": text, "explanation": text, "evidence": [{"chunk_id": 7}]}],
        "unsupported_items": [],
    })
    result = intelligence.generate_document_summary(11, "test-key")
    assert result["response_language"] == language
    assert result["short_summary"] == text


def test_generation_diagnostics_name_the_actual_operation(monkeypatch, caplog):
    text = "A concise document summary"
    _configure(monkeypatch, "en", {
        "short_summary": text,
        "detailed_summary": text,
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    })
    caplog.set_level("INFO", logger="app.services.document_intelligence_service")
    intelligence.generate_document_summary(11, "test-key")
    assert any("generation_type=summary" in record.message for record in caplog.records)
    assert any("document_summary generation diagnostics" in record.message for record in caplog.records)
    assert not any("generation_type=unknown" in record.message for record in caplog.records)


def test_summary_retries_once_after_wrong_language_payload(monkeypatch):
    _configure(monkeypatch, "hi", {
        "short_summary": "अच्छा सारांश",
        "detailed_summary": "यह दस्तावेज़ महत्वपूर्ण जानकारी बताता है।",
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    })
    wrong = {
        "short_summary": "A short English summary",
        "detailed_summary": "This is an English detailed summary.",
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    }
    right = {
        "short_summary": "अच्छा सारांश",
        "detailed_summary": "यह दस्तावेज़ महत्वपूर्ण जानकारी बताता है।",
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    }
    calls = []
    def fake_invoke(*_args, **_kwargs):
        calls.append(True)
        return wrong if len(calls) == 1 else right

    monkeypatch.setattr(intelligence, "_invoke", fake_invoke)

    result = intelligence.generate_document_summary(11, "test-key")

    assert len(calls) == 2
    assert result["short_summary"] == right["short_summary"]


def test_explicit_urdu_request_overrides_persisted_english(monkeypatch):
    urdu = "یہ دستاویز عوامی خدمات کے بارے میں مختصر خلاصہ پیش کرتی ہے۔"
    payload = {
        "short_summary": urdu,
        "detailed_summary": urdu,
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    }
    _configure(monkeypatch, "en", payload)

    result = intelligence.generate_document_summary(11, "test-key", response_language="ur")

    assert result["response_language"] == "ur"


def test_topics_forward_explicit_response_language(monkeypatch):
    captured = {}

    def fake_summary(user_id, api_key, response_language=None):
        captured["language"] = response_language
        return {"status": "success", "active_document": "source.pdf", "response_language": response_language, "topics": [], "unsupported_items": [], "message": "ok"}

    monkeypatch.setattr(intelligence, "generate_document_summary", fake_summary)
    intelligence.generate_document_topics(11, "test-key", response_language="ur")

    assert captured["language"] == "ur"


def test_questions_use_explicit_response_language(monkeypatch):
    captured = {}
    chunks = [_chunk()]
    monkeypatch.setattr(intelligence, "_base", lambda _user_id, _api_key, response_language: ({"active_filename": "source.pdf"}, chunks, response_language))
    monkeypatch.setattr(intelligence, "_invoke", lambda _api_key, language, _chunks, _task, operation="document_intelligence": captured.update(language=language) or {"questions": [{"question": "یہ سوال کیا ہے؟", "question_type": "factual", "difficulty": "easy", "suggested_answer": None, "evidence": [{"chunk_id": 7}]}], "unsupported_items": []})

    result = intelligence.generate_document_questions(11, "test-key", response_language="ur")

    assert captured["language"] == "ur"
    assert result["response_language"] == "ur"


def test_invalid_language_is_rejected_by_api_model():
    from pydantic import ValidationError
    from app.models.document_intelligence_models import DocumentLanguageRequest

    with pytest.raises(ValidationError):
        DocumentLanguageRequest(response_language="xx")


def test_malformed_generation_fails_closed(monkeypatch):
    chunk = _chunk()
    record = SimpleNamespace(metadata=chunk, text=chunk["text"])
    monkeypatch.setattr(intelligence, "get_active_document", lambda _user_id: {"active_doc_id": "doc-a", "active_filename": "source.pdf", "response_language": "en"})
    monkeypatch.setattr(intelligence, "load_sparse_index", lambda _path: SimpleNamespace(records=[record]))

    class Malformed:
        def invoke(self, _input):
            return SimpleNamespace(content="not json")

    monkeypatch.setattr(intelligence, "get_llm", lambda _api_key: Malformed())
    with pytest.raises(ValueError, match="malformed JSON"):
        intelligence.generate_document_summary(11, "test-key")


def test_summary_validation_diagnostics_identify_language_failure(monkeypatch, caplog):
    _configure(monkeypatch, "ur", {
        "short_summary": "A summary in English.",
        "detailed_summary": "A detailed summary in English.",
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    })
    caplog.set_level("INFO", logger="app.services.document_intelligence_service")

    with pytest.raises(ValueError, match="language or structure validation"):
        intelligence.generate_document_summary(11, "test-key")

    assert any(
        "missing_short_summary=False" in record.message
        and "missing_detailed_summary=False" in record.message
        and "language_validation_failed=True" in record.message
        for record in caplog.records
    )


def test_legacy_summary_uses_persisted_language(monkeypatch):
    text = "A concise document summary"
    _configure(monkeypatch, "en", {
        "short_summary": text,
        "detailed_summary": text,
        "main_conclusions": [],
        "topics": [],
        "unsupported_items": [],
    })

    result = intelligence.generate_document_summary(11, "test-key")

    assert result["response_language"] == "en"


def test_unknown_citation_is_not_returned_as_grounded_topic(monkeypatch):
    payload = {
        "short_summary": "A concise document summary",
        "detailed_summary": "A detailed document summary",
        "main_conclusions": [],
        "topics": [{"title": "A topic", "explanation": "A supported explanation", "evidence": [{"chunk_id": 999}]}],
        "unsupported_items": [],
    }
    _configure(monkeypatch, "en", payload)
    result = intelligence.generate_document_summary(11, "test-key")
    assert result["topics"] == []
    assert result["unsupported_items"]


@pytest.mark.parametrize(
    ("language", "question"),
    [
        ("en", "What does NREGA 2025-26 say about e-Governance?"),
        ("kn", "ನರೇಗಾ ಯೋಜನೆಯ ಮುಖ್ಯ ಉದ್ದೇಶವೇನು?"),
        ("ta", "இந்த திட்டத்தின் முக்கிய நோக்கம் என்ன?"),
        ("te", "ఈ పథకం యొక్క ముఖ్య ఉద్దేశ్యం ఏమిటి?"),
    ],
)
def test_questions_accept_non_sequential_grounded_chunk_ids(monkeypatch, language, question):
    chunks = [_chunk(chunk_id=12, page=4), _chunk(chunk_id=91, page=9)]
    monkeypatch.setattr(
        intelligence,
        "_base",
        lambda _user_id, _api_key: (
            {"active_filename": "source.pdf"},
            chunks,
            language,
        ),
    )
    monkeypatch.setattr(
        intelligence,
        "_invoke",
        lambda *_args, **_kwargs: {
            "questions": [{
                "question": question,
                "question_type": "factual",
                "difficulty": "easy",
                "suggested_answer": None,
                "evidence": [{"chunk_id": 91}],
            }],
            "unsupported_items": [],
        },
    )

    result = intelligence.generate_document_questions(11, "test-key")

    assert len(result["questions"]) == 1
    assert result["questions"][0]["evidence"][0]["chunk_id"] == 91
    assert result["questions"][0]["evidence"][0]["page_number"] == 9


def test_questions_reject_missing_and_incorrect_chunk_ids(monkeypatch, caplog):
    caplog.set_level("INFO", logger="app.services.document_intelligence_service")
    chunks = [_chunk(chunk_id=12, page=4), _chunk(chunk_id=91, page=9)]
    monkeypatch.setattr(
        intelligence,
        "_base",
        lambda _user_id, _api_key: ({"active_filename": "source.pdf"}, chunks, "en"),
    )
    monkeypatch.setattr(
        intelligence,
        "_invoke",
        lambda *_args, **_kwargs: {
            "questions": [
                {"question": "What is covered?", "evidence": [{"chunk_id": 999}]},
                {"question": "What is missing?", "evidence": []},
            ],
            "unsupported_items": [],
        },
    )

    with pytest.raises(ValueError, match="no grounded questions"):
        intelligence.generate_document_questions(11, "test-key")

    assert any("invalid_reference_items=1" in record.message for record in caplog.records)
    assert any("accepted=0" in record.message for record in caplog.records)


def test_questions_with_no_grounded_items_fail_closed(monkeypatch):
    monkeypatch.setattr(
        intelligence,
        "_base",
        lambda _user_id, _api_key: ({"active_filename": "source.pdf"}, [_chunk(chunk_id=12)], "en"),
    )
    monkeypatch.setattr(
        intelligence,
        "_invoke",
        lambda *_args, **_kwargs: {"questions": [], "unsupported_items": ["Insufficient evidence"]},
    )

    with pytest.raises(ValueError, match="no grounded questions"):
        intelligence.generate_document_questions(11, "test-key")


def test_questions_empty_model_response_emits_diagnostics(monkeypatch, caplog):
    caplog.set_level("INFO", logger="app.services.document_intelligence_service")
    monkeypatch.setattr(
        intelligence,
        "_base",
        lambda _user_id, _api_key: ({"active_filename": "source.pdf"}, [_chunk(chunk_id=12)], "en"),
    )
    monkeypatch.setattr(intelligence, "_invoke", lambda *_args, **_kwargs: {"questions": []})

    with pytest.raises(ValueError, match="no grounded questions"):
        intelligence.generate_document_questions(11, "test-key")

    messages = [record.message for record in caplog.records]
    assert any("loaded_chunks=1" in message for message in messages)
    assert any("raw_questions=0" in message and "model_questions_empty=True" in message for message in messages)
    assert any("accepted=0" in message for message in messages)


def test_questions_missing_key_is_distinguished_from_empty_array(monkeypatch, caplog):
    caplog.set_level("INFO", logger="app.services.document_intelligence_service")
    monkeypatch.setattr(intelligence, "_base", lambda *_args: ({"active_filename": "source.pdf"}, [_chunk(chunk_id=12)], "en"))
    monkeypatch.setattr(intelligence, "_invoke", lambda *_args, **_kwargs: {"unsupported_items": ["Insufficient evidence"]})

    with pytest.raises(ValueError, match="no grounded questions"):
        intelligence.generate_document_questions(11, "test-key")

    messages = [record.message for record in caplog.records]
    assert any("questions_key_present=False" in message for message in messages)
    assert any("model_questions_missing=True" in message for message in messages)


def test_questions_malformed_model_response_emits_diagnostics(monkeypatch, caplog):
    caplog.set_level("INFO", logger="app.services.document_intelligence_service")
    monkeypatch.setattr(
        intelligence,
        "_base",
        lambda _user_id, _api_key: ({"active_filename": "source.pdf"}, [_chunk(chunk_id=12)], "en"),
    )
    monkeypatch.setattr(
        intelligence,
        "get_llm",
        lambda _api_key: SimpleNamespace(
            invoke=lambda _prompt: SimpleNamespace(content="{malformed-json")
        ),
    )

    with pytest.raises(ValueError, match="malformed JSON"):
        intelligence.generate_document_questions(11, "test-key")

    assert any(
        "document_questions model response diagnostics: model_response_empty_or_malformed=true" in record.message
        for record in caplog.records
    )


def test_registry_language_update_preserves_document_identity(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    monkeypatch.setattr(document_registry_service, "_get_registry_path", lambda _user_id=None: path)
    document_registry_service.save_active_document("doc-a", "source.pdf", user_id=5, input_language="hin", evidence_language="hi", response_language="en")
    document_registry_service.update_document_languages(5, response_language="kn")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["active_doc_id"] == "doc-a"
    assert data["active_filename"] == "source.pdf"
    assert data["input_language"] == "hin"
    assert data["evidence_language"] == "hi"
    assert data["response_language"] == "kn"


def test_language_and_summary_endpoints_are_authenticated_and_contractual(monkeypatch):
    user = User(id=42, email="intelligence@example.com", is_active=True)
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        monkeypatch.setattr(
            "app.api.routes.document_intelligence.save_response_language",
            lambda user_id, response_language: {
                "status": "success",
                "active_document": "source.pdf",
                "input_language": "hin",
                "evidence_language": "hi",
                "response_language": response_language,
                "message": "saved",
            },
        )
        monkeypatch.setattr(
            "app.api.routes.document_intelligence.resolve_authorized_api_key",
            lambda **_kwargs: "test-key",
        )
        monkeypatch.setattr(
            "app.api.routes.document_intelligence.generate_document_summary",
            lambda _user_id, _api_key, response_language=None: {
                "status": "success",
                "active_document": "source.pdf",
                "response_language": response_language,
                "short_summary": "ಕನ್ನಡ ಸಾರಾಂಶ",
                "detailed_summary": "ಕನ್ನಡ ವಿವರ",
                "evidence": [],
                "main_conclusions": [],
                "topics": [],
                "unsupported_items": [],
                "message": "ok",
            },
        )
        client = TestClient(app)
        language_response = client.post("/api/document/language", json={"response_language": "kn"})
        summary_response = client.post("/api/document/summary", json={"response_language": "ur"})
        assert language_response.status_code == 200
        assert language_response.json()["response_language"] == "kn"
        assert summary_response.status_code == 200
        assert summary_response.json()["response_language"] == "ur"
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_current_document_endpoint_returns_persisted_language_metadata(monkeypatch):
    user = User(id=42, email="current@example.com", is_active=True)
    app.dependency_overrides[get_current_user] = lambda: user
    monkeypatch.setattr(
        "app.api.routes.document_intelligence.get_active_document",
        lambda user_id: {
            "active_doc_id": "doc-a",
            "active_filename": "source.pdf",
            "evidence_language": "kn",
            "response_language": "en",
        } if user_id == 42 else None,
    )
    try:
        response = TestClient(app).get("/api/document/current")
        assert response.status_code == 200
        assert response.json()["document"]["evidence_language"] == "kn"
        assert response.json()["document"]["response_language"] == "en"
    finally:
        app.dependency_overrides.pop(get_current_user, None)


def test_evidence_loader_isolated_to_active_user_document(monkeypatch):
    records = [
        SimpleNamespace(metadata=_chunk(doc_id="doc-a", chunk_id=1), text="A"),
        SimpleNamespace(metadata=_chunk(doc_id="doc-b", chunk_id=2), text="B"),
    ]
    monkeypatch.setattr(intelligence, "get_active_document", lambda user_id: {"active_doc_id": f"doc-{user_id}", "active_filename": "source.pdf", "response_language": "en"})
    monkeypatch.setattr(intelligence, "load_sparse_index", lambda _path: SimpleNamespace(records=records))
    monkeypatch.setattr(intelligence, "get_user_vectorstore_dir", lambda user_id: user_id)
    _active, chunks = intelligence._load_evidence("c")
    assert chunks == []
    monkeypatch.setattr(intelligence, "get_active_document", lambda _user_id: {"active_doc_id": "doc-a", "active_filename": "source.pdf", "response_language": "en"})
    _active, chunks = intelligence._load_evidence(1)
    assert [chunk["chunk_id"] for chunk in chunks] == [1]
