from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.routes import voice
from app.core.database import Base
from app.models.auth_models import User
from app.models.voice_models import VoicePreferenceRequest, VoicePreviewRequest
from app.services import gnani_timbre_service as timbre
from app.services.voice_catalogue import available_voices, validate_voice_language


def test_catalogue_exposes_all_verified_v25_voices_and_metadata():
    voices = available_voices()
    assert len(voices) == 42
    assert all(voice["id"] for voice in voices)
    assert len({voice["id"] for voice in voices}) == 42
    assert all(voice["model"] == "timbre-v2.5" for voice in voices)
    assert {voice["gender"] for voice in voices} == {"Female", "Male"}
    assert all(voice["languages"] for voice in voices)


def test_catalogue_rejects_unknown_voice_and_language():
    with pytest.raises(ValueError):
        validate_voice_language("not-a-provider-voice", "en")
    with pytest.raises(ValueError):
        validate_voice_language(available_voices()[0]["id"], "ur")


def test_preview_validates_voice_before_calling_timbre(monkeypatch):
    called = {}
    monkeypatch.setattr(voice, "synthesize_speech", lambda text, **kwargs: called.update(kwargs) or (b"RIFF", "audio/wav"))
    request = VoicePreviewRequest(voice_id=available_voices()[0]["id"], language="en", text="Hello")
    response = voice.preview_voice(request, SimpleNamespace(id=1))
    assert response.body == b"RIFF"
    assert called["voice"] == request.voice_id


def test_preferences_are_isolated_per_user():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        first = User(email="first@example.com", auth_provider="local", is_active=True)
        second = User(email="second@example.com", auth_provider="local", is_active=True)
        db.add_all([first, second])
        db.commit()
        voice_id = available_voices()[0]["id"]
        voice.save_voice_preference(VoicePreferenceRequest(voice_id=voice_id, response_emotion="PROFESSIONAL"), first, db)
        saved = voice.get_voice_preference(first, db)
        assert saved["voice_id"] == voice_id
        assert saved["response_emotion"] == "PROFESSIONAL"
        assert voice.get_voice_preference(second, db)["voice_id"] is None
    finally:
        db.close()


def test_selected_voice_is_sent_to_buffered_and_streaming_payloads(monkeypatch):
    monkeypatch.setattr(timbre, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(timbre, "GNANI_TTS_MODEL", "timbre-v2.5")
    monkeypatch.setattr(timbre.httpx, "post", lambda *args, **kwargs: SimpleNamespace(status_code=200, content=b"RIFF", headers={"content-type": "audio/wav"}))
    captured = {}
    original = timbre._payload
    monkeypatch.setattr(timbre, "_payload", lambda text, language, container, voice=None: captured.update(voice=voice) or original(text, language, container, voice))
    timbre.synthesize_speech("Hello", language="en", voice="Kaveri")
    assert captured["voice"] == "Kaveri"
