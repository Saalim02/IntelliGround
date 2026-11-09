import asyncio
from types import SimpleNamespace

import pytest

from app.api.routes import voice
from app.services.security import auth_dependencies


class FakeWebSocket:
    def __init__(self, cookies=None):
        self.cookies = cookies or {}


class FakeQuery:
    def __init__(self, user):
        self.user = user

    def filter(self, *args):
        return self

    def first(self):
        return self.user


class FakeDb:
    def __init__(self, user=None):
        self.user = user
        self.expunge_calls = []

    def query(self, _model):
        return FakeQuery(self.user)

    def expunge(self, user):
        self.expunge_calls.append(user)

    def close(self):
        pass

    def commit(self):
        pass


def test_websocket_auth_uses_the_shared_local_development_identity(monkeypatch):
    user = SimpleNamespace(id=41, email="localdev@intelliground.local")
    db = FakeDb(user)
    monkeypatch.setattr(voice, "SessionLocal", lambda: db)
    monkeypatch.setattr(auth_dependencies, "LOCAL_DEV_AUTH_BYPASS", True)
    monkeypatch.setattr(auth_dependencies, "get_local_development_user", lambda _db: user)

    result = voice._websocket_user(FakeWebSocket())

    assert result is user
    assert db.expunge_calls == [user]


def test_websocket_auth_rejects_missing_cookie_when_bypass_disabled(monkeypatch):
    monkeypatch.setattr(voice, "SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(auth_dependencies, "LOCAL_DEV_AUTH_BYPASS", False)

    assert voice._websocket_user(FakeWebSocket()) is None


@pytest.mark.parametrize("decode_error", [ValueError("expired"), ValueError("invalid")])
def test_websocket_auth_rejects_invalid_access_token(monkeypatch, decode_error):
    monkeypatch.setattr(voice, "SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(auth_dependencies, "LOCAL_DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(voice, "decode_access_token", lambda _token: (_ for _ in ()).throw(decode_error))

    assert voice._websocket_user(FakeWebSocket({"access_token": "not-logged"})) is None


def test_websocket_auth_validates_user_activity_and_token_version(monkeypatch):
    inactive = SimpleNamespace(id=7, is_active=False, token_version=3)
    db = FakeDb(inactive)
    monkeypatch.setattr(voice, "SessionLocal", lambda: db)
    monkeypatch.setattr(auth_dependencies, "LOCAL_DEV_AUTH_BYPASS", False)
    monkeypatch.setattr(voice, "decode_access_token", lambda _token: (7, 3))

    assert voice._websocket_user(FakeWebSocket({"access_token": "opaque"})) is None

    active = SimpleNamespace(id=7, is_active=True, token_version=4)
    db.user = active
    assert voice._websocket_user(FakeWebSocket({"access_token": "opaque"})) is None


def test_stream_turn_preserves_conversation_context_and_tts(monkeypatch):
    captured = {}
    conversation = SimpleNamespace(
        id="conversation-1",
        user_id=9,
        response_language="hi",
        speech_output_language="hi",
        document_id="document-1",
    )
    user = SimpleNamespace(id=9)

    monkeypatch.setattr(voice, "SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(voice, "get_owned_conversation", lambda _db, _user, _id: conversation)
    monkeypatch.setattr(voice, "get_active_document", lambda _user_id: {"active_doc_id": "document-1", "active_filename": "source.pdf"})
    monkeypatch.setattr(voice, "recent_context", lambda _db, _conversation: "User: previous question")
    monkeypatch.setattr(voice, "add_turn", lambda *args, **kwargs: None)

    def fake_query(request, _user, endpoint):
        captured["request"] = request
        captured["endpoint"] = endpoint
        return {"status": "success", "answer": "उत्तर", "evidence": [{"page_number": 4, "snippet": "मूल अनुच्छेद"}], "evaluation": {"evaluation_id": "eval-1", "decision": "AUTO_APPROVE"}}

    monkeypatch.setattr(voice, "run_document_query", fake_query)
    monkeypatch.setattr(voice, "synthesize_speech", lambda text, language: (b"RIFF", "audio/wav"))

    async def no_stream(*_args, **_kwargs):
        if False:
            yield {}

    monkeypatch.setattr(voice, "stream_speech", no_stream)

    async def fake_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(voice.asyncio, "to_thread", fake_to_thread)

    class FakeSocket:
        def __init__(self):
            self.messages = []

        async def send_json(self, message):
            self.messages.append(message)

    socket = FakeSocket()
    asyncio.run(voice._process_stream_turn(
        socket,
        user,
        "conversation-1",
        "यह प्रश्न है",
        {"access_password": "project-password"},
        "provider-request-1",
    ))

    assert captured["endpoint"] == "/api/voice"
    assert captured["request"].conversation_context == "User: previous question"
    assert captured["request"].response_language == "hi"
    assert captured["request"].query_language == "hi"
    assert [message["type"] for message in socket.messages] == ["speaking", "audio"]
    speaking = socket.messages[0]
    assert speaking["evidence"] == [{"page_number": 4, "snippet": "मूल अनुच्छेद"}]
    assert speaking["evaluation"]["evaluation_id"] == "eval-1"


def test_stream_turn_speaks_review_message(monkeypatch):
    conversation = SimpleNamespace(
        id="conversation-1", user_id=9, response_language="en",
        speech_output_language="en", document_id="document-1",
    )
    user = SimpleNamespace(id=9)
    socket = type("Socket", (), {"messages": [], "send_json": lambda self, message: self.messages.append(message)})()

    async def send_json(self, message):
        self.messages.append(message)

    socket.send_json = send_json.__get__(socket)
    monkeypatch.setattr(voice, "SessionLocal", lambda: FakeDb())
    monkeypatch.setattr(voice, "get_owned_conversation", lambda *_args: conversation)
    monkeypatch.setattr(voice, "get_active_document", lambda _user_id: {"active_doc_id": "document-1", "active_filename": "source.pdf"})
    monkeypatch.setattr(voice, "recent_context", lambda *_args: "")
    monkeypatch.setattr(voice, "add_turn", lambda *args, **kwargs: None)
    monkeypatch.setattr(voice, "run_document_query", lambda *_args, **_kwargs: {"status": "human_review_required", "answer": None, "message": "A human review is required."})

    async def fake_stream(*_args, **_kwargs):
        if False:
            yield {}

    monkeypatch.setattr(voice, "stream_speech", fake_stream)
    monkeypatch.setattr(voice, "synthesize_speech", lambda text, language: (b"RIFF", "audio/wav"))
    monkeypatch.setattr(voice.asyncio, "to_thread", lambda function, *args, **kwargs: _immediate(function, *args, **kwargs))

    asyncio.run(voice._process_stream_turn(socket, user, "conversation-1", "please review", {}, "req-2"))

    assert [message["type"] for message in socket.messages] == ["answer", "audio"]


async def _immediate(function, *args, **kwargs):
    return function(*args, **kwargs)


