import asyncio
import base64
import json

import pytest

from app.services import gnani_prisma_service as prisma
from app.services import gnani_timbre_service as timbre


class FakeResponse:
    def __init__(self, status_code=200, payload=None, content=b"audio", text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.text = text
        self.headers = {"content-type": "audio/wav"}

    def json(self):
        return self._payload


def test_prisma_uses_documented_multipart_contract(monkeypatch):
    calls = {}
    monkeypatch.setattr(prisma, "GNANI_API_KEY", "test-key")

    def fake_post(url, **kwargs):
        calls.update(url=url, kwargs=kwargs)
        return FakeResponse(payload={"transcript": "hello", "detected_language_code": "en-IN"})

    monkeypatch.setattr(prisma.httpx, "post", fake_post)
    result = prisma.transcribe_audio(b"wav", filename="x.wav", content_type="audio/wav", language="en")

    assert calls["url"] == "https://api.vachana.ai/stt/v3"
    assert calls["kwargs"]["headers"]["X-API-Key-ID"] == "test-key"
    assert calls["kwargs"]["files"]["audio_file"][0] == "x.wav"
    assert calls["kwargs"]["data"]["language_code"] == "en-IN"
    assert result["transcript"] == "hello"


def test_prisma_provider_language_metadata_is_normalized():
    assert prisma.provider_detected_language({"type": "transcript", "data": {"detected_language_code": "hi-IN"}}) == "hi"
    assert prisma.provider_detected_language({"type": "transcript", "language_code": "ta-IN"}) == "ta"
    assert prisma.provider_detected_language({"type": "transcript", "language": "unknown"}) is None


def test_prisma_logs_sanitized_upstream_rejection(monkeypatch, caplog):
    monkeypatch.setattr(prisma, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(
        prisma.httpx,
        "post",
        lambda *args, **kwargs: FakeResponse(
            status_code=415,
            payload={"message": "unsupported media type"},
            text="unsupported media type api_key=should-not-appear",
        ),
    )

    with pytest.raises(prisma.GnaniProviderError, match="unsupported media type"):
        prisma.transcribe_audio(
            b"wav", filename="clip.wav", content_type="audio/wav", language="en"
        )

    record = next(record for record in caplog.records if "STT rejected audio" in record.message)
    assert "status=415" in record.message
    assert "filename=clip.wav" in record.message
    assert "bytes=3" in record.message
    assert "should-not-appear" not in record.message


def test_prisma_rejects_unsupported_audio_format(monkeypatch):
    monkeypatch.setattr(prisma, "GNANI_API_KEY", "test-key")

    with pytest.raises(prisma.GnaniProviderError, match="Unsupported audio format"):
        prisma.transcribe_audio(
            b"audio", filename="clip.aac", content_type="audio/aac", language="en"
        )


def test_prisma_converts_webm_before_upload(monkeypatch):
    calls = {}
    monkeypatch.setattr(prisma, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(prisma.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    class Converted:
        returncode = 0
        stdout = b"RIFF-wav"
        stderr = b""

    monkeypatch.setattr(prisma.subprocess, "run", lambda *args, **kwargs: Converted())

    def fake_post(url, **kwargs):
        calls.update(kwargs=kwargs)
        return FakeResponse(payload={"transcript": "converted"})

    monkeypatch.setattr(prisma.httpx, "post", fake_post)
    result = prisma.transcribe_audio(
        b"webm", filename="clip.webm", content_type="audio/webm;codecs=opus", language="en"
    )

    assert result["transcript"] == "converted"
    assert calls["kwargs"]["files"]["audio_file"] == ("clip.wav", b"RIFF-wav", "audio/wav")


def test_timbre_returns_binary_audio(monkeypatch):
    monkeypatch.setattr(timbre, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(timbre, "GNANI_TTS_MODEL", "timbre-v2.5")
    monkeypatch.setattr(timbre, "GNANI_TTS_VOICE", "Kaveri")
    monkeypatch.setattr(timbre.httpx, "post", lambda *args, **kwargs: FakeResponse(content=b"RIFF"))

    audio, content_type = timbre.synthesize_speech("Hello", language="en")

    assert audio == b"RIFF"
    assert content_type == "audio/wav"


def test_timbre_rejects_unverified_urdu_voice_output(monkeypatch):
    monkeypatch.setattr(timbre, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(timbre, "GNANI_TTS_MODEL", "timbre-v2.5")
    monkeypatch.setattr(timbre, "GNANI_TTS_VOICE", "Kaveri")

    with pytest.raises(timbre.GnaniConfigurationError):
        timbre.synthesize_speech("مرحبا", language="ur")


class FakeTimbreSocket:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def send(self, message):
        self.sent.append(json.loads(message))

    def __aiter__(self):
        async def iterator():
            for message in self.messages:
                yield message
        return iterator()


def test_timbre_realtime_preserves_chunk_order_and_payload(monkeypatch):
    monkeypatch.setattr(timbre, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(timbre, "GNANI_TTS_MODEL", "timbre-v2.5")
    monkeypatch.setattr(timbre, "GNANI_TTS_VOICE", "Kaveri")
    socket = FakeTimbreSocket([
        json.dumps({"type": "start", "data": {"request_id": "req-1"}}),
        json.dumps({"type": "audio", "data": {"audio": base64.b64encode(b"one").decode()}}),
        json.dumps({"type": "audio", "data": {"audio": base64.b64encode(b"two").decode(), "is_final": True}}),
        json.dumps({"type": "complete"}),
    ])
    monkeypatch.setattr(timbre.websockets, "connect", lambda *args, **kwargs: socket)

    chunks = asyncio.run(_collect(timbre.stream_speech("Hello", language="en")))

    assert [chunk["audio"] for chunk in chunks] == [b"one", b"two"]
    assert [chunk["sequence"] for chunk in chunks] == [0, 1]
    assert chunks[-1]["is_final"] is True
    assert socket.sent[0]["audio_config"]["container"] == "raw"
    assert socket.sent[0]["language"] == "en-IN"


def test_timbre_realtime_provider_error_is_not_silenced(monkeypatch):
    monkeypatch.setattr(timbre, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(timbre, "GNANI_TTS_MODEL", "timbre-v2.5")
    monkeypatch.setattr(timbre, "GNANI_TTS_VOICE", "Kaveri")
    socket = FakeTimbreSocket([json.dumps({"type": "error", "message": "unauthorized"})])
    monkeypatch.setattr(timbre.websockets, "connect", lambda *args, **kwargs: socket)

    with pytest.raises(timbre.GnaniProviderError, match="unauthorized"):
        asyncio.run(_collect(timbre.stream_speech("Hello", language="en")))


def test_timbre_realtime_cancellation_closes_provider_stream(monkeypatch):
    monkeypatch.setattr(timbre, "GNANI_API_KEY", "test-key")
    monkeypatch.setattr(timbre, "GNANI_TTS_MODEL", "timbre-v2.5")
    monkeypatch.setattr(timbre, "GNANI_TTS_VOICE", "Kaveri")
    started = asyncio.Event()
    closed = False

    class BlockingSocket(FakeTimbreSocket):
        def __aiter__(self):
            async def iterator():
                nonlocal closed
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    closed = True
                yield ""
            return iterator()

    monkeypatch.setattr(timbre.websockets, "connect", lambda *args, **kwargs: BlockingSocket([]))

    async def scenario():
        task = asyncio.create_task(_collect(timbre.stream_speech("Hello", language="en")))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert closed is True


async def _collect(iterator):
    return [chunk async for chunk in iterator]
