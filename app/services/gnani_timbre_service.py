"""Gnani Timbre REST and realtime WebSocket integrations."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator

import httpx
import websockets

from app.core.config import GNANI_API_KEY, GNANI_REQUEST_TIMEOUT_SECONDS, GNANI_TTS_MODEL, GNANI_TTS_URL, GNANI_TTS_VOICE, GNANI_TTS_WS_URL
from app.services.gnani_prisma_service import VOICE_LANGUAGE_CODES, GnaniConfigurationError, GnaniProviderError
from app.services.observability.langsmith_service import observed


def _validate(text: str, language: str) -> None:
    if not text.strip():
        raise GnaniProviderError("Text-to-speech input is empty.")
    if not GNANI_API_KEY or not GNANI_TTS_MODEL or not GNANI_TTS_VOICE:
        raise GnaniConfigurationError("Gnani Timbre requires GNANI_API_KEY, GNANI_TTS_MODEL, and GNANI_TTS_VOICE.")
    if language not in VOICE_LANGUAGE_CODES:
        raise GnaniConfigurationError(f"Gnani Timbre does not have a verified mapping for language '{language}'.")


def _payload(text: str, language: str, container: str) -> dict:
    return {"text": text, "voice": GNANI_TTS_VOICE, "model": GNANI_TTS_MODEL, "language": VOICE_LANGUAGE_CODES[language], "audio_config": {"sample_rate": 48000, "num_channels": 1, "sample_width": 2, "encoding": "linear_pcm", "container": container}}


@observed("voice.tts", voice_stage="tts", provider="gnani_timbre")
def synthesize_speech(text: str, *, language: str) -> tuple[bytes, str]:
    """Buffered REST synthesis used by non-realtime callers and fallback."""
    _validate(text, language)
    try:
        response = httpx.post(GNANI_TTS_URL, headers={"X-API-Key-ID": GNANI_API_KEY, "Content-Type": "application/json"}, json=_payload(text, language, "wav"), timeout=GNANI_REQUEST_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise GnaniProviderError("Gnani Timbre could not be reached.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("message")
        except ValueError:
            detail = None
        raise GnaniProviderError(detail or "Gnani Timbre rejected the synthesis request.")
    return response.content, response.headers.get("content-type", "audio/wav")


def _decode_audio_event(message: dict) -> bytes | None:
    data = message.get("data")
    encoded = data.get("audio") if isinstance(data, dict) else message.get("audio")
    if not encoded:
        return None
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise GnaniProviderError("Gnani Timbre returned an invalid audio chunk.") from exc


async def stream_speech(text: str, *, language: str) -> AsyncIterator[dict]:
    """Yield ordered raw PCM chunks from the documented Timbre WebSocket."""
    _validate(text, language)
    try:
        async with websockets.connect(GNANI_TTS_WS_URL, additional_headers={"X-API-Key-ID": GNANI_API_KEY, "Content-Type": "application/json"}, open_timeout=GNANI_REQUEST_TIMEOUT_SECONDS, close_timeout=2) as provider:
            await provider.send(json.dumps(_payload(text, language, "raw")))
            sequence = 0
            async for raw in provider:
                if isinstance(raw, bytes):
                    raise GnaniProviderError("Gnani Timbre returned an unexpected binary frame.")
                try:
                    message = json.loads(raw)
                except (TypeError, json.JSONDecodeError) as exc:
                    raise GnaniProviderError("Gnani Timbre returned invalid JSON.") from exc
                event_type = message.get("type")
                if event_type == "error":
                    raise GnaniProviderError(str(message.get("message") or message.get("error") or "Gnani Timbre stream failed."))
                if event_type == "complete":
                    return
                if event_type != "audio":
                    continue
                audio = _decode_audio_event(message)
                if audio is None:
                    continue
                data = message.get("data") if isinstance(message.get("data"), dict) else {}
                yield {"audio": audio, "sequence": sequence, "sample_rate": 48000, "channels": 1, "sample_width": 2, "encoding": "linear_pcm", "is_final": bool(message.get("is_final") or data.get("is_final"))}
                sequence += 1
    except asyncio.CancelledError:
        raise
    except (GnaniConfigurationError, GnaniProviderError):
        raise
    except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as exc:
        raise GnaniProviderError("Gnani Timbre realtime stream could not be reached.") from exc
