"""Gnani Prisma v3 REST and WebSocket integration."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import httpx

from app.core.config import (
    GNANI_API_KEY,
    GNANI_MAX_AUDIO_BYTES,
    GNANI_REQUEST_TIMEOUT_SECONDS,
    GNANI_STT_MODEL,
    GNANI_STT_URL,
    GNANI_STT_WS_URL,
)
from app.services.language_registry import GNANI_VOICE_LANGUAGE_CODES, provider_capability


logger = logging.getLogger(__name__)


VOICE_LANGUAGE_CODES = dict(GNANI_VOICE_LANGUAGE_CODES)

# This is the single allow-list for languages that have a verified Gnani
# speech mapping.  Urdu remains a supported text-response language, but is
# deliberately absent here until Gnani voice support is verified.
SUPPORTED_GNANI_SPEECH_LANGUAGES = frozenset(VOICE_LANGUAGE_CODES)


class GnaniConfigurationError(RuntimeError):
    pass


class GnaniProviderError(RuntimeError):
    pass


SUPPORTED_REST_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})


def _sanitized_provider_body(body: str, limit: int = 500) -> str:
    """Return a short provider error body with credential-like fields removed."""
    sanitized = re.sub(
        r"(?i)(api[-_ ]?key|authorization|token|secret|password)\s*[:=]\s*[^,;\s}]+",
        r"\1=[REDACTED]",
        body or "",
    )
    sanitized = " ".join(sanitized.split())
    return sanitized[:limit]


def _convert_webm_to_wav(audio: bytes, filename: str) -> tuple[bytes, str, str]:
    """Convert browser WebM/Opus audio into Prisma's documented WAV input."""
    if shutil.which("ffmpeg") is None:
        raise GnaniProviderError(
            "WebM audio is not supported by Prisma REST and ffmpeg is unavailable for conversion."
        )
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1",
            ],
            input=audio,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GnaniProviderError("WebM audio conversion failed.") from exc
    if result.returncode != 0 or not result.stdout:
        detail = _sanitized_provider_body(result.stderr.decode("utf-8", errors="replace"))
        raise GnaniProviderError(
            f"WebM audio conversion failed{(': ' + detail) if detail else '.'}"
        )
    return result.stdout, f"{Path(filename or 'recording').stem}.wav", "audio/wav"


def _normalize_audio(audio: bytes, filename: str, content_type: str | None) -> tuple[bytes, str, str]:
    name = filename or "audio.wav"
    extension = Path(name).suffix.lower()
    mime = (content_type or "").split(";", 1)[0].lower()
    if extension == ".webm" or mime == "audio/webm":
        return _convert_webm_to_wav(audio, name)
    if extension not in SUPPORTED_REST_AUDIO_EXTENSIONS:
        raise GnaniProviderError(
            "Unsupported audio format. Prisma REST accepts WAV, MP3, FLAC, OGG, or M4A."
        )
    return audio, name, content_type or "application/octet-stream"


def _require_key() -> str:
    if not GNANI_API_KEY:
        raise GnaniConfigurationError("Gnani API access is not configured.")
    return GNANI_API_KEY


def to_prisma_language(language: str | None) -> str:
    if not language or language == "auto":
        return "indic-auto"
    if language in VOICE_LANGUAGE_CODES:
        return VOICE_LANGUAGE_CODES[language]
    if language in VOICE_LANGUAGE_CODES.values():
        return language
    raise GnaniConfigurationError(
        f"Gnani Prisma does not have a verified mapping for language '{language}'."
    )


def transcribe_audio(
    audio: bytes,
    *,
    filename: str,
    content_type: str | None,
    language: str | None = None,
) -> dict:
    if not audio:
        raise GnaniProviderError("Audio input is empty.")
    if len(audio) > GNANI_MAX_AUDIO_BYTES:
        raise GnaniProviderError("Audio input exceeds the configured size limit.")

    normalized_audio, normalized_filename, normalized_content_type = _normalize_audio(
        audio, filename, content_type
    )
    api_key = _require_key()
    request_id = str(uuid.uuid4())
    files = {
        "audio_file": (
            normalized_filename,
            normalized_audio,
            normalized_content_type,
        )
    }
    data = {
        "model": GNANI_STT_MODEL,
        "language_code": to_prisma_language(language),
        "format": "transcribe",
    }
    try:
        response = httpx.post(
            GNANI_STT_URL,
            headers={
                "X-API-Key-ID": api_key,
                "X-API-Request-ID": request_id,
            },
            files=files,
            data=data,
            timeout=GNANI_REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        raise GnaniProviderError("Gnani Prisma could not be reached.") from exc

    if response.status_code >= 400:
        logger.warning(
            "Gnani Prisma STT rejected audio: status=%s content_type=%s filename=%s bytes=%d body=%s",
            response.status_code,
            normalized_content_type,
            normalized_filename,
            len(normalized_audio),
            _sanitized_provider_body(response.text),
        )

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code >= 400:
        message = payload.get("message") if isinstance(payload, dict) else None
        raise GnaniProviderError(message or "Gnani Prisma rejected the audio request.")
    if not isinstance(payload, dict) or not payload.get("transcript"):
        raise GnaniProviderError("Gnani Prisma returned no transcript.")
    return {
        "transcript": str(payload["transcript"]),
        "detected_language_code": payload.get("detected_language_code"),
        "request_id": payload.get("request_id", request_id),
        "model": payload.get("model", GNANI_STT_MODEL),
    }


def prisma_websocket_headers(language: str | None = None) -> dict[str, str]:
    return {
        "x-api-key-id": _require_key(),
        "language_code": to_prisma_language(language),
        "model": GNANI_STT_MODEL,
        "format": "transcribe",
        "x-sample-rate": "16000",
        "x-min-silence-ms": "700",
    }


def parse_provider_message(message: str | bytes) -> dict:
    if isinstance(message, bytes):
        raise GnaniProviderError("Gnani Prisma returned an unexpected binary message.")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise GnaniProviderError("Gnani Prisma returned malformed event data.") from exc
    if not isinstance(payload, dict):
        raise GnaniProviderError("Gnani Prisma returned an invalid event.")
    return payload


def provider_detected_language(payload: dict) -> str | None:
    """Read Prisma's language metadata without trusting arbitrary event data.

    Prisma deployments have returned the language on either the event itself
    or its metadata/data object.  Only recognized application language codes
    are returned; script detection remains the controlled fallback.
    """
    candidates = [payload]
    for key in ("data", "metadata", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for item in candidates:
        for key in ("detected_language_code", "detected_language", "language_code", "language"):
            value = item.get(key)
            if not isinstance(value, str):
                continue
            normalized = value.strip().lower().replace("_", "-")
            base = normalized.split("-", 1)[0]
            if base in GNANI_VOICE_LANGUAGE_CODES or base == "ur":
                return base
    return None


GNANI_STT_WS_ENDPOINT = GNANI_STT_WS_URL
