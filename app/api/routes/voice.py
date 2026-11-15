from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from contextlib import suppress
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.models.auth_models import User, VoicePreference
from app.models.query_models import QueryRequest
from app.models.voice_models import OnboardingSpeechRequest, VoicePreviewRequest, VoicePreferenceRequest, VoiceSessionCreateRequest, VoiceSynthesisRequest, VoiceTranscriptionResponse
from app.services.conversation_service import (
    NO_DOCUMENT_ID,
    NO_DOCUMENT_NAME,
    add_turn_if_new,
    create_conversation,
    get_owned_conversation,
    recent_context,
    serialize_conversation,
    validate_conversation_document,
)

# Keep the existing route-level seam used by integrations and tests while the
# implementation underneath remains retry-safe.
def add_turn(*args, **kwargs):
    return add_turn_if_new(*args, **kwargs)
from app.services.document_registry_service import document_artifacts_available, get_active_document
from app.services.gnani_prisma_service import (
    GnaniConfigurationError,
    GnaniProviderError,
    parse_provider_message,
    provider_detected_language,
    prisma_websocket_headers,
    transcribe_audio,
)
from app.services.gnani_timbre_service import SUPPORTED_TTS_EMOTIONS, stream_speech, synthesize_speech
from app.services.rag_orchestration_service import run_document_query
from app.services.language_detection_service import detect_voice_language
from app.services.security import auth_dependencies
from app.services.security.auth_dependencies import get_current_user
from app.services.security.auth_service import decode_access_token
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.voice_response_templates import localized_voice_message
from app.services.voice_catalogue import available_voices, get_voice, validate_voice_language
from app.services.emotion_state_service import get_settings, normalize_response_emotion, progress_announcement

router = APIRouter(prefix="/voice", tags=["Voice"])
logger = logging.getLogger(__name__)
ONBOARDING_SPEECH_MAX_BODY_BYTES = 8 * 1024


def _saved_voice(db: Session, user: User, language: str | None = None) -> str | None:
    preference = db.query(VoicePreference).filter(VoicePreference.user_id == user.id).first()
    if not preference or not get_voice(preference.voice_id):
        return None
    if language:
        try:
            validate_voice_language(preference.voice_id, language)
        except ValueError:
            # Do not invent a voice identifier or silently speak another
            # language. The Timbre default is used only when its catalogue
            # mapping is verified by the synthesis layer.
            return None
    return preference.voice_id


def _saved_response_emotion(db: Session, user: User) -> str:
    preference = db.query(VoicePreference).filter(VoicePreference.user_id == user.id).first()
    return normalize_response_emotion(preference.response_emotion if preference else None)


def _synthesize_for_user(db: Session, user: User, text: str, language: str, requested_voice: str | None = None):
    # Keeping direct-call compatibility is useful for integrations that invoke
    # this route function in isolation; FastAPI supplies a real Session in HTTP.
    if not hasattr(db, "query"):
        return synthesize_speech(text, language=language, **({"voice": requested_voice} if requested_voice else {}))
    voice_id = requested_voice or _saved_voice(db, user)
    if requested_voice and get_voice(requested_voice) is None:
        raise ValueError("This voice is not available in the configured Gnani Timbre v2.5 catalogue.")
    if requested_voice:
        validate_voice_language(requested_voice, language)
    if voice_id:
        return synthesize_speech(text, language=language, voice=voice_id)
    return synthesize_speech(text, language=language)


@router.get("/voices")
def list_voices(current_user: User = Depends(get_current_user)):
    del current_user
    return {"voices": available_voices()}


@router.post("/voices/preview")
def preview_voice(
    request: VoicePreviewRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        validate_voice_language(request.voice_id, request.language)
        audio, content_type = synthesize_speech(request.text, language=request.language, voice=request.voice_id)
        return Response(content=audio, media_type=content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc


@router.get("/preferences")
def get_voice_preference(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    preference = db.query(VoicePreference).filter(VoicePreference.user_id == current_user.id).first()
    saved_voice_id = preference.voice_id if preference else None
    effective_voice_id = saved_voice_id if saved_voice_id and get_voice(saved_voice_id) else None
    return {
        "voice_id": effective_voice_id,
        "saved_voice_id": saved_voice_id,
        "fallback": bool(saved_voice_id and effective_voice_id is None),
        "language": preference.language if preference else None,
        "response_emotion": normalize_response_emotion(preference.response_emotion if preference else None),
    }


@router.put("/preferences")
def save_voice_preference(
    request: VoicePreferenceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        voice = validate_voice_language(request.voice_id, request.language or "en")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    preference = db.query(VoicePreference).filter(VoicePreference.user_id == current_user.id).first()
    if preference is None:
        preference = VoicePreference(user_id=current_user.id, voice_id=voice["id"], language=request.language, response_emotion=request.response_emotion)
        db.add(preference)
    else:
        preference.voice_id = voice["id"]
        preference.language = request.language
        preference.response_emotion = request.response_emotion
    db.commit()
    db.refresh(preference)
    return {"voice_id": preference.voice_id, "language": preference.language}


async def validate_onboarding_speech_size(request: Request) -> None:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > ONBOARDING_SPEECH_MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Speech request is too large.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
    # Reading here also protects requests that omit Content-Length. Starlette
    # caches the body, so Pydantic can still parse it afterward.
    if len(await request.body()) > ONBOARDING_SPEECH_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Speech request is too large.")


def _voice_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GnaniConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, GnaniProviderError):
        return HTTPException(status_code=502, detail=str(exc))
    return HTTPException(status_code=500, detail="Voice processing failed.")


@router.post("/onboarding-speech")
def onboarding_speech(
    request: OnboardingSpeechRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    _size_check: None = Depends(validate_onboarding_speech_size),
):
    """Speech-only tour synthesis; it intentionally never enters document retrieval."""
    try:
        audio, content_type = _synthesize_for_user(db, current_user, request.text, request.language)
        return Response(content=audio, media_type=content_type)
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc


@router.post("/transcribe", response_model=VoiceTranscriptionResponse)
async def transcribe(
    audio: UploadFile = File(...),
    speech_input_language: str = Form(default="auto"),
    current_user: User = Depends(get_current_user),
):
    del current_user
    content = await audio.read()
    try:
        # Retain the legacy form field for API compatibility, but never let a
        # previous turn or the text-chat selector lock REST STT to one language.
        if speech_input_language != "auto":
            logger.info("voice_transcription_language_override_ignored requested=%s mode=auto", speech_input_language)
        return transcribe_audio(
            content,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type,
            language="auto",
        )
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc
@router.post("/sessions")
def create_voice_session(
    request: VoiceSessionCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info(
        "voice_session_creation_start user_id=%s text_response_language=%s voice_response_language=auto",
        current_user.id, request.response_language,
    )
    active = get_active_document(current_user.id)
    resolve_authorized_api_key(
        user=current_user,
        access_password=request.access_password,
        user_openai_api_key=request.user_openai_api_key,
    )
    conversation = create_conversation(
        db,
        current_user,
        document=active or {"active_doc_id": NO_DOCUMENT_ID, "active_filename": NO_DOCUMENT_NAME},
        mode="voice",
        # Input language is auto-detected per turn, while output language is
        # persisted from the user's voice-session selection.
        response_language=(request.response_language if request.response_language != "auto" else "en"),
        speech_input_language=request.speech_input_language,
        speech_output_language="auto",
    )
    serialized = serialize_conversation(conversation)
    logger.info("voice_session_creation_complete user_id=%s conversation_id=%s document_id=%s", current_user.id, conversation.id, conversation.document_id)
    serialized["response_emotion"] = normalize_response_emotion(request.response_emotion)
    return serialized


@router.get("/conversations")
def list_voice_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.models.conversation_models import Conversation

    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return {"conversations": [serialize_conversation(item) for item in conversations]}


@router.delete("/conversations/{conversation_id}")
def delete_voice_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = get_owned_conversation(db, current_user, conversation_id)
    from app.models.conversation_models import ConversationTurn

    db.query(ConversationTurn).filter(ConversationTurn.conversation_id == conversation.id).delete()
    db.delete(conversation)
    db.commit()
    return {"status": "success", "message": "Conversation deleted."}


@router.post("/sessions/{conversation_id}/synthesize")
def synthesize(
    conversation_id: str,
    request: VoiceSynthesisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    conversation = get_owned_conversation(db, current_user, conversation_id)
    try:
        audio, content_type = _synthesize_for_user(db, current_user, request.text, request.speech_output_language)
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc
    return Response(content=audio, media_type=content_type)


@router.post("/synthesize-answer")
def synthesize_answer(
    request: VoiceSynthesisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Synthesize one already-generated answer on explicit user request."""
    try:
        audio, content_type = _synthesize_for_user(
            db,
            current_user,
            request.text,
            request.speech_output_language,
            request.voice_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc
    return Response(content=audio, media_type=content_type)


@router.post("/sessions/{conversation_id}/turn")
async def voice_turn(
    conversation_id: str,
    audio: UploadFile = File(...),
    access_password: str = Form(default=""),
    user_openai_api_key: str = Form(default=""),
    query_language: str | None = Form(default=None),
    response_length: Literal["short", "medium", "large"] = Form(default="medium"),
    response_emotion: str = Form(default="AUTO"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info("voice_turn_start user_id=%s conversation_id=%s", current_user.id, conversation_id)
    conversation = get_owned_conversation(db, current_user, conversation_id)
    active_document = get_active_document(current_user.id)
    validate_conversation_document(conversation, active_document)
    if active_document and active_document.get("source_path"):
        available, reason = document_artifacts_available(active_document, current_user.id)
        if not available:
            raise HTTPException(status_code=409, detail={
                "code": "DOCUMENT_UNAVAILABLE",
                "message": f"This conversation cannot be answered safely: {reason}",
                "reason": reason,
                "conversation_id": conversation.id,
                "document_id": conversation.document_id,
                "recoverable": True,
            })
    content = await audio.read()
    response_language = conversation.response_language if conversation.response_language not in {None, "auto"} else "en"
    try:
        stt_started = time.perf_counter()
        transcription = await asyncio.to_thread(
            transcribe_audio,
            content,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type,
            # Auto-detect on every REST turn so a session can switch
            # Kannada -> Telugu -> English without restarting.
            language="auto",
        )
        stt_latency_ms = round((time.perf_counter() - stt_started) * 1000, 1)
        logger.info("voice_transcript_received user_id=%s conversation_id=%s", current_user.id, conversation_id)
        detection = detect_voice_language(
            transcription["transcript"],
            conversation.speech_input_language,
            provider_language=transcription.get("detected_language_code"),
        )
        detected_query_language = detection["language"]
        conversation.query_language = detected_query_language
        conversation.speech_input_language = detected_query_language
        conversation.speech_output_language = response_language
        db.commit()
        logger.info(
            "voice_turn_language_diagnostics session_id=%s input_language=%s detection_confidence=%s detection_source=%s text_response_language=%s voice_response_language=%s tts_language=%s voice_id=%s stt_latency_ms=%s",
            conversation_id, detected_query_language, detection.get("confidence"), detection.get("source"),
            "not_used_by_voice", response_language, response_language,
            _saved_voice(db, current_user, response_language), stt_latency_ms,
        )
        context = recent_context(db, conversation)
        request = QueryRequest(
            question=transcription["transcript"],
            access_password=access_password,
            user_openai_api_key=user_openai_api_key,
            query_language=detected_query_language,
            response_language=response_language,
            response_length=response_length,
            conversation_id=conversation.id,
            conversation_context=context or None,
            use_semantic_intent=True,
            selected_emotion=normalize_response_emotion(response_emotion),
        )
        rag_started = time.perf_counter()
        result = await asyncio.to_thread(run_document_query, request, current_user, endpoint="/api/voice")
        rag_latency_ms = round((time.perf_counter() - rag_started) * 1000, 1)
        result.update({
            "voice_input_language": detected_query_language,
            "voice_response_language": response_language,
            "voice_language_confidence": detection.get("confidence"),
        })
        logger.info("voice_rag_complete user_id=%s conversation_id=%s status=%s", current_user.id, conversation_id, result.get("status"))
        add_turn(db, conversation, transcript=transcription["transcript"], answer=None, evidence=[], provider_request_id=transcription.get("request_id"), role="user")
        answer = result.get("answer")
        if answer and result.get("status") in {"success", "review_recommended"}:
            request_id = transcription.get("request_id")
            add_turn(db, conversation, transcript=None, answer=answer, evidence=result.get("evidence", []), evaluation_id=(result.get("evaluation") or {}).get("evaluation_id"), provider_request_id=f"{request_id}:assistant" if request_id else None, role="assistant")
            voice_id = _saved_voice(db, current_user, response_language)
            tts_started = time.perf_counter()
            audio_bytes, content_type = await asyncio.to_thread(synthesize_speech, answer, language=response_language, emotion=normalize_response_emotion(response_emotion), **({"voice": voice_id} if voice_id else {}))
            tts_latency_ms = round((time.perf_counter() - tts_started) * 1000, 1)
            logger.info("voice_turn_latency_diagnostics session_id=%s stt_latency_ms=%s rag_latency_ms=%s tts_latency_ms=%s total_latency_ms=%s", conversation_id, stt_latency_ms, rag_latency_ms, tts_latency_ms, round((time.perf_counter() - stt_started) * 1000, 1))
            logger.info("voice_tts_complete user_id=%s conversation_id=%s content_type=%s", current_user.id, conversation_id, content_type)
            result["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
            result["audio_content_type"] = content_type
        elif result.get("status") == "document_required" and result.get("message"):
            safe_message = result["message"]
            request_id = transcription.get("request_id")
            add_turn(
                db,
                conversation,
                transcript=None,
                answer=safe_message,
                evidence=[],
                provider_request_id=f"{request_id}:assistant" if request_id else None,
                role="assistant",
            )
            voice_id = _saved_voice(db, current_user, response_language)
            tts_started = time.perf_counter()
            audio_bytes, content_type = await asyncio.to_thread(synthesize_speech, safe_message, language=response_language, emotion=normalize_response_emotion(response_emotion), **({"voice": voice_id} if voice_id else {}))
            logger.info("voice_turn_latency_diagnostics session_id=%s stt_latency_ms=%s rag_latency_ms=%s tts_latency_ms=%s total_latency_ms=%s", conversation_id, stt_latency_ms, rag_latency_ms, round((time.perf_counter() - tts_started) * 1000, 1), round((time.perf_counter() - stt_started) * 1000, 1))
            result["audio_base64"] = base64.b64encode(audio_bytes).decode("ascii")
            result["audio_content_type"] = content_type
        result["transcript"] = transcription["transcript"]
        result["detected_language_code"] = transcription.get("detected_language_code")
        result["conversation_id"] = conversation.id
        return result
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc
    except Exception:
        logger.exception("voice_turn_failed user_id=%s conversation_id=%s", current_user.id, conversation_id)
        raise


def _websocket_user(websocket: WebSocket) -> User | None:
    db = SessionLocal()
    try:
        if auth_dependencies.LOCAL_DEV_AUTH_BYPASS:
            user = auth_dependencies.get_local_development_user(db)
            db.expunge(user)
            return user

        token = websocket.cookies.get("access_token")
        if not token:
            return None
        try:
            user_id, token_version = decode_access_token(token)
        except Exception:
            return None
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active or user.token_version != token_version:
            return None
        db.expunge(user)
        return user
    finally:
        db.close()


@router.websocket("/sessions/{conversation_id}/stream")
async def voice_stream(websocket: WebSocket, conversation_id: str):
    user = _websocket_user(websocket)
    if user is None:
        logger.warning("voice_websocket_auth_rejected conversation_id=%s", conversation_id)
        await websocket.close(code=1008)
        return
    await websocket.accept()
    logger.info("voice_websocket_open user_id=%s conversation_id=%s", user.id, conversation_id)
    provider = None
    try:
        first = await websocket.receive_json()
        if first.get("type") != "auth":
            await websocket.send_json({"type": "error", "message": "Voice session authorization is required."})
            await websocket.close(code=1008)
            return
        db = SessionLocal()
        saved_response_emotion = "AUTO"
        try:
            conversation = get_owned_conversation(db, user, conversation_id)
            saved_response_emotion = _saved_response_emotion(db, user)
            active_document = get_active_document(user.id)
            validate_conversation_document(conversation, active_document)
            api_key = (
                resolve_authorized_api_key(
                    user=user,
                    access_password=first.get("access_password", ""),
                    user_openai_api_key=first.get("user_openai_api_key", ""),
                )
                if active_document
                else None
            )
        finally:
            db.close()
        del api_key
        response_length = first.get("response_length", "medium")
        response_emotion = normalize_response_emotion(first.get("response_emotion") or saved_response_emotion)
        if response_length not in {"short", "medium", "large"}:
            await websocket.send_json({"type": "error", "message": "response_length must be short, medium, or large."})
            await websocket.close(code=1008)
            return
        import websockets

        provider = await websockets.connect(
            __import__("app.core.config", fromlist=["GNANI_STT_WS_URL"]).GNANI_STT_WS_URL,
            # Keep the provider stream in automatic mode. Each completed
            # utterance is independently classified before RAG and TTS.
            additional_headers=prisma_websocket_headers("auto"),
        )
        logger.info("voice_prisma_websocket_connected user_id=%s conversation_id=%s document_id=%s", user.id, conversation_id, conversation.document_id)
        await websocket.send_json({"type": "listening", "provider": "prisma"})

        async def forward_audio():
            nonlocal response_emotion
            first_audio = True
            while True:
                message = await websocket.receive()
                if message.get("bytes") is not None:
                    if first_audio:
                        logger.info("voice_first_pcm_frame user_id=%s conversation_id=%s bytes=%s", user.id, conversation_id, len(message["bytes"]))
                        first_audio = False
                    await provider.send(message["bytes"])
                elif message.get("text"):
                    control = json.loads(message["text"])
                    if control.get("type") == "end":
                        return
                    if control.get("type") == "interrupt":
                        await cancel_active_turn()
                        continue
                    if control.get("type") == "config":
                        response_emotion = normalize_response_emotion(control.get("response_emotion"))
                        await websocket.send_json({"type": "voice_configured", "response_emotion": response_emotion})
                        continue
                    if control.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

        turn_lock = asyncio.Lock()
        processed_request_ids: set[str] = set()
        active_turn_task: asyncio.Task | None = None
        response_generation = 0

        async def cancel_active_turn() -> None:
            nonlocal active_turn_task, response_generation
            response_generation += 1
            task = active_turn_task
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            active_turn_task = None
            with suppress(Exception):
                await websocket.send_json({"type": "interrupted", "generation": response_generation})

        async def consume_provider():
            nonlocal active_turn_task
            async for message in provider:
                event = parse_provider_message(message)
                detected_provider_language = provider_detected_language(event)
                await websocket.send_json(event)
                if event.get("type") == "transcript" and event.get("text"):
                    # Prisma can emit interim transcript events. They remain
                    # visible to the client but must not start or cancel a
                    # grounded turn until the utterance is final.
                    if not _provider_transcript_is_final(event):
                        continue
                    request_id = event.get("request_id")
                    if request_id and request_id in processed_request_ids:
                        logger.warning("voice_duplicate_transcript_ignored user_id=%s conversation_id=%s request_id=%s", user.id, conversation_id, request_id)
                        continue
                    if request_id:
                        processed_request_ids.add(request_id)
                    async with turn_lock:
                        await cancel_active_turn()
                        await websocket.send_json({"type": "processing", "text": event["text"]})
                        active_turn_task = asyncio.create_task(
                            _process_stream_turn(
                                websocket, user, conversation_id, event["text"], first, request_id,
                                generation=response_generation,
                                is_current=lambda generation=response_generation: generation == response_generation,
                                provider_language=detected_provider_language,
                                response_length=response_length,
                                response_emotion=response_emotion,
                            ),
                            name=f"voice-turn-{conversation_id}",
                        )
                        try:
                            await active_turn_task
                        except asyncio.CancelledError:
                            if active_turn_task is not None and not active_turn_task.cancelled():
                                raise
                        finally:
                            active_turn_task = None

        tasks = {
            asyncio.create_task(forward_audio(), name=f"voice-forward-{conversation_id}"),
            asyncio.create_task(consume_provider(), name=f"voice-consume-{conversation_id}"),
        }
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        try:
            for task in done:
                task.result()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except (GnaniConfigurationError, GnaniProviderError, HTTPException) as exc:
        with suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(exc.detail if isinstance(exc, HTTPException) else exc)})
    except Exception:
        logger.exception("voice_websocket_failed user_id=%s conversation_id=%s", user.id if user else None, conversation_id)
        with suppress(Exception):
            await websocket.send_json({"type": "error", "message": "Voice session failed."})
    finally:
        logger.info("voice_websocket_close user_id=%s conversation_id=%s", user.id, conversation_id)
        if 'active_turn_task' in locals() and active_turn_task is not None and not active_turn_task.done():
            active_turn_task.cancel()
            with suppress(Exception):
                await active_turn_task
        if provider is not None:
            with suppress(Exception):
                await provider.close()
        with suppress(Exception):
            await websocket.close()


async def _send_stream_tts(websocket, text, *, language, user_id, conversation_id, voice_id=None, emotion=None, generation=0, is_current=None):
    started = asyncio.get_running_loop().time()
    logger.info("tts_request_started user_id=%s conversation_id=%s language=%s", user_id, conversation_id, language)
    streamed = False
    first_audio_started = None
    try:
        tts_kwargs = {"voice": voice_id} if voice_id else {}
        if emotion:
            tts_kwargs["emotion"] = emotion
        async for chunk in stream_speech(text, language=language, **tts_kwargs):
            if is_current is not None and not is_current():
                return False
            audio = chunk.get("audio") or b""
            if not audio:
                raise GnaniProviderError("Gnani Timbre returned an empty audio chunk.")
            if len(audio) % 2:
                raise GnaniProviderError("Gnani Timbre returned invalid 16-bit PCM audio.")
            if first_audio_started is None:
                first_audio_started = asyncio.get_running_loop().time()
                logger.info(
                    "voice_tts_first_audio user_id=%s conversation_id=%s time_to_first_audio_ms=%s mode=stream",
                    user_id,
                    conversation_id,
                    round((first_audio_started - started) * 1000, 1),
                )
            logger.info("audio_payload_validated user_id=%s conversation_id=%s bytes=%s mime=audio/pcm sample_rate=%s channels=%s codec=%s", user_id, conversation_id, len(audio), chunk.get("sample_rate"), chunk.get("channels"), chunk.get("encoding"))
            logger.info("audio_decoding_completed user_id=%s conversation_id=%s bytes=%s", user_id, conversation_id, len(audio))
            streamed = True
            logger.info(
                "tts_audio_received user_id=%s conversation_id=%s bytes=%s sample_rate=%s channels=%s codec=%s sequence=%s",
                user_id, conversation_id, len(audio), chunk.get("sample_rate"), chunk.get("channels"), chunk.get("encoding"), chunk.get("sequence"),
            )
            await websocket.send_json({
                "type": "audio_chunk",
                "generation": generation,
                "sequence": chunk["sequence"],
                "sample_rate": chunk["sample_rate"],
                "channels": chunk["channels"],
                "sample_width": chunk["sample_width"],
                "encoding": chunk["encoding"],
                "audio_base64": base64.b64encode(audio).decode("ascii"),
                "is_final": chunk["is_final"],
            })
        if streamed:
            await websocket.send_json({"type": "audio_complete", "generation": generation})
            logger.info("tts_request_completed user_id=%s conversation_id=%s mode=stream duration_ms=%s", user_id, conversation_id, round((asyncio.get_running_loop().time() - started) * 1000))
            return True
    except asyncio.CancelledError:
        raise


    except GnaniProviderError as exc:
        logger.warning("tts_request_failed user_id=%s conversation_id=%s mode=stream error=%s duration_ms=%s", user_id, conversation_id, str(exc), round((asyncio.get_running_loop().time() - started) * 1000))

    try:
        audio, content_type = await asyncio.to_thread(synthesize_speech, text, language=language, **({"voice": voice_id} if voice_id else {}))
        if is_current is not None and not is_current():
            return False
        if not audio:
            raise GnaniProviderError("Gnani Timbre returned an empty buffered audio response.")
        logger.info("tts_audio_received user_id=%s conversation_id=%s bytes=%s mime=%s mode=buffered", user_id, conversation_id, len(audio), content_type)
        logger.info(
            "voice_tts_first_audio user_id=%s conversation_id=%s time_to_first_audio_ms=%s mode=buffered",
            user_id,
            conversation_id,
            round((asyncio.get_running_loop().time() - started) * 1000, 1),
        )
        await websocket.send_json({"type": "audio", "generation": generation, "content_type": content_type, "audio_base64": base64.b64encode(audio).decode("ascii"), "is_final": True})
        logger.info("tts_request_completed user_id=%s conversation_id=%s mode=buffered duration_ms=%s", user_id, conversation_id, round((asyncio.get_running_loop().time() - started) * 1000))
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("tts_request_failed user_id=%s conversation_id=%s mode=buffered error=%s duration_ms=%s", user_id, conversation_id, str(exc), round((asyncio.get_running_loop().time() - started) * 1000))
        raise


def _provider_transcript_is_final(event: dict) -> bool:
    """Return whether a provider transcript event is actionable."""
    candidates = [event]
    if isinstance(event.get("data"), dict):
        candidates.append(event["data"])
    for candidate in candidates:
        for key in ("is_final", "final", "final_transcript", "utterance_final"):
            if key in candidate:
                return bool(candidate[key])
    event_type = str(event.get("event") or event.get("status") or event.get("type") or "").lower()
    if event_type in {"partial", "interim", "transcript_partial", "speech_partial"}:
        return False
    return True


async def _process_stream_turn(websocket, user, conversation_id, transcript, auth, provider_request_id, *, generation=0, is_current=None, provider_language=None, response_length="medium", response_emotion="AUTO"):
    db = SessionLocal()
    conversation = None
    response_language = "en"
    try:
        conversation = get_owned_conversation(db, user, conversation_id)
        active_document = get_active_document(user.id)
        validate_conversation_document(conversation, active_document)
        if active_document and active_document.get("source_path"):
            available, reason = document_artifacts_available(active_document, user.id)
            if not available:
                await websocket.send_json({"type": "error", "message": f"This conversation cannot be answered safely: {reason}"})
                return
        response_language = conversation.response_language if conversation.response_language not in {None, "auto"} else "en"
        try:
            detection = detect_voice_language(
                transcript,
                getattr(conversation, "speech_input_language", None),
                provider_language=provider_language,
            )
            detected_query_language = detection["language"]
        except ValueError as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            return
        conversation.query_language = detected_query_language
        conversation.speech_input_language = detected_query_language
        conversation.speech_output_language = response_language
        db.commit()
        logger.info(
            "voice_turn_language_diagnostics session_id=%s input_language=%s detection_confidence=%s detection_source=%s text_response_language=%s voice_response_language=%s tts_language=%s voice_id=%s",
            conversation_id, detected_query_language, detection.get("confidence"), detection.get("source"),
            "not_used_by_voice", response_language, response_language, _saved_voice(db, user, response_language),
        )
        context = recent_context(db, conversation)
        request = QueryRequest(
            question=transcript,
            access_password=auth.get("access_password", ""),
            user_openai_api_key=auth.get("user_openai_api_key", ""),
            query_language=detected_query_language,
            response_language=response_language,
            response_length=response_length,
            conversation_id=conversation.id,
            conversation_context=context or None,
            use_semantic_intent=True,
            selected_emotion=normalize_response_emotion(response_emotion),
        )
        rag_started = time.perf_counter()
        voice_settings = get_settings()
        threshold = float(voice_settings.get("progress_threshold_seconds", 2.0))
        progress_task = None

        async def delayed_progress():
            await asyncio.sleep(threshold)
            if is_current is not None and not is_current():
                return
            message = progress_announcement(response_language, asyncio.get_running_loop().time() - rag_started, settings=voice_settings)
            if not message:
                return
            # This is a real provider-backed progress utterance. It uses the
            # same generation and streaming path as the final answer, so a
            # barge-in or newer turn invalidates it before any audio is sent.
            await websocket.send_json({"type": "processing_progress", "generation": generation, "text": message})
            try:
                await _send_stream_tts(
                    websocket, message, language=response_language, user_id=user.id,
                    conversation_id=conversation_id, voice_id=_saved_voice(db, user, response_language),
                    emotion=normalize_response_emotion(response_emotion), generation=generation, is_current=is_current,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("voice_progress_announcement_failed user_id=%s conversation_id=%s", user.id, conversation_id, exc_info=True)

        if voice_settings.get("progress_enabled") and voice_settings.get("enabled"):
            progress_task = asyncio.create_task(delayed_progress(), name=f"voice-progress-{conversation_id}")
        try:
            result = await asyncio.to_thread(run_document_query, request, user, endpoint="/api/voice")
        finally:
            if progress_task is not None and not progress_task.done():
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)
            elif progress_task is not None:
                await progress_task
        rag_latency_ms = round((time.perf_counter() - rag_started) * 1000, 1)
        result.update({
            "voice_input_language": detected_query_language,
            "voice_response_language": response_language,
            "voice_language_confidence": detection.get("confidence"),
        })
        logger.info(
            "voice_turn_latency_diagnostics session_id=%s stt_latency_ms=provider_event rag_latency_ms=%s tts_latency_ms=logged_by_tts total_latency_ms=%s",
            conversation_id, rag_latency_ms, rag_latency_ms,
        )
        logger.info("voice_rag_complete user_id=%s conversation_id=%s status=%s", user.id, conversation_id, result.get("status"))
        add_turn(db, conversation, transcript=transcript, answer=None, evidence=[], provider_request_id=provider_request_id, role="user")
        answer = result.get("answer")
        if answer and result.get("status") in {"success", "review_recommended"}:
            logger.info("assistant_response_created user_id=%s conversation_id=%s chars=%s status=%s", user.id, conversation_id, len(answer), result.get("status"))
            add_turn(db, conversation, transcript=None, answer=answer, evidence=result.get("evidence", []), evaluation_id=(result.get("evaluation") or {}).get("evaluation_id"), provider_request_id=f"{provider_request_id}:assistant" if provider_request_id else None, role="assistant")
            await websocket.send_json({"type": "speaking", "generation": generation, "answer": answer, "evidence": result.get("evidence", []), "evaluation": result.get("evaluation"), "voice_input_language": detected_query_language, "voice_response_language": response_language, "voice_language_confidence": detection.get("confidence"), "conversation_intent": result.get("question_type"), "detected_emotion": result.get("detected_emotion"), "emotion_confidence": result.get("emotion_confidence"), "response_emotion": result.get("response_emotion"), "response_tone": result.get("response_tone"), "tts_emotion_applied": bool(normalize_response_emotion(response_emotion) in SUPPORTED_TTS_EMOTIONS)})
            await _send_stream_tts(websocket, answer, language=response_language, user_id=user.id, conversation_id=conversation_id, voice_id=_saved_voice(db, user, response_language), emotion=normalize_response_emotion(response_emotion), generation=generation, is_current=is_current)
        else:
            safe_message = result.get("message") or "I could not find a grounded answer in the uploaded PDF."
            logger.info("assistant_response_created user_id=%s conversation_id=%s chars=%s status=%s", user.id, conversation_id, len(safe_message), result.get("status"))
            add_turn(db, conversation, transcript=None, answer=safe_message, evidence=[], provider_request_id=f"{provider_request_id}:assistant" if provider_request_id else None, role="assistant")
            await websocket.send_json({"type": "answer", "generation": generation, "result": result})
            if result.get("status") in {"review_recommended", "human_review_required", "out_of_scope", "error"}:
                await _send_stream_tts(websocket, safe_message, language=response_language, user_id=user.id, conversation_id=conversation_id, voice_id=_saved_voice(db, user, response_language), emotion=normalize_response_emotion(response_emotion), generation=generation, is_current=is_current)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        message = localized_voice_message(response_language, "safe_refusal")
        logger.info("assistant_response_created user_id=%s conversation_id=%s kind=policy_message", user.id, conversation_id)
        with suppress(Exception):
            await websocket.send_json({"type": "answer", "result": {"status": "security_blocked", "answer": None, "message": message, "question_type": detail.get("question_type", "SECURITY_BLOCKED")}})
            if conversation is not None:
                await _send_stream_tts(websocket, message, language=response_language, user_id=user.id, conversation_id=conversation_id, voice_id=_saved_voice(db, user, response_language) if conversation is not None else None, generation=generation, is_current=is_current)
    except Exception:
        logger.exception("voice_stream_turn_failed user_id=%s conversation_id=%s", user.id, conversation_id)
        raise
    finally:
        db.close()
