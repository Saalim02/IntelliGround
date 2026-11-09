from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, get_db
from app.models.auth_models import User
from app.models.query_models import QueryRequest
from app.models.voice_models import OnboardingSpeechRequest, VoiceSessionCreateRequest, VoiceSynthesisRequest, VoiceTranscriptionResponse
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
from app.services.document_registry_service import get_active_document
from app.services.gnani_prisma_service import (
    GnaniConfigurationError,
    GnaniProviderError,
    parse_provider_message,
    provider_detected_language,
    prisma_websocket_headers,
    transcribe_audio,
)
from app.services.gnani_timbre_service import stream_speech, synthesize_speech
from app.services.rag_orchestration_service import run_document_query
from app.services.language_detection_service import detect_voice_response_language
from app.services.security import auth_dependencies
from app.services.security.auth_dependencies import get_current_user
from app.services.security.auth_service import decode_access_token
from app.services.security.rag_authorization import resolve_authorized_api_key
from app.services.voice_response_templates import localized_voice_message

router = APIRouter(prefix="/voice", tags=["Voice"])
logger = logging.getLogger(__name__)
ONBOARDING_SPEECH_MAX_BODY_BYTES = 8 * 1024


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
    _size_check: None = Depends(validate_onboarding_speech_size),
):
    """Speech-only tour synthesis; it intentionally never enters document retrieval."""
    del current_user
    try:
        audio, content_type = synthesize_speech(request.text, language=request.language)
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
        return transcribe_audio(
            content,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type,
            language=speech_input_language,
        )
    except (GnaniConfigurationError, GnaniProviderError) as exc:
        raise _voice_error(exc) from exc
@router.post("/sessions")
def create_voice_session(
    request: VoiceSessionCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info("voice_session_creation_start user_id=%s response_language=%s", current_user.id, request.response_language)
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
        response_language=request.response_language,
        speech_input_language=request.speech_input_language,
        speech_output_language=request.speech_output_language or request.response_language,
    )
    serialized = serialize_conversation(conversation)
    logger.info("voice_session_creation_complete user_id=%s conversation_id=%s document_id=%s", current_user.id, conversation.id, conversation.document_id)
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
        audio, content_type = synthesize_speech(
            request.text,
            language=request.speech_output_language,
        )
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info("voice_turn_start user_id=%s conversation_id=%s", current_user.id, conversation_id)
    conversation = get_owned_conversation(db, current_user, conversation_id)
    validate_conversation_document(conversation, get_active_document(current_user.id))
    content = await audio.read()
    voice_language = conversation.speech_output_language or conversation.response_language or "en"
    try:
        transcription = await asyncio.to_thread(
            transcribe_audio,
            content,
            filename=audio.filename or "audio.wav",
            content_type=audio.content_type,
            language=conversation.speech_input_language,
        )
        logger.info("voice_transcript_received user_id=%s conversation_id=%s", current_user.id, conversation_id)
        voice_language = detect_voice_response_language(
            transcription["transcript"],
            voice_language,
            provider_language=transcription.get("detected_language_code"),
        )
        conversation.query_language = voice_language
        conversation.speech_input_language = voice_language
        conversation.speech_output_language = voice_language
        conversation.response_language = voice_language
        db.commit()
        context = recent_context(db, conversation)
        request = QueryRequest(
            question=transcription["transcript"],
            access_password=access_password,
            user_openai_api_key=user_openai_api_key,
            query_language=voice_language,
            response_language=voice_language,
            conversation_id=conversation.id,
            conversation_context=context or None,
        )
        result = await asyncio.to_thread(run_document_query, request, current_user, endpoint="/api/voice")
        logger.info("voice_rag_complete user_id=%s conversation_id=%s status=%s", current_user.id, conversation_id, result.get("status"))
        add_turn(db, conversation, transcript=transcription["transcript"], answer=None, evidence=[], provider_request_id=transcription.get("request_id"), role="user")
        answer = result.get("answer")
        if answer and result.get("status") in {"success", "review_recommended"}:
            request_id = transcription.get("request_id")
            add_turn(db, conversation, transcript=None, answer=answer, evidence=result.get("evidence", []), evaluation_id=(result.get("evaluation") or {}).get("evaluation_id"), provider_request_id=f"{request_id}:assistant" if request_id else None, role="assistant")
            audio_bytes, content_type = await asyncio.to_thread(synthesize_speech, answer, language=voice_language)
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
            audio_bytes, content_type = await asyncio.to_thread(
                synthesize_speech, safe_message, language=voice_language
            )
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
        try:
            conversation = get_owned_conversation(db, user, conversation_id)
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
        import websockets

        provider = await websockets.connect(
            __import__("app.core.config", fromlist=["GNANI_STT_WS_URL"]).GNANI_STT_WS_URL,
            additional_headers=prisma_websocket_headers(conversation.speech_input_language),
        )
        logger.info("voice_prisma_websocket_connected user_id=%s conversation_id=%s document_id=%s", user.id, conversation_id, conversation.document_id)
        await websocket.send_json({"type": "listening", "provider": "prisma"})

        async def forward_audio():
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


async def _send_stream_tts(websocket, text, *, language, user_id, conversation_id, generation=0, is_current=None):
    started = asyncio.get_running_loop().time()
    logger.info("tts_request_started user_id=%s conversation_id=%s language=%s", user_id, conversation_id, language)
    streamed = False
    try:
        async for chunk in stream_speech(text, language=language):
            if is_current is not None and not is_current():
                return False
            audio = chunk.get("audio") or b""
            if not audio:
                raise GnaniProviderError("Gnani Timbre returned an empty audio chunk.")
            if len(audio) % 2:
                raise GnaniProviderError("Gnani Timbre returned invalid 16-bit PCM audio.")
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
        audio, content_type = await asyncio.to_thread(synthesize_speech, text, language=language)
        if is_current is not None and not is_current():
            return False
        if not audio:
            raise GnaniProviderError("Gnani Timbre returned an empty buffered audio response.")
        logger.info("tts_audio_received user_id=%s conversation_id=%s bytes=%s mime=%s mode=buffered", user_id, conversation_id, len(audio), content_type)
        await websocket.send_json({"type": "audio", "generation": generation, "content_type": content_type, "audio_base64": base64.b64encode(audio).decode("ascii"), "is_final": True})
        logger.info("tts_request_completed user_id=%s conversation_id=%s mode=buffered duration_ms=%s", user_id, conversation_id, round((asyncio.get_running_loop().time() - started) * 1000))
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("tts_request_failed user_id=%s conversation_id=%s mode=buffered error=%s duration_ms=%s", user_id, conversation_id, str(exc), round((asyncio.get_running_loop().time() - started) * 1000))
        raise


async def _process_stream_turn(websocket, user, conversation_id, transcript, auth, provider_request_id, *, generation=0, is_current=None, provider_language=None):
    db = SessionLocal()
    conversation = None
    voice_language = "en"
    try:
        conversation = get_owned_conversation(db, user, conversation_id)
        validate_conversation_document(conversation, get_active_document(user.id))
        previous_language = conversation.speech_output_language or conversation.response_language
        try:
            voice_language = detect_voice_response_language(
                transcript,
                previous_language,
                provider_language=provider_language,
            )
        except ValueError as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            return
        conversation.query_language = voice_language
        conversation.speech_input_language = voice_language
        conversation.speech_output_language = voice_language
        conversation.response_language = voice_language
        db.commit()
        logger.info("voice_language_detected user_id=%s conversation_id=%s input_language=%s response_language=%s", user.id, conversation_id, voice_language, voice_language)
        context = recent_context(db, conversation)
        request = QueryRequest(
            question=transcript,
            access_password=auth.get("access_password", ""),
            user_openai_api_key=auth.get("user_openai_api_key", ""),
            query_language=voice_language,
            response_language=voice_language,
            conversation_id=conversation.id,
            conversation_context=context or None,
        )
        result = await asyncio.to_thread(run_document_query, request, user, endpoint="/api/voice")
        logger.info("voice_rag_complete user_id=%s conversation_id=%s status=%s", user.id, conversation_id, result.get("status"))
        add_turn(db, conversation, transcript=transcript, answer=None, evidence=[], provider_request_id=provider_request_id, role="user")
        answer = result.get("answer")
        if answer and result.get("status") in {"success", "review_recommended"}:
            logger.info("assistant_response_created user_id=%s conversation_id=%s chars=%s status=%s", user.id, conversation_id, len(answer), result.get("status"))
            add_turn(db, conversation, transcript=None, answer=answer, evidence=result.get("evidence", []), evaluation_id=(result.get("evaluation") or {}).get("evaluation_id"), provider_request_id=f"{provider_request_id}:assistant" if provider_request_id else None, role="assistant")
            await websocket.send_json({"type": "speaking", "answer": answer, "evidence": result.get("evidence", []), "evaluation": result.get("evaluation")})
            await _send_stream_tts(websocket, answer, language=voice_language, user_id=user.id, conversation_id=conversation_id, generation=generation, is_current=is_current)
        else:
            safe_message = result.get("message") or "I could not find a grounded answer in the uploaded PDF."
            logger.info("assistant_response_created user_id=%s conversation_id=%s chars=%s status=%s", user.id, conversation_id, len(safe_message), result.get("status"))
            add_turn(db, conversation, transcript=None, answer=safe_message, evidence=[], provider_request_id=f"{provider_request_id}:assistant" if provider_request_id else None, role="assistant")
            await websocket.send_json({"type": "answer", "result": result})
            if result.get("status") in {"review_recommended", "human_review_required", "out_of_scope", "error"}:
                await _send_stream_tts(websocket, safe_message, language=voice_language, user_id=user.id, conversation_id=conversation_id, generation=generation, is_current=is_current)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        message = localized_voice_message(voice_language, "safe_refusal")
        logger.info("assistant_response_created user_id=%s conversation_id=%s kind=policy_message", user.id, conversation_id)
        with suppress(Exception):
            await websocket.send_json({"type": "answer", "result": {"status": "security_blocked", "answer": None, "message": message, "question_type": detail.get("question_type", "SECURITY_BLOCKED")}})
            if conversation is not None:
                await _send_stream_tts(websocket, message, language=voice_language, user_id=user.id, conversation_id=conversation_id, generation=generation, is_current=is_current)
    except Exception:
        logger.exception("voice_stream_turn_failed user_id=%s conversation_id=%s", user.id, conversation_id)
        raise
    finally:
        db.close()
