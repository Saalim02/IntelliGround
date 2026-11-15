from pydantic import BaseModel, Field, field_validator
from typing import Literal

from app.services.gnani_prisma_service import SUPPORTED_GNANI_SPEECH_LANGUAGES


class VoiceSessionCreateRequest(BaseModel):
    response_language: str = "en"
    speech_input_language: str = "auto"
    speech_output_language: str | None = None
    response_length: Literal["short", "medium", "large"] = "medium"
    access_password: str = ""
    user_openai_api_key: str = ""
    response_emotion: Literal["AUTO", "HAPPY", "SAD", "ANGRY", "CALM", "PROFESSIONAL", "FRIENDLY"] = "AUTO"


class OnboardingSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1200)
    language: str = "en"

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in SUPPORTED_GNANI_SPEECH_LANGUAGES:
            raise ValueError("This language does not have verified Gnani speech support.")
        return value


class VoiceSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=12000)
    speech_output_language: str
    voice_id: str | None = Field(default=None, min_length=1, max_length=120)

    @field_validator("speech_output_language")
    @classmethod
    def validate_speech_output_language(cls, value: str) -> str:
        if value not in SUPPORTED_GNANI_SPEECH_LANGUAGES:
            raise ValueError("This language does not have verified Gnani speech support.")
        return value


class VoicePreviewRequest(BaseModel):
    voice_id: str = Field(min_length=1, max_length=120)
    language: str
    text: str = Field(min_length=1, max_length=1200)

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in SUPPORTED_GNANI_SPEECH_LANGUAGES:
            raise ValueError("This language does not have verified Gnani speech support.")
        return value


class VoicePreferenceRequest(BaseModel):
    voice_id: str = Field(min_length=1, max_length=120)
    language: str | None = None
    response_emotion: Literal["AUTO", "HAPPY", "SAD", "ANGRY", "CALM", "PROFESSIONAL", "FRIENDLY"] = "AUTO"

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str | None) -> str | None:
        if value is not None and value not in SUPPORTED_GNANI_SPEECH_LANGUAGES:
            raise ValueError("This language does not have verified Gnani speech support.")
        return value


class VoiceTranscriptionResponse(BaseModel):
    transcript: str
    detected_language_code: str | None = None
    request_id: str | None = None
    model: str | None = None
