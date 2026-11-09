from pydantic import BaseModel, Field, field_validator

from app.services.gnani_prisma_service import SUPPORTED_GNANI_SPEECH_LANGUAGES


class VoiceSessionCreateRequest(BaseModel):
    response_language: str = "en"
    speech_input_language: str = "auto"
    speech_output_language: str | None = None
    access_password: str = ""
    user_openai_api_key: str = ""


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


class VoiceTranscriptionResponse(BaseModel):
    transcript: str
    detected_language_code: str | None = None
    request_id: str | None = None
    model: str | None = None
