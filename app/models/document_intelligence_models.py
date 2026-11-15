from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.services.language_registry import validate_language


ResponseLanguage = str


class DocumentLanguageRequest(BaseModel):
    response_language: ResponseLanguage
    response_length: Literal["short", "medium", "large"] = "medium"
    conversation_id: Optional[str] = None
    document_id: Optional[str] = None
    document_language: Optional[str] = None
    access_password: Optional[str] = ""
    user_openai_api_key: Optional[str] = ""

    @field_validator("response_language")
    @classmethod
    def validate_response_language(cls, value: str) -> str:
        return validate_language(value)


class EvidenceReference(BaseModel):
    page_number: int
    chunk_id: int
    text: str


class DocumentTopic(BaseModel):
    title: str
    explanation: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class DocumentConclusion(BaseModel):
    text: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class GeneratedDocumentQuestion(BaseModel):
    question: str
    question_type: str
    difficulty: str
    suggested_answer: Optional[str] = None
    evidence: list[EvidenceReference] = Field(default_factory=list)
    document_id: Optional[str] = None
    document_language: Optional[str] = None
    supporting_chunk_ids: list[int] = Field(default_factory=list)
    supporting_pages: list[int] = Field(default_factory=list)
    answerability: str = "validated"


class DocumentSummaryResponse(BaseModel):
    status: str
    conversation_id: Optional[str] = None
    active_document: Optional[str] = None
    response_language: str
    response_length: Literal["short", "medium", "large"] = "medium"
    short_summary: str = ""
    detailed_summary: str = ""
    evidence: list[EvidenceReference] = Field(default_factory=list)
    main_conclusions: list[DocumentConclusion] = Field(default_factory=list)
    topics: list[DocumentTopic] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)
    message: str


class DocumentTopicsResponse(BaseModel):
    status: str
    conversation_id: Optional[str] = None
    active_document: Optional[str] = None
    response_language: str
    response_length: Literal["short", "medium", "large"] = "medium"
    topics: list[DocumentTopic] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)
    message: str


class DocumentQuestionsResponse(BaseModel):
    status: str
    conversation_id: Optional[str] = None
    active_document: Optional[str] = None
    response_language: str
    response_length: Literal["short", "medium", "large"] = "medium"
    questions: list[GeneratedDocumentQuestion] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)
    message: str
    validation_diagnostics: dict[str, int] = Field(default_factory=dict)


class DocumentLanguageResponse(BaseModel):
    status: str
    active_document: Optional[str] = None
    input_language: Optional[str] = None
    evidence_language: Optional[str] = None
    response_language: str
    message: str
