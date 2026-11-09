from typing import Literal, Optional

from pydantic import BaseModel, Field


ResponseLanguage = Literal[
    "kn", "en", "hi", "ta", "te", "ml", "mr", "gu", "bn", "pa", "ur"
]


class DocumentLanguageRequest(BaseModel):
    response_language: ResponseLanguage
    access_password: Optional[str] = ""
    user_openai_api_key: Optional[str] = ""


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


class DocumentSummaryResponse(BaseModel):
    status: str
    active_document: Optional[str] = None
    response_language: str
    short_summary: str = ""
    detailed_summary: str = ""
    evidence: list[EvidenceReference] = Field(default_factory=list)
    main_conclusions: list[DocumentConclusion] = Field(default_factory=list)
    topics: list[DocumentTopic] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)
    message: str


class DocumentTopicsResponse(BaseModel):
    status: str
    active_document: Optional[str] = None
    response_language: str
    topics: list[DocumentTopic] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)
    message: str


class DocumentQuestionsResponse(BaseModel):
    status: str
    active_document: Optional[str] = None
    response_language: str
    questions: list[GeneratedDocumentQuestion] = Field(default_factory=list)
    unsupported_items: list[str] = Field(default_factory=list)
    message: str


class DocumentLanguageResponse(BaseModel):
    status: str
    active_document: Optional[str] = None
    input_language: Optional[str] = None
    evidence_language: Optional[str] = None
    response_language: str
    message: str
