from pydantic import BaseModel, Field
from typing import List, Optional

from app.models.evaluation_models import (
    EvaluationDecision,
    EvaluationResult,
    FeedbackCategory,
)


class QueryRequest(BaseModel):
    question: str
    access_password: Optional[str] = ""
    user_openai_api_key: Optional[str] = ""
    query_language: Optional[str] = None
    response_language: Optional[str] = None
    conversation_id: Optional[str] = None
    conversation_context: Optional[str] = None
    turn_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: int
    file_name: str
    page_number: int
    text: str
    score: float

    # Document and paragraph metadata
    doc_id: Optional[str] = None
    source: Optional[str] = None
    page_chunk_id: Optional[int] = None
    paragraph_ids: Optional[List[int]] = None

    # Character offsets in the original extracted page text
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Language/script metadata
    page_language: Optional[str] = None
    page_language_confidence: Optional[float] = None
    language: Optional[str] = None
    language_confidence: Optional[float] = None


class HumanReviewRequest(BaseModel):
    decision: EvaluationDecision
    feedback_category: Optional[FeedbackCategory] = None
    feedback_text: Optional[str] = None
    corrected_answer: Optional[str] = None


class AnswerEvaluationRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    answer_id: int = Field(gt=0)
    access_password: Optional[str] = ""
    user_openai_api_key: Optional[str] = ""


class AnswerFeedbackRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=64)
    answer_id: int = Field(gt=0)
    helpful: bool


class QueryResponse(BaseModel):
    status: str
    active_document: Optional[str]
    question: str
    question_type: str
    answer: Optional[str]
    confidence: Optional[str]
    best_score: Optional[float]
    average_score: Optional[float]
    evidence: list
    retrieved_chunks: List[RetrievedChunk]
    evaluation: Optional[EvaluationResult] = None
    dense_candidate_count: Optional[int] = None
    sparse_candidate_count: Optional[int] = None
    fused_candidate_count: Optional[int] = None
    reranked_candidate_count: Optional[int] = None
    retrieval_stage_timings: Optional[dict] = None
    reranker_enabled: Optional[bool] = None
    reranker_status: Optional[str] = None
    message: str
    conversation_id: Optional[str] = None
    answer_id: Optional[int] = None
