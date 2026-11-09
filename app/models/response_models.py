from pydantic import BaseModel
from typing import Optional


class UploadPDFResponse(BaseModel):
    status: str
    filename: str
    saved_filename: str
    file_size_mb: float
    total_pages: int
    extracted_characters: int
    num_chunks: int
    selected_chunk_size: int
    selected_chunk_overlap: int
    preview_chunk: Optional[str] = None
    input_language: Optional[str] = None
    evidence_language: Optional[str] = None
    response_language: Optional[str] = None
    message: Optional[str] = None
    job_id: Optional[str] = None


class ErrorResponse(BaseModel):
    status: str
    message: str
