"""
Pydantic models for request/response payloads.

Ocean Professional guidance:
- Structure responses with status, data, error, meta
- Keep descriptions crisp and informative
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ApiError(BaseModel):
    code: str = Field(..., description="Machine-friendly error code.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[dict] = Field(
        default=None, description="Optional error details for diagnostics."
    )


class Meta(BaseModel):
    version: str = Field(..., description="API version.")
    service: str = Field(..., description="Service name.")
    extra: Optional[dict] = Field(default=None, description="Additional metadata.")


class DocumentInfo(BaseModel):
    filename: str = Field(..., description="Original PDF filename.")
    file_id: str = Field(..., description="Internal ID for the stored PDF.")
    num_chunks: int = Field(..., description="Number of indexed text chunks.")
    size_bytes: int = Field(..., description="File size in bytes.")


class UploadResponse(BaseModel):
    status: str = Field(..., description="Operation status. 'ok' or 'error'.")
    data: DocumentInfo = Field(..., description="Uploaded document info.")
    error: Optional[ApiError] = Field(default=None, description="Error info if any.")
    meta: Meta = Field(..., description="Response metadata.")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question to ask against PDFs.")
    top_k: Optional[int] = Field(
        default=None, description="Override default number of retrieved passages."
    )
    max_answer_length: Optional[int] = Field(
        default=512, description="Max number of characters in the final answer."
    )


class Passage(BaseModel):
    text: str = Field(..., description="Retrieved passage text.")
    score: Optional[float] = Field(
        default=None, description="Similarity score (higher is better)."
    )
    source: str = Field(..., description="Source reference (file ID or path).")
    chunk_id: int = Field(..., description="Chunk index within the source.")


class QueryAnswer(BaseModel):
    answer: str = Field(..., description="Synthesized natural language answer.")
    passages: List[Passage] = Field(
        default_factory=list, description="Top passages supporting the answer."
    )


class QueryResponse(BaseModel):
    status: str = Field(..., description="Operation status. 'ok' or 'error'.")
    data: QueryAnswer = Field(..., description="Answer and retrieved contexts.")
    error: Optional[ApiError] = Field(default=None, description="Error info if any.")
    meta: Meta = Field(..., description="Response metadata.")


class ListDocumentsResponse(BaseModel):
    status: str = Field(..., description="Operation status. 'ok' or 'error'.")
    data: List[DocumentInfo] = Field(..., description="List of indexed documents.")
    error: Optional[ApiError] = Field(default=None, description="Error info if any.")
    meta: Meta = Field(..., description="Response metadata.")
