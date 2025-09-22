"""
Chat router: ask questions against the local PDF index using RAG.

Ocean Professional:
- Concise answers
- Include passages with optional scores to promote transparency
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..models.schemas import QueryRequest, QueryResponse, QueryAnswer, Passage, ApiError, Meta
from ..services.rag import RagPipeline
from .documents import get_pipeline

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Queries the local index with the user's question and returns a synthesized answer with supporting passages.",
)
# PUBLIC_INTERFACE
async def query_rag(payload: QueryRequest) -> JSONResponse:
    """
    Query the RAG pipeline with a question.

    Args:
        payload: QueryRequest containing the question, optional top_k and max_answer_length.

    Returns:
        QueryResponse: answer string and retrieved passages with optional scores.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        rag: RagPipeline = get_pipeline()
        answer, results = rag.query(
            question=payload.question,
            top_k=payload.top_k,
            max_answer_length=payload.max_answer_length or 512,
        )

        passages = []
        for c, score in results:
            passages.append(
                Passage(text=c.text, score=score, source=c.source, chunk_id=c.chunk_id).model_dump()
            )

        data = QueryAnswer(answer=answer, passages=passages)  # type: ignore[arg-type]
        meta = Meta(version="1.0.0", service="faq_chatbot_backend")
        return JSONResponse(
            content={"status": "ok", "data": data.model_dump(), "error": None, "meta": meta.model_dump()}
        )

    except ValidationError as ve:
        err = ApiError(code="validation_error", message=str(ve), details={"fields": ve.errors()})
        meta = Meta(version="1.0.0", service="faq_chatbot_backend")
        return JSONResponse(
            status_code=422,
            content={"status": "error", "data": None, "error": err.model_dump(), "meta": meta.model_dump()},
        )
    except Exception as e:
        err = ApiError(code="query_failed", message="Failed to run query.", details={"reason": str(e)})
        meta = Meta(version="1.0.0", service="faq_chatbot_backend")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "data": None, "error": err.model_dump(), "meta": meta.model_dump()},
        )
