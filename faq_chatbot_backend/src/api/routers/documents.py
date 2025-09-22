"""
Documents router: upload PDFs and list indexed documents.

Ocean Professional:
- Return structured JSON with status/data/error/meta
- Provide clear, actionable error messages with error codes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..models.schemas import (
    UploadResponse,
    ApiError,
    Meta,
    DocumentInfo,
    ListDocumentsResponse,
)
from ..services.rag import RagPipeline

router = APIRouter()

# Initialize a singleton pipeline for the app instance
# In a larger app, this might be moved to app state.
pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RagPipeline()
    return pipeline


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF to index",
    description="Accepts a PDF file, extracts text, chunks, embeds, and updates the local index.",
)
# PUBLIC_INTERFACE
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload a single PDF and index it for RAG.

    Args:
        file: The PDF file to upload.

    Returns:
        UploadResponse: Operation status, document info, and meta.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        rag = get_pipeline()
        file_id, num_chunks, size_bytes = rag.ingest_pdf(file.filename, contents)
        data = DocumentInfo(
            filename=file.filename,
            file_id=file_id,
            num_chunks=num_chunks,
            size_bytes=size_bytes,
        )
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
        err = ApiError(code="upload_failed", message="Failed to process the PDF.", details={"reason": str(e)})
        meta = Meta(version="1.0.0", service="faq_chatbot_backend")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "data": None, "error": err.model_dump(), "meta": meta.model_dump()},
        )


@router.get(
    "",
    response_model=ListDocumentsResponse,
    summary="List indexed documents",
    description="Returns a list of uploaded PDF documents with chunk counts.",
)
# PUBLIC_INTERFACE
async def list_documents() -> JSONResponse:
    """
    List indexed documents.

    Returns:
        ListDocumentsResponse: documents info.
    """
    try:
        rag = get_pipeline()
        docs = rag.list_documents()
        data = []
        for file_id, count in docs:
            data.append(
                DocumentInfo(
                    filename=f"{file_id}.pdf",
                    file_id=file_id,
                    num_chunks=count,
                    size_bytes=0,  # size not tracked per file after ingestion; set to 0
                ).model_dump()
            )
        meta = Meta(version="1.0.0", service="faq_chatbot_backend")
        return JSONResponse(
            content={"status": "ok", "data": data, "error": None, "meta": meta.model_dump()}
        )
    except Exception as e:
        err = ApiError(code="list_failed", message="Failed to list documents.", details={"reason": str(e)})
        meta = Meta(version="1.0.0", service="faq_chatbot_backend")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "data": None, "error": err.model_dump(), "meta": meta.model_dump()},
        )
