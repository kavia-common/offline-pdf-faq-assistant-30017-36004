"""
FastAPI application entrypoint for the FAQ Chatbot Backend.

Ocean Professional style guide:
- Apply clear, consistent response structures with keys: status, data, error, meta
- Use primary (#2563EB) and secondary (#F59E0B) accents in docs descriptions/comments
- Keep messages concise and professional, with actionable error details
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .routers import documents, chat
from .core.config import settings
from .core.openapi import api_tags_metadata


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI app with CORS and routers.
    """
    app = FastAPI(
        title="Offline PDF FAQ Chatbot API",
        description=(
            "A Retrieval-Augmented Generation (RAG) backend for offline FAQ and question answering over uploaded PDFs.\n\n"
            "Ocean Professional theme: Blue (#2563EB) primary, Amber (#F59E0B) secondary accents.\n"
            "Endpoints provide structured responses with 'status', 'data', 'error', and 'meta'."
        ),
        version="1.0.0",
        openapi_tags=api_tags_metadata,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(documents.router, prefix="/documents", tags=["Documents"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])

    @app.get("/", summary="Health Check", tags=["System"])
    # PUBLIC_INTERFACE
    def health_check():
        """Health check endpoint.
        Returns a simple status message indicating the service is up.

        Returns:
            JSONResponse: { status: 'ok', data: { message: 'Healthy' }, error: None, meta: { version, service } }
        """
        return JSONResponse(
            content={
                "status": "ok",
                "data": {"message": "Healthy"},
                "error": None,
                "meta": {"version": app.version, "service": "faq_chatbot_backend"},
            }
        )

    return app


app = create_app()


def custom_openapi():
    """
    Customize OpenAPI schema for Ocean Professional presentation.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Tags are already applied via openapi_tags
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
