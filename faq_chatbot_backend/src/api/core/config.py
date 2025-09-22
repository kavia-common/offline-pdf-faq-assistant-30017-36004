"""
Configuration for the FAQ Chatbot Backend.

Ocean Professional accents:
- Defaults aim for professionalism, clarity, and local-first operation.
"""

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env from .env file if present
load_dotenv()


class Settings(BaseModel):
    """
    Application settings loaded from environment variables with safe defaults.
    """

    # Storage directories
    BASE_DIR: Path = Path(os.getenv("BASE_DIR", "./storage")).resolve()
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./storage/uploads")).resolve()
    INDEX_DIR: Path = Path(os.getenv("INDEX_DIR", "./storage/index")).resolve()

    # CORS
    CORS_ALLOW_ORIGINS: list[str] = (
        os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
        if os.getenv("CORS_ALLOW_ORIGINS")
        else ["*"]
    )

    # RAG parameters
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "4"))
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))  # for default MiniLM
    # Model names kept configurable for future extensibility
    SENTENCE_EMBEDDING_MODEL: str = os.getenv(
        "SENTENCE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # Response style toggles
    INCLUDE_SCORES: bool = os.getenv("INCLUDE_SCORES", "true").lower() == "true"


def ensure_dirs(settings: Settings) -> None:
    """
    Ensure required directories exist.
    """
    settings.BASE_DIR.mkdir(parents=True, exist_ok=True)
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
ensure_dirs(settings)
