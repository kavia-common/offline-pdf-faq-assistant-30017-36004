import io
from contextlib import contextmanager
from typing import Iterator
import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.main import app


@pytest.fixture(autouse=True)
def isolation_tmp_dirs(monkeypatch, tmp_path):
    """
    Ensure each test runs with isolated storage directories by patching env variables.
    Also clear the module-level singleton pipeline so state doesn't leak between tests.
    """
    # Point storage paths to a unique temporary directory per test
    base_dir = tmp_path / "storage"
    uploads_dir = base_dir / "uploads"
    index_dir = base_dir / "index"

    monkeypatch.setenv("BASE_DIR", str(base_dir))
    monkeypatch.setenv("UPLOAD_DIR", str(uploads_dir))
    monkeypatch.setenv("INDEX_DIR", str(index_dir))

    # Clear OPENAI to avoid accidental network calls
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Reset the router-level singleton pipeline to None so each test creates fresh instance
    from src.api.routers import documents as docs_module
    docs_module.pipeline = None

    yield
    # no explicit cleanup needed; tmp_path will be cleaned by pytest


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """
    Provide a TestClient for the FastAPI app.
    """
    with TestClient(app) as c:
        yield c


class DummyModel:
    """
    Minimal dummy embedding model to avoid network/download.
    Produces deterministic small-dimension embeddings.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        # Accept single string or list
        if isinstance(texts, str):
            texts = [texts]

        embs = []
        for t in texts:
            # hash-based pseudo-embedding
            h = abs(hash(t)) % (10**9)
            vec = [(h + i * 15485863) % 1000 for i in range(self.dim)]
            v = np.array(vec, dtype="float32")
            if normalize_embeddings:
                norm = float((v**2).sum()) ** 0.5
                if norm > 0:
                    v = v / norm
            embs.append(v)
        return np.vstack(embs)


@pytest.fixture()
def mock_sentence_transformer(monkeypatch):
    """
    Monkeypatch SentenceTransformer to use a dummy model with fixed small dimension,
    and also monkeypatch EMBEDDING_DIM in settings to match.
    """
    # Patch SentenceTransformer init to return DummyModel and its encode method
    from src.api import services as services_pkg  # noqa
    from src.api.services import rag as rag_module

    # Force embedding dim to 8 for deterministic behavior
    monkeypatch.setenv("EMBEDDING_DIM", "8")

    def dummy_init(model_name):
        return DummyModel(dim=8)

    monkeypatch.setattr(rag_module, "SentenceTransformer", lambda *args, **kwargs: DummyModel(dim=8))


def make_minimal_pdf_bytes(text: str = "Hello PDF") -> bytes:
    """
    Create a minimal valid PDF in memory containing the provided text.
    Using a tiny hand-crafted PDF object to avoid extra dependencies.
    Note: Many PDF parsers (including PyPDF2) can extract 'Hello PDF' from such a basic file.
    """
    # A very simple PDF that includes the text in a text object.
    # It may not be robust, but suffices for simple extraction attempts.
    content_stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    pdf_bytes = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Resources << /Font << /F1 5 0 R >> >>
/Contents 4 0 R
>>
endobj
4 0 obj
<< /Length {len(content_stream)} >>
stream
{content_stream}
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000061 00000 n 
0000000118 00000 n 
0000000334 00000 n 
0000000501 00000 n 
trailer
<< /Root 1 0 R /Size 6 >>
startxref
610
%%EOF
"""
    return pdf_bytes.encode("latin-1")


@contextmanager
def upload_pdf(client: TestClient, filename: str = "sample.pdf", text: str = "Hello PDF"):
    """
    Context manager to upload a minimal PDF, yielding the API response JSON.
    """
    pdf_bytes = make_minimal_pdf_bytes(text=text)
    files = {"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")}
    resp = client.post("/documents/upload", files=files)
    yield resp
