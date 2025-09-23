import io
from fastapi.testclient import TestClient

from .conftest import upload_pdf


def test_health_check(client: TestClient):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["data"]["message"] == "Healthy"
    assert body["meta"]["service"] == "faq_chatbot_backend"


def test_upload_rejects_non_pdf(client: TestClient):
    files = {"file": ("not_pdf.txt", io.BytesIO(b"hello"), "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    data = resp.json()
    # FastAPI raises HTTPException -> default detail field
    assert "Only PDF files are supported." in data["detail"]


def test_upload_valid_pdf_and_list_documents(client: TestClient, mock_sentence_transformer):
    # Upload a valid PDF
    with upload_pdf(client, filename="doc1.pdf", text="This is a test PDF with some content for chunking."):
        pass
    # List documents
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    body = list_resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["data"], list)
    # There should be exactly one document listed
    assert len(body["data"]) == 1
    doc = body["data"][0]
    assert doc["file_id"]  # some UUID-like id
    assert doc["filename"].endswith(".pdf")
    # num_chunks >= 0 (depends on extraction/chunking); we check it's an int
    assert isinstance(doc["num_chunks"], int)


def test_chat_query_without_documents_returns_friendly_message(client: TestClient, mock_sentence_transformer):
    # Query before any document is uploaded
    payload = {"question": "What is in the documents?", "top_k": 3}
    resp = client.post("/chat/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "couldn't find" in data["data"]["answer"].lower()
    assert data["data"]["passages"] == []


def test_chat_query_with_document_returns_passages(client: TestClient, mock_sentence_transformer):
    # Upload a document first
    with upload_pdf(client, filename="doc2.pdf", text="Python is a programming language. It supports unit testing with pytest."):
        pass

    # Now query
    payload = {"question": "What does the document say about Python?", "top_k": 4, "max_answer_length": 300}
    resp = client.post("/chat/query", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Answer should be non-empty
    assert isinstance(body["data"]["answer"], str) and len(body["data"]["answer"]) > 0
    # Passages should be a list with at least 1 item (since we have content)
    assert isinstance(body["data"]["passages"], list)
    # When there is content, expect at least one passage returned
    assert len(body["data"]["passages"]) >= 1
    first = body["data"]["passages"][0]
    assert "text" in first and "source" in first and "chunk_id" in first


def test_chat_query_validation_empty_question(client: TestClient):
    # Empty question should be 400
    resp = client.post("/chat/query", json={"question": "   "})
    assert resp.status_code == 400
    body = resp.json()
    assert "cannot be empty" in (body.get("detail") or "").lower()
