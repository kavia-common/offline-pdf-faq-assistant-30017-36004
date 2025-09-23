# Test Suite for FAQ Chatbot Backend

This folder contains pytest-based integration and validation tests for the FastAPI backend.

Coverage:
- GET `/` health check
- POST `/documents/upload` with valid and invalid files
- GET `/documents` listing
- POST `/chat/query` with and without indexed documents, and empty question validation

Key Notes:
- Tests run in isolation with per-test temporary storage (UPLOAD_DIR, INDEX_DIR).
- SentenceTransformer is monkeypatched with a local dummy model to avoid network access and large downloads.
- OPENAI API calls are implicitly disabled by unsetting OPENAI_API_KEY.
- Minimal valid PDF bytes are generated in-memory; no files are written outside the test temp folders.

Run tests:
```
pytest -q
```
