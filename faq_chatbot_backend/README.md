# Offline PDF FAQ Chatbot Backend (FastAPI)

An offline Retrieval-Augmented Generation (RAG) backend that lets users upload PDF files and ask questions against them. The service runs locally without cloud dependencies, storing embeddings and indexes on disk.

Theme: Ocean Professional — blue primary (#2563EB) and amber secondary (#F59E0B). Responses are structured with status, data, error, and meta for clarity.

## Features
- PDF upload and ingestion (text extraction, chunking)
- Local embeddings via sentence-transformers
- Persistent vector index on disk
- Chat endpoint that retrieves top passages and synthesizes concise answers
- Well-documented OpenAPI schema at /docs

## Tech Stack
- FastAPI + Uvicorn
- PyPDF2 for PDF parsing
- sentence-transformers for embeddings (default: all-MiniLM-L6-v2)
- NumPy + scikit-learn (NearestNeighbors) for vector search
- Local storage for uploads and index under ./storage

## Getting Started

1) Create and populate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Configure environment (optional):
```
cp .env.example .env
# Edit paths and defaults as needed
```

3) Run the server:
```
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI docs: http://localhost:8000/docs

## Endpoints

- GET `/` — Health Check
  - Returns: `{ status: "ok", data: { message: "Healthy" }, error: null, meta: { version, service } }`

- POST `/documents/upload` — Upload a PDF to index
  - Form-data file field name: `file`
  - Response: UploadResponse (status, data with file_id and chunks, meta)
  - Errors:
    - 400 if non-PDF file
    - 500 on processing failures

- GET `/documents` — List indexed documents
  - Response: ListDocumentsResponse (status, data[] of DocumentInfo, meta)

- POST `/chat/query` — Ask a question against the indexed PDFs
  - Body: `{ "question": "your question", "top_k": 4, "max_answer_length": 512 }`
  - Response: QueryResponse with synthesized `answer` and `passages[]` (each with text, score, source, chunk_id)
  - Errors:
    - 400 if question is empty
    - 500 on query failures

## Storage Layout
- `./storage/uploads/` — original PDF files (`{file_id}.pdf`)
- `./storage/index/` — vector store
  - `embeddings.npy` — float32 matrix
  - `chunks.jsonl` — texts and metadata
  - `meta.json` — store metadata

Note: Sizes in list response are shown as 0 (post-ingestion); size is reported accurately in the upload response.

## Configuration
Environment variables (see `.env.example`):
- BASE_DIR, UPLOAD_DIR, INDEX_DIR
- CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
- SENTENCE_EMBEDDING_MODEL, EMBEDDING_DIM
- CORS_ALLOW_ORIGINS
- INCLUDE_SCORES

## Ocean Professional Style
- Response format is consistent: `status`, `data`, `error`, `meta`
- Clean, professional naming with concise error messages and machine-readable `error.code`
- API docs include summaries, descriptions, and tags

## Notes
- The default embedding model is downloaded the first time it is used; ensure internet access at least once to cache, or pre-cache the model locally.
- For strict offline environments, pre-download the model and set SENTENCE_EMBEDDING_MODEL to a local path.

## License
MIT
