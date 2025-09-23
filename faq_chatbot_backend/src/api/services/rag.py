"""
RAG pipeline service for offline PDF FAQ chatbot.

Implements:
- PDF ingestion: parse, chunk, embed, and index text
- Persistent vector index on disk (embeddings.npy + chunks.jsonl + meta.json)
- Query: retrieve top passages by cosine similarity and synthesize concise answer

Ocean Professional style:
- Clear structure, robust error handling, concise public interfaces
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from ..core.config import settings


@dataclass
class Chunk:
    """Represents a single text chunk with source metadata."""
    text: str
    source: str  # file_id or path
    chunk_id: int


class SimpleVectorStore:
    """
    A minimal persistent vector store backed by numpy arrays and JSONL metadata.

    Files inside INDEX_DIR:
    - embeddings.npy: float32 matrix of shape (N, D)
    - chunks.jsonl: one JSON object per line with { "text", "source", "chunk_id" }
    - meta.json: metadata about dimensionality and counts
    """

    def __init__(self, index_dir: Path, embedding_dim: int) -> None:
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.meta_path = self.index_dir / "meta.json"

        self._embeddings: Optional[np.ndarray] = None
        self._chunks: list[Chunk] = []
        self._nn: Optional[NearestNeighbors] = None

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load index from disk if present."""
        if self.embeddings_path.exists() and self.chunks_path.exists():
            try:
                self._embeddings = np.load(self.embeddings_path)
                self._chunks = []
                with self.chunks_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        j = json.loads(line)
                        self._chunks.append(Chunk(text=j["text"], source=j["source"], chunk_id=j["chunk_id"]))
                # Validate shape
                if self._embeddings.ndim != 2 or self._embeddings.shape[1] != self.embedding_dim:
                    # Corrupt or mismatched; reset
                    self._reset_index()
                else:
                    self._build_nn()
            except Exception:
                # In case of partial/corrupted state, reset cleanly
                self._reset_index()
        else:
            self._reset_index()

        # Write meta to reflect current state
        self._write_meta()

    def _reset_index(self) -> None:
        self._embeddings = np.zeros((0, self.embedding_dim), dtype=np.float32)
        self._chunks = []
        self._nn = None
        self._persist()

    def _persist(self) -> None:
        """Persist current embeddings and chunks to disk."""
        assert self._embeddings is not None
        np.save(self.embeddings_path, self._embeddings)
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps({"text": c.text, "source": c.source, "chunk_id": c.chunk_id}, ensure_ascii=False) + "\n")
        self._write_meta()
        self._build_nn()

    def _write_meta(self) -> None:
        meta = {
            "embedding_dim": self.embedding_dim,
            "num_chunks": len(self._chunks),
        }
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _build_nn(self) -> None:
        """Build or refresh the nearest neighbors index."""
        assert self._embeddings is not None
        if self._embeddings.shape[0] == 0:
            self._nn = None
            return
        # Use cosine metric; NearestNeighbors with brute force for simplicity
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._nn.fit(self._embeddings)

    def add(self, new_embeddings: np.ndarray, new_chunks: List[Chunk]) -> None:
        """Append new embeddings and chunks, then persist and rebuild."""
        assert new_embeddings.ndim == 2 and new_embeddings.shape[1] == self.embedding_dim
        assert len(new_chunks) == new_embeddings.shape[0]
        assert self._embeddings is not None

        if self._embeddings.shape[0] == 0:
            self._embeddings = new_embeddings.astype(np.float32)
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings.astype(np.float32)])
        self._chunks.extend(new_chunks)
        self._persist()

    def query(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top_k nearest chunks by cosine similarity.
        Returns list of (Chunk, score) where score is similarity (1 - distance).
        """
        assert query_embedding.ndim == 1 and query_embedding.shape[0] == self.embedding_dim
        if self._nn is None or self._embeddings is None or self._embeddings.shape[0] == 0:
            return []
        # reshape to 2D
        distances, indices = self._nn.kneighbors(query_embedding.reshape(1, -1), n_neighbors=min(top_k, self._embeddings.shape[0]))
        results: List[Tuple[Chunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            sim = float(1.0 - dist)
            results.append((self._chunks[int(idx)], sim))
        return results

    def list_documents(self) -> List[Tuple[str, int]]:
        """
        Return list of (file_id, num_chunks_for_that_file).
        We infer file_id from chunk.source (which is the file_id).
        """
        counts: dict[str, int] = {}
        for c in self._chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return sorted(counts.items(), key=lambda x: x[0])


class PDFParser:
    """Utility to parse and extract text from PDF bytes."""

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        """
        Extract text from a PDF using PyPDF2.
        """
        from io import BytesIO

        reader = PdfReader(BytesIO(pdf_bytes))
        texts: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                if t:
                    texts.append(t)
            except Exception:
                # continue parsing other pages
                continue
        return "\n".join(texts).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks by character count boundaries.

    Keeps it simple and robust for offline usage.
    """
    if not text:
        return []
    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks


class RagPipeline:
    """
    End-to-end pipeline combining PDF parsing, chunking, embeddings, local index, and query.

    Embedding model default: all-MiniLM-L6-v2 (384-dim) per README/config.
    """

    def __init__(self) -> None:
        # Ensure storage dirs exist
        from ..core.config import ensure_dirs  # local import to avoid cycles in module graph
        ensure_dirs(settings)

        self.upload_dir: Path = settings.UPLOAD_DIR
        self.index_dir: Path = settings.INDEX_DIR
        self.chunk_size = settings.CHUNK_SIZE
        self.overlap = settings.CHUNK_OVERLAP
        self.top_k_default = settings.TOP_K
        self.embedding_dim = settings.EMBEDDING_DIM
        self.include_scores = settings.INCLUDE_SCORES

        # Initialize embedding model
        self.model_name = settings.SENTENCE_EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

        # Initialize store
        self.store = SimpleVectorStore(index_dir=self.index_dir, embedding_dim=self.embedding_dim)

    def _load_model(self) -> None:
        """
        Load sentence-transformers model.
        If fully offline, ensure model is pre-cached or local path is provided.
        """
        try:
            self._model = SentenceTransformer(self.model_name)
            # Verify embedding dimension matches settings
            test_vec = self._model.encode(["test"], normalize_embeddings=True)
            if test_vec.shape[1] != self.embedding_dim:
                # Adjust to actual dimension to prevent mismatches
                self.embedding_dim = int(test_vec.shape[1])
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")

    def _embed(self, texts: List[str]) -> np.ndarray:
        assert self._model is not None
        # normalize=True so cosine similarity via 1 - cosine distance makes sense
        emb = self._model.encode(texts, normalize_embeddings=True)
        # ensure float32
        return emb.astype(np.float32)

    def _save_pdf(self, original_name: str, content: bytes) -> tuple[str, Path, int]:
        """
        Save the uploaded PDF to disk and return (file_id, path, size_bytes).
        """
        file_id = uuid.uuid4().hex
        path = self.upload_dir / f"{file_id}.pdf"
        with path.open("wb") as f:
            f.write(content)
        return file_id, path, len(content)

    # PUBLIC_INTERFACE
    def ingest_pdf(self, filename: str, file_bytes: bytes) -> Tuple[str, int, int]:
        """
        Ingest a PDF file into the local vector index.

        Args:
            filename: Original filename (used only for display).
            file_bytes: Content of the PDF file.

        Returns:
            tuple: (file_id, num_chunks, size_bytes)
        """
        # Save PDF
        file_id, saved_path, size_bytes = self._save_pdf(filename, file_bytes)

        # Extract text
        text = PDFParser.extract_text(file_bytes)
        if not text:
            # No text extracted; still record upload but 0 chunks
            return file_id, 0, size_bytes

        # Chunk
        texts = chunk_text(text, self.chunk_size, self.overlap)
        if not texts:
            return file_id, 0, size_bytes

        # Embed
        embeddings = self._embed(texts)
        # Build chunk metadata
        chunks = [Chunk(text=t, source=file_id, chunk_id=i) for i, t in enumerate(texts)]

        # Add to store
        self.store.add(embeddings, chunks)

        return file_id, len(texts), size_bytes

    # PUBLIC_INTERFACE
    def list_documents(self) -> List[Tuple[str, int]]:
        """
        List documents in the index.

        Returns:
            List of tuples (file_id, num_chunks).
        """
        return self.store.list_documents()

    # PUBLIC_INTERFACE
    def query(self, question: str, top_k: Optional[int] = None, max_answer_length: int = 512) -> Tuple[str, List[Tuple[Chunk, float]]]:
        """
        Run a RAG query:
        - Embed the question
        - Retrieve top_k passages
        - Synthesize a concise answer by heuristic summarization of top contexts
          (For offline mode without LLM access, we compose a summary using top chunks.)
          If an OPENAI_API_KEY environment variable is present, attempt a light LLM synthesis.

        Args:
            question: natural language question
            top_k: number of passages to retrieve (defaults to settings.TOP_K)
            max_answer_length: max characters for final answer

        Returns:
            (answer, [(Chunk, score)])
        """
        k = top_k or self.top_k_default
        q_emb = self._embed([question])[0]  # shape (D,)

        results = self.store.query(q_emb, top_k=k)

        # If no results, return a friendly message
        if not results:
            return "I couldn't find relevant information in the indexed documents.", []

        # Attempt LLM synthesis if OpenAI key is provided; otherwise provide heuristic answer.
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        if openai_key:
            try:
                answer = self._llm_summarize(question, results, max_answer_length)
                if answer:
                    return answer, results
            except Exception:
                # fall back to heuristic if LLM call fails
                pass

        # Heuristic synthesis: concatenate top snippets with light trimming
        answer = self._heuristic_compose_answer(question, results, max_answer_length)
        return answer, results

    def _heuristic_compose_answer(self, question: str, results: List[Tuple[Chunk, float]], max_len: int) -> str:
        """
        Build a concise answer from top chunks.
        """
        parts: list[str] = []
        # Leading sentence referencing relevance
        lead = f"Based on the indexed documents, here is a concise answer to: \"{question.strip()}\""
        parts.append(lead)

        # Append top passages (first few sentences or first 300 chars)
        for chunk, _score in results[:3]:
            snippet = chunk.text.strip().replace("\n", " ")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            parts.append(f"- {snippet}")

        answer = "\n".join(parts)
        if len(answer) > max_len:
            answer = answer[: max(0, max_len - 3)] + "..."
        return answer

    def _llm_summarize(self, question: str, results: List[Tuple[Chunk, float]], max_len: int) -> str:
        """
        Summarize using OpenAI if available. Uses a compact prompt and limits context.
        """
        try:
            from openai import OpenAI  # OpenAI Python SDK v1+
        except Exception as e:
            raise RuntimeError(f"OpenAI SDK not available: {e}")

        client = OpenAI()

        # Build context text with top passages
        context_parts: list[str] = []
        for chunk, score in results[:5]:
            line = chunk.text.strip().replace("\n", " ")
            if len(line) > 600:
                line = line[:597] + "..."
            context_parts.append(f"[source:{chunk.source} chunk:{chunk.chunk_id}] {line}")
        context = "\n".join(context_parts)

        prompt = (
            "You are a helpful assistant answering strictly based on the provided context.\n"
            "If the context does not contain the answer, say that you do not have enough information.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\n"
            f"Provide a concise answer within {max_len} characters:\n"
        )

        # Use responses API to get a short answer
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer using only the provided context. Be concise and accurate."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=256,
            )
            answer = resp.choices[0].message.content or ""
            answer = (answer or "").strip()
            if len(answer) > max_len:
                answer = answer[: max(0, max_len - 3)] + "..."
            return answer
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
