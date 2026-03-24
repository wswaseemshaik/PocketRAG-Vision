from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from utils.config import OllamaConfig
from utils.llm import ChatMessage, OllamaChatClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    chunk_text: str
    doc_path: str
    chunk_index: int


class MiniRAGPipeline:
    def __init__(
        self,
        *,
        docs_dir: Path,
        db_path: Path,
        embedding_model_name: str,
        top_k: int,
        llm: OllamaConfig,
        chunk_size_words: int,
        chunk_overlap_words: int,
    ) -> None:
        if chunk_overlap_words >= chunk_size_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be < CHUNK_SIZE_WORDS")

        self.docs_dir = docs_dir
        self.db_path = db_path
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k

        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words

        self.llm_client = OllamaChatClient(llm)

        self._index_ready = asyncio.Event()
        self._build_lock = asyncio.Lock()
        self._embed_lock = asyncio.Lock()

        # In-memory index (loaded after build).
        self._chunks: List[RagChunk] = []
        self._embeddings: Optional[np.ndarray] = None  # shape: (n, dim)
        self._embedding_dim: Optional[int] = None

        self._embedder = None

    async def build_index_if_needed(self) -> None:
        if self._index_ready.is_set():
            return

        async with self._build_lock:
            if self._index_ready.is_set():
                return

            await asyncio.to_thread(self._build_index_sync)
            self._index_ready.set()

    async def answer(self, query: str, memory: Sequence[Dict[str, str]]) -> str:
        if not query.strip():
            return "Please provide a query: `/ask <query>`."

        if not await self._ensure_index_ready():
            return "RAG index is still building. Try again in a moment."

        # Embed query without blocking the event loop.
        async with self._embed_lock:
            q_emb = await asyncio.to_thread(self._encode_query_sync, query)

        # Dot product because sentence-transformers embeddings are normalized.
        assert self._embeddings is not None
        scores = self._embeddings @ q_emb
        top_idx = np.argsort(scores)[::-1][: self.top_k]
        retrieved = [self._chunks[i] for i in top_idx]
        source_docs = [Path(c.doc_path).name for c in retrieved]
        # Preserve retrieval order while removing duplicates.
        source_docs = list(dict.fromkeys(source_docs))
        source_line = f"Source: {', '.join(source_docs)}"

        context = "\n\n".join(
            f"[{i+1}] (source: {Path(c.doc_path).name})\n{c.chunk_text}"
            for i, c in enumerate(retrieved)
        )
        # Keep prompt size predictable for faster local generation.
        context = context[:1800]

        memory_text = "\n".join(
            f"- {m.get('role','user')}: {m.get('content','')}"
            for m in memory[-3:]
            if (m.get("content") or "").strip()
        ).strip()

        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "If the context is insufficient, reply with: \"I don't know.\" "
            "Keep your answer concise (max 6 sentences)."
        )

        user_prompt = (
            (f"Conversation history:\n{memory_text}\n\n" if memory_text else "")
            + f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        )

        try:
            answer_text = await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=140,
                temperature=0.1,
            )
            return f"{answer_text}\n\n{source_line}"
        except Exception:
            logger.exception("LLM call failed")
            return (
                "Sorry, I couldn't generate an answer right now (LLM unavailable)."
                f"\n\n{source_line}"
            )

    async def summarize(self, query: Optional[str] = None) -> str:
        if not await self._ensure_index_ready():
            return "RAG index is still building. Try again in a moment."

        assert self._embeddings is not None
        if not self._chunks:
            return "No documents are indexed yet. Add files under `data/docs/` and restart."

        cleaned_query = (query or "").strip()
        if cleaned_query:
            async with self._embed_lock:
                q_emb = await asyncio.to_thread(self._encode_query_sync, cleaned_query)
            scores = self._embeddings @ q_emb
            top_idx = np.argsort(scores)[::-1][: max(4, self.top_k + 1)]
            selected_chunks = [self._chunks[i] for i in top_idx]
            objective = f"Provide a concise summary focused on: {cleaned_query}"
        else:
            selected_chunks = self._select_summary_chunks(max_chunks=5)
            objective = "Provide a concise summary of the available documents."

        context = "\n\n".join(
            f"[{i+1}] (source: {Path(c.doc_path).name})\n{c.chunk_text}"
            for i, c in enumerate(selected_chunks)
        )
        context = context[:2800]

        system_prompt = (
            "You are a document summarizer. Use ONLY the provided context. "
            "Return a short summary in 5-8 bullet points."
        )
        user_prompt = f"{objective}\n\nContext:\n{context}\n\nSummary:"

        try:
            return await self.llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=170,
                temperature=0.1,
            )
        except Exception:
            logger.exception("LLM summarize call failed")
            return self._fallback_summary(selected_chunks, query=cleaned_query)

    async def _ensure_index_ready(self) -> bool:
        if self._index_ready.is_set():
            return True
        try:
            await asyncio.wait_for(self.build_index_if_needed(), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            return False

    def _select_summary_chunks(self, max_chunks: int = 8) -> List[RagChunk]:
        # Prefer early chunks from each doc for broad summaries.
        by_doc: Dict[str, List[RagChunk]] = {}
        for chunk in self._chunks:
            by_doc.setdefault(chunk.doc_path, []).append(chunk)

        selected: List[RagChunk] = []
        for doc_path in sorted(by_doc.keys()):
            chunks = sorted(by_doc[doc_path], key=lambda c: c.chunk_index)
            selected.extend(chunks[:2])

        if not selected:
            return self._chunks[:max_chunks]
        return selected[:max_chunks]

    def _fallback_summary(self, chunks: Sequence[RagChunk], query: str = "") -> str:
        if not chunks:
            return "I couldn't summarize because no document chunks were available."

        lines: List[str] = []
        header = (
            f"Summary (focused on: {query}):"
            if query
            else "Summary of indexed documents:"
        )
        lines.append(header)
        for chunk in chunks[:6]:
            snippet = " ".join(chunk.chunk_text.split()[:24]).strip()
            if snippet and not snippet.endswith("."):
                snippet += "..."
            lines.append(f"- ({Path(chunk.doc_path).name}) {snippet}")
        return "\n".join(lines)

    def _encode_query_sync(self, query: str) -> np.ndarray:
        if self._embedder is None:
            # Should only happen if build_index_if_needed was not called.
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        vec = self._embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vec = np.asarray(vec, dtype=np.float32)
        return vec

    def _build_index_sync(self) -> None:
        start = time.time()
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_db()
        doc_state = self._compute_doc_state()
        if not doc_state["files"]:
            raise RuntimeError(
                f"No documents found in `{self.docs_dir}`. Add 3-5 .txt/.md files."
            )

        # Fast path: avoid loading embedding model if index is unchanged.
        if self._is_index_fresh(doc_state):
            self._load_in_memory_index()
            elapsed = time.time() - start
            logger.info("RAG index up-to-date; loaded in %.2fs", elapsed)
            return

        # Import heavy dependency only if we actually need to (re)embed.
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.embedding_model_name)

        all_chunks: List[RagChunk] = []
        for doc_path in self._iter_doc_files():
            text = doc_path.read_text(encoding="utf-8", errors="ignore")
            doc_chunks = self._chunk_text(text)
            for idx, chunk_text in enumerate(doc_chunks):
                chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                all_chunks.append(
                    RagChunk(
                        chunk_id=chunk_hash,
                        chunk_text=chunk_text,
                        doc_path=str(doc_path),
                        chunk_index=idx,
                    )
                )

        if not all_chunks:
            raise RuntimeError(
                f"No documents found in `{self.docs_dir}`. Add 3-5 .txt/.md files."
            )

        logger.info("Rebuilding embedding index: %d chunks", len(all_chunks))
        chunk_texts = [c.chunk_text for c in all_chunks]
        embedded = self._embedder.encode(
            chunk_texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embedded = np.asarray(embedded, dtype=np.float32)
        embedding_dim = int(embedded.shape[1])

        rows = []
        for i, chunk in enumerate(all_chunks):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.doc_path,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    embedding_dim,
                    embedded[i].tobytes(),
                )
            )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings (
                    chunk_id, doc_path, chunk_index, chunk_text, embedding_dim, embedding
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._set_meta(conn, "doc_state", json.dumps(doc_state, sort_keys=True))
            self._set_meta(conn, "embedding_model_name", self.embedding_model_name)
            self._set_meta(conn, "chunk_size_words", str(self.chunk_size_words))
            self._set_meta(conn, "chunk_overlap_words", str(self.chunk_overlap_words))
            conn.commit()

        self._load_in_memory_index()
        elapsed = time.time() - start
        logger.info("RAG index build finished in %.2fs", elapsed)

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    doc_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_doc_path ON embeddings(doc_path)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def _iter_doc_files(self) -> Sequence[Path]:
        if not self.docs_dir.exists():
            return []

        exts = {".txt", ".md", ".markdown"}
        paths: List[Path] = []
        for p in sorted(self.docs_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
        return paths

    def _compute_doc_state(self) -> Dict[str, object]:
        files = []
        for p in self._iter_doc_files():
            stat = p.stat()
            files.append(
                {
                    "path": str(p.relative_to(self.docs_dir)),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        return {"files": files}

    def _get_meta(self, conn: sqlite3.Connection, key: str) -> Optional[str]:
        row = conn.execute("SELECT value FROM rag_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row[0])

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO rag_meta(key, value) VALUES(?, ?)",
            (key, value),
        )

    def _is_index_fresh(self, current_doc_state: Dict[str, object]) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cnt_row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            emb_count = int(cnt_row[0]) if cnt_row else 0
            if emb_count == 0:
                return False

            saved_doc_state = self._get_meta(conn, "doc_state")
            saved_model = self._get_meta(conn, "embedding_model_name")
            saved_chunk_size = self._get_meta(conn, "chunk_size_words")
            saved_overlap = self._get_meta(conn, "chunk_overlap_words")

        if saved_doc_state != json.dumps(current_doc_state, sort_keys=True):
            return False
        if saved_model != self.embedding_model_name:
            return False
        if saved_chunk_size != str(self.chunk_size_words):
            return False
        if saved_overlap != str(self.chunk_overlap_words):
            return False
        return True

    def _chunk_text(self, text: str) -> List[str]:
        # Word-based chunking keeps it simple and fast.
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size_words - self.chunk_overlap_words)
        chunks: List[str] = []
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            if not chunk_words:
                continue
            chunk = " ".join(chunk_words).strip()
            # Avoid extremely tiny chunks.
            if len(chunk) < 50:
                continue
            chunks.append(chunk)
        return chunks

    def _load_in_memory_index(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT chunk_id, doc_path, chunk_index, chunk_text, embedding_dim, embedding FROM embeddings"
            ).fetchall()

        chunks: List[RagChunk] = []
        embeddings: List[np.ndarray] = []
        embedding_dim: Optional[int] = None

        for chunk_id, doc_path, chunk_index, chunk_text, dim, emb_blob in rows:
            dim = int(dim)
            if embedding_dim is None:
                embedding_dim = dim
            # Convert BLOB -> float32 vector.
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            if vec.shape[0] != dim:
                # Skip corrupted rows.
                continue
            chunks.append(
                RagChunk(
                    chunk_id=str(chunk_id),
                    chunk_text=str(chunk_text),
                    doc_path=str(doc_path),
                    chunk_index=int(chunk_index),
                )
            )
            embeddings.append(vec)

        if not chunks or embedding_dim is None:
            raise RuntimeError("Embeddings index loaded empty/corrupted.")

        self._chunks = chunks
        self._embeddings = np.stack(embeddings, axis=0)
        self._embedding_dim = embedding_dim

