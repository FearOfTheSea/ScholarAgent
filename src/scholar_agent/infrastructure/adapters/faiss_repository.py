"""Persistent FAISS implementation of the vector-store port."""

import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from scholar_agent.application.dtos.retrieval import DocumentChunk, RetrievedChunk
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.domain.value_objects.document_id import DocumentId


class FAISSRepository(IVectorStore):
    """Stores normalized embeddings in FAISS and source metadata in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self._database_path / "index.faiss"
        self._connection = sqlite3.connect(
            self._database_path / "metadata.sqlite3",
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        self._index: Any | None = None
        self._initialize_schema()

    def add(
        self,
        chunks: tuple[DocumentChunk, ...],
        embeddings: tuple[tuple[float, ...], ...],
    ) -> None:
        """Persist chunks and their normalized embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding.")
        if not chunks:
            return

        vectors = self._normalize_vectors(embeddings)
        with self._lock:
            index = self._get_index(vectors.shape[1])
            vector_ids: list[int] = []
            with self._connection:
                for chunk in chunks:
                    cursor = self._connection.execute(
                        """
                        INSERT INTO chunks
                            (document_id, chunk_id, content,
                             page_number, section, ordinal)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.document_id.value,
                            chunk.chunk_id,
                            chunk.content,
                            chunk.page_number,
                            chunk.section,
                            chunk.ordinal,
                        ),
                    )
                    last_row_id = cursor.lastrowid
                    if last_row_id is None:
                        raise RuntimeError("SQLite did not return a vector identifier.")
                    vector_ids.append(last_row_id)
            index.add_with_ids(vectors, np.asarray(vector_ids, dtype=np.int64))
            self._save_index(index)

    def search(
        self,
        embedding: tuple[float, ...],
        limit: int = 5,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> tuple[RetrievedChunk, ...]:
        """Return nearest chunks, optionally limited to selected documents."""
        if limit < 1:
            return ()
        with self._lock:
            if self._index is None and not self._index_path.exists():
                return ()
            index = self._get_index(len(embedding))
            if index.ntotal == 0:
                return ()
            query = self._normalize_vectors((embedding,))
            probe_size = min(max(limit * 10, 50), int(index.ntotal))
            distances, vector_ids = index.search(query, probe_size)
            allowed_document_ids = {document_id.value for document_id in document_ids}
            results: list[RetrievedChunk] = []
            for vector_id, distance in zip(vector_ids[0], distances[0], strict=True):
                if vector_id < 0:
                    continue
                row = self._connection.execute(
                    "SELECT * FROM chunks WHERE vector_id = ?",
                    (int(vector_id),),
                ).fetchone()
                if row is None:
                    continue
                document_id = str(row["document_id"])
                if allowed_document_ids and document_id not in allowed_document_ids:
                    continue
                results.append(
                    RetrievedChunk(
                        document_id=DocumentId(document_id),
                        content=str(row["content"]),
                        page_number=self._optional_int(row["page_number"]),
                        section=self._optional_str(row["section"]),
                        chunk_id=str(row["chunk_id"]),
                        similarity_score=float(distance),
                    ),
                )
                if len(results) == limit:
                    break
            return tuple(results)

    def list_document_chunks(
        self, document_id: DocumentId
    ) -> tuple[DocumentChunk, ...]:
        """Return persisted document chunks in their original order."""
        rows = self._connection.execute(
            """
            SELECT * FROM chunks
            WHERE document_id = ?
            ORDER BY ordinal ASC
            """,
            (document_id.value,),
        ).fetchall()
        return tuple(self._to_document_chunk(row) for row in rows)

    def get_chunk(
        self,
        document_id: DocumentId,
        chunk_id: str,
    ) -> DocumentChunk | None:
        """Return one persisted chunk when it belongs to the specified document."""
        row = self._connection.execute(
            """
            SELECT * FROM chunks
            WHERE document_id = ? AND chunk_id = ?
            """,
            (document_id.value, chunk_id),
        ).fetchone()
        return self._to_document_chunk(row) if row is not None else None

    def delete_document(self, document_id: DocumentId) -> None:
        """Remove a document's FAISS vectors and SQLite metadata."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT vector_id FROM chunks WHERE document_id = ?",
                (document_id.value,),
            ).fetchall()
            vector_ids = [int(row["vector_id"]) for row in rows]
            if vector_ids and (self._index is not None or self._index_path.exists()):
                index = self._get_index_from_existing()
                if index is not None:
                    index.remove_ids(np.asarray(vector_ids, dtype=np.int64))
                    self._save_index(index)
            with self._connection:
                self._connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?",
                    (document_id.value,),
                )

    def close(self) -> None:
        """Release the SQLite metadata connection when a local job is complete."""
        with self._lock:
            self._connection.close()

    def _get_index(self, dimension: int) -> Any:
        if self._index is None:
            import faiss

            if self._index_path.exists():
                self._index = faiss.read_index(str(self._index_path))
                if int(self._index.d) != dimension:
                    raise ValueError(
                        "Embedding dimension does not match the stored index."
                    )
            else:
                self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        return self._index

    def _get_index_from_existing(self) -> Any | None:
        if self._index is not None:
            return self._index
        if not self._index_path.exists():
            return None
        import faiss

        self._index = faiss.read_index(str(self._index_path))
        return self._index

    def _save_index(self, index: Any) -> None:
        import faiss

        faiss.write_index(index, str(self._index_path))

    def _initialize_schema(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                page_number INTEGER,
                section TEXT,
                ordinal INTEGER NOT NULL
            )
            """,
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
        )
        self._connection.commit()

    @staticmethod
    def _normalize_vectors(
        embeddings: tuple[tuple[float, ...], ...],
    ) -> np.ndarray[Any, Any]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a non-empty two-dimensional matrix.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embeddings must not contain a zero vector.")
        return vectors / norms

    @staticmethod
    def _to_document_chunk(row: sqlite3.Row) -> DocumentChunk:
        return DocumentChunk(
            document_id=DocumentId(str(row["document_id"])),
            content=str(row["content"]),
            page_number=FAISSRepository._optional_int(row["page_number"]),
            section=FAISSRepository._optional_str(row["section"]),
            chunk_id=str(row["chunk_id"]),
            ordinal=int(row["ordinal"]),
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, (str, bytes, bytearray)):
            return int(value)
        raise TypeError("SQLite integer values must be scalar values.")

    @staticmethod
    def _optional_str(value: object) -> str | None:
        return str(value) if value is not None else None
