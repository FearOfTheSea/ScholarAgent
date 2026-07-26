"""SQLite implementation of the document repository contract."""

import sqlite3
from datetime import datetime
from pathlib import Path

from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.repositories.document_repository import DocumentRepository
from scholar_agent.domain.value_objects.document_id import DocumentId


class SQLiteDocumentRepository(DocumentRepository):
    """Persists the local library catalog in one SQLite database."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
        )
        self._connection.commit()

    def save(self, document: Document) -> None:
        """Insert or replace a local document catalog record."""
        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO documents
                    (document_id, title, source, page_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document.identifier.value,
                    document.title,
                    document.source,
                    document.page_count,
                    document.created_at.isoformat(),
                ),
            )

    def get_by_id(self, document_id: DocumentId) -> Document | None:
        """Return a document when it exists in the local catalog."""
        row = self._connection.execute(
            "SELECT * FROM documents WHERE document_id = ?",
            (document_id.value,),
        ).fetchone()
        return self._to_document(row) if row is not None else None

    def list_all(self) -> tuple[Document, ...]:
        """Return local documents from newest to oldest."""
        rows = self._connection.execute(
            "SELECT * FROM documents ORDER BY created_at DESC",
        ).fetchall()
        return tuple(self._to_document(row) for row in rows)

    def delete(self, document_id: DocumentId) -> bool:
        """Delete a local catalog record."""
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id.value,),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _to_document(row: sqlite3.Row) -> Document:
        return Document(
            identifier=DocumentId(str(row["document_id"])),
            title=str(row["title"]),
            source=str(row["source"]),
            page_count=int(row["page_count"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )
