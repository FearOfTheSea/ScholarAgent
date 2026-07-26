"""Document entity."""

from dataclasses import dataclass
from datetime import datetime

from scholar_agent.domain.value_objects.document_id import DocumentId


@dataclass(frozen=True, slots=True)
class Document:
    """A source document known to ScholarAgent."""

    identifier: DocumentId
    title: str
    source: str
    page_count: int
    created_at: datetime
