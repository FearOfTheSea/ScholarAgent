"""Text-chunking port."""

from abc import ABC, abstractmethod

from scholar_agent.application.dtos.retrieval import DocumentChunk, LoadedPage


class ITextChunker(ABC):
    """Splits extracted pages into embedding-ready chunks."""

    @abstractmethod
    def chunk(self, pages: tuple[LoadedPage, ...]) -> tuple[DocumentChunk, ...]:
        """Create source-aware chunks from document pages."""
