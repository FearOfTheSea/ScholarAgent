"""LangChain text-splitter implementation of the chunking port."""

from scholar_agent.application.dtos.retrieval import DocumentChunk, LoadedPage
from scholar_agent.application.output_ports.text_chunker import ITextChunker


class LangChainTextChunker(ITextChunker):
    """Creates page-aware chunks using LangChain's recursive splitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, pages: tuple[LoadedPage, ...]) -> tuple[DocumentChunk, ...]:
        """Split each extracted page while retaining page metadata."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        chunks: list[DocumentChunk] = []
        for page in pages:
            for page_chunk in splitter.split_text(page.content):
                ordinal = len(chunks)
                chunks.append(
                    DocumentChunk(
                        document_id=page.document_id,
                        content=page_chunk,
                        page_number=page.page_number,
                        section=None,
                        chunk_id=f"{page.document_id.value}-{ordinal}",
                        ordinal=ordinal,
                    ),
                )
        return tuple(chunks)
