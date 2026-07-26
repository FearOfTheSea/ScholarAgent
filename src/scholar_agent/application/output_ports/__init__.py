"""Output-port contracts for external capabilities."""

from scholar_agent.application.output_ports.document_library import IDocumentLibrary
from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider
from scholar_agent.application.output_ports.graph_runner import IGraphRunner
from scholar_agent.application.output_ports.llm_provider import ILLMProvider
from scholar_agent.application.output_ports.memory_store import IMemoryStore
from scholar_agent.application.output_ports.pdf_loader import IPDFLoader
from scholar_agent.application.output_ports.retriever import IRetriever
from scholar_agent.application.output_ports.text_chunker import ITextChunker
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.application.output_ports.vector_store import IVectorStore

__all__ = [
    "IDocumentLibrary",
    "IEmbeddingProvider",
    "IGraphRunner",
    "ILLMProvider",
    "IMemoryStore",
    "IPDFLoader",
    "IRetriever",
    "IToolExecutor",
    "ITextChunker",
    "IVectorStore",
]
