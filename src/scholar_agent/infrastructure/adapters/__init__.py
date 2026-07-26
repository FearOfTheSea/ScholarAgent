"""Infrastructure implementations of application output ports."""

from scholar_agent.infrastructure.adapters.faiss_repository import FAISSRepository
from scholar_agent.infrastructure.adapters.in_memory_store import InMemoryStore
from scholar_agent.infrastructure.adapters.langchain_retriever import LangChainRetriever
from scholar_agent.infrastructure.adapters.langchain_text_chunker import (
    LangChainTextChunker,
)
from scholar_agent.infrastructure.adapters.langgraph_runner import LangGraphRunner
from scholar_agent.infrastructure.adapters.local_document_library import (
    LocalDocumentLibrary,
)
from scholar_agent.infrastructure.adapters.ollama_adapter import OllamaAdapter
from scholar_agent.infrastructure.adapters.placeholder_tool_executor import (
    PlaceholderToolExecutor,
)
from scholar_agent.infrastructure.adapters.pymupdf_loader import PyMuPDFLoader
from scholar_agent.infrastructure.adapters.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from scholar_agent.infrastructure.adapters.sqlite_document_repository import (
    SQLiteDocumentRepository,
)

__all__ = [
    "FAISSRepository",
    "InMemoryStore",
    "LangChainRetriever",
    "LangChainTextChunker",
    "LangGraphRunner",
    "LocalDocumentLibrary",
    "OllamaAdapter",
    "PlaceholderToolExecutor",
    "PyMuPDFLoader",
    "SentenceTransformerEmbedding",
    "SQLiteDocumentRepository",
]
