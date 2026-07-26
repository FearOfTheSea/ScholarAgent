"""Retrieval evaluation over the committed English lecture corpus."""

import hashlib
import json
import re
from pathlib import Path

from scholar_agent.application.output_ports.embedding_provider import IEmbeddingProvider
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.faiss_repository import FAISSRepository
from scholar_agent.infrastructure.adapters.langchain_retriever import LangChainRetriever
from scholar_agent.infrastructure.adapters.langchain_text_chunker import (
    LangChainTextChunker,
)
from scholar_agent.infrastructure.adapters.pymupdf_loader import PyMuPDFLoader

CORPUS_DIRECTORY = Path(__file__).parent / "fixtures" / "evaluation"
MANIFEST_PATH = CORPUS_DIRECTORY / "manifest.json"
TOKEN_PATTERN = re.compile(r"[a-z]+")
STOP_WORDS = frozenset({"a", "an", "and", "for", "in", "is", "of", "the", "to", "what"})


class KeywordEmbeddingProvider(IEmbeddingProvider):
    """Deterministic lexical vectors for offline retrieval regression tests."""

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * 256
        for token in TOKEN_PATTERN.findall(text.lower()):
            if token in STOP_WORDS:
                continue
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], byteorder="big") % len(vector)] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return tuple(vector)

    def embed_many(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.embed(text) for text in texts)


def test_english_lecture_corpus_retrieves_expected_page_citations(
    tmp_path: Path,
) -> None:
    """Expected source pages remain discoverable through local adapter composition."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    embedding_provider = KeywordEmbeddingProvider()
    vector_store = FAISSRepository(tmp_path / "vectors")
    loader = PyMuPDFLoader()
    chunker = LangChainTextChunker(chunk_size=800, chunk_overlap=120)
    document_ids: dict[str, DocumentId] = {}

    for document in manifest["documents"]:
        path = CORPUS_DIRECTORY / document["filename"]
        assert _file_hash(path) == document["sha256"]
        document_id = DocumentId(document["id"])
        document_ids[document["id"]] = document_id
        pages = loader.load(document_id, path)
        assert len(pages) == document["page_count"]
        chunks = chunker.chunk(pages)
        vector_store.add(
            chunks,
            embedding_provider.embed_many(tuple(chunk.content for chunk in chunks)),
        )

    retriever = LangChainRetriever(embedding_provider, vector_store)
    for document in manifest["documents"]:
        document_id = document_ids[document["id"]]
        for case in document["cases"]:
            citations = retriever.retrieve(
                case["question"],
                limit=4,
                document_ids=(document_id,),
            )
            assert any(
                citation.document_id == document_id
                and citation.page_number == case["expected_page"]
                for citation in citations
            ), case["id"]


def _file_hash(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
