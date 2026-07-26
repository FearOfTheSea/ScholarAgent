"""Evaluate BGE-M3 retrieval on the committed local PDF corpus."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from scholar_agent.config.settings import Settings
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.adapters.faiss_repository import FAISSRepository
from scholar_agent.infrastructure.adapters.langchain_retriever import LangChainRetriever
from scholar_agent.infrastructure.adapters.langchain_text_chunker import (
    LangChainTextChunker,
)
from scholar_agent.infrastructure.adapters.pymupdf_loader import PyMuPDFLoader
from scholar_agent.infrastructure.adapters.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)

ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIRECTORY = ROOT_DIRECTORY / "tests" / "fixtures" / "evaluation"


def main() -> None:
    """Run the semantic retrieval corpus and save a machine-readable report."""
    arguments = _parse_arguments()
    manifest = _read_manifest(arguments.manifest)
    settings = Settings()
    embedding_provider = SentenceTransformerEmbedding(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )
    loader = PyMuPDFLoader()
    chunker = LangChainTextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    with tempfile.TemporaryDirectory(prefix="scholaragent-evaluation-") as directory:
        vector_store = FAISSRepository(Path(directory) / "vectors")
        try:
            document_ids = _index_documents(
                manifest,
                arguments.manifest.parent,
                loader,
                chunker,
                embedding_provider,
                vector_store,
            )
            retriever = LangChainRetriever(embedding_provider, vector_store)
            report = _evaluate_cases(
                manifest,
                document_ids,
                retriever,
                settings.retrieval_top_k,
            )
        finally:
            vector_store.close()

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["failed_cases"]:
        raise SystemExit(1)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_CORPUS_DIRECTORY / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIRECTORY / "data" / "evaluation" / "retrieval.json",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("documents"), list
    ):
        raise ValueError("The evaluation manifest must contain a documents list.")
    return manifest


def _index_documents(
    manifest: dict[str, Any],
    corpus_directory: Path,
    loader: PyMuPDFLoader,
    chunker: LangChainTextChunker,
    embedding_provider: SentenceTransformerEmbedding,
    vector_store: FAISSRepository,
) -> dict[str, DocumentId]:
    document_ids: dict[str, DocumentId] = {}
    for document in manifest["documents"]:
        document_id = DocumentId(str(document["id"]))
        document_ids[document_id.value] = document_id
        path = corpus_directory / str(document["filename"])
        if _file_hash(path) != document["sha256"]:
            raise ValueError(f"The evaluation fixture hash does not match: {path.name}")
        pages = loader.load(document_id, path)
        chunks = chunker.chunk(pages)
        vector_store.add(
            chunks,
            embedding_provider.embed_many(tuple(chunk.content for chunk in chunks)),
        )
    return document_ids


def _evaluate_cases(
    manifest: dict[str, Any],
    document_ids: dict[str, DocumentId],
    retriever: LangChainRetriever,
    limit: int,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for document in manifest["documents"]:
        document_id = document_ids[str(document["id"])]
        for case in document["cases"]:
            citations = retriever.retrieve(
                str(case["question"]),
                limit=limit,
                document_ids=(document_id,),
            )
            expected_page = int(case["expected_page"])
            passed = any(
                citation.page_number == expected_page for citation in citations
            )
            cases.append(
                {
                    "id": case["id"],
                    "document_id": document_id.value,
                    "question": case["question"],
                    "expected_page": expected_page,
                    "passed": passed,
                    "citations": [
                        {
                            "page_number": citation.page_number,
                            "chunk_id": citation.chunk_id,
                            "similarity_score": citation.similarity_score,
                        }
                        for citation in citations
                    ],
                },
            )
    passed_cases = sum(case["passed"] for case in cases)
    return {
        "summary": {
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "failed_cases": len(cases) - passed_cases,
        },
        "cases": cases,
    }


def _file_hash(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


if __name__ == "__main__":
    main()
