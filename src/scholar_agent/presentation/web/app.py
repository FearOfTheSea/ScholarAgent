"""Streamlit user interface for the local study library."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

from scholar_agent.application.dtos.agent import PrepareStudySessionRequest
from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    IngestDocumentRequest,
)
from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.study_requests import (
    AnswerQuestionRequest,
    CompareDocumentsRequest,
    GenerateFlashcardsRequest,
    GenerateQuizRequest,
    SummarizeDocumentRequest,
)
from scholar_agent.config.settings import Settings
from scholar_agent.domain.entities.document import Document
from scholar_agent.infrastructure.di.container import Container, build_container


@st.cache_resource
def _container() -> Container:
    """Build one local dependency container for the Streamlit process."""
    return build_container(Settings())


def main() -> None:
    """Render the local study-library UI."""
    st.set_page_config(
        page_title="ScholarAgent", page_icon="ScholarAgent", layout="wide"
    )
    container = _container()
    documents = container.list_documents_use_case().execute().documents

    with st.sidebar:
        st.header("ScholarAgent")
        readiness = container.check_runtime_readiness_use_case().execute()
        if readiness.ollama_available and readiness.model_available:
            st.success("Local model ready")
        else:
            st.warning("Local model unavailable")
        page = st.radio(
            "Study tools",
            (
                "Library",
                "Study Agent",
                "Ask",
                "Summary",
                "Quiz",
                "Flashcards",
                "Compare",
            ),
        )

    st.title("ScholarAgent")
    if page == "Library":
        _render_library(container, documents)
    elif page == "Study Agent":
        _render_study_agent(container, documents)
    elif page == "Ask":
        _render_question_answering(container, documents)
    elif page == "Summary":
        _render_summary(container, documents)
    elif page == "Quiz":
        _render_quiz(container, documents)
    elif page == "Flashcards":
        _render_flashcards(container, documents)
    else:
        _render_comparison(container, documents)


def run() -> None:
    """Launch this module through Streamlit when invoked as a project script."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())],
        check=True,
    )


def _render_library(container: Container, documents: tuple[Document, ...]) -> None:
    st.subheader("Local PDF library")
    uploaded_file = st.file_uploader("Add an English PDF", type=["pdf"])
    if uploaded_file is not None and st.button("Add to library"):
        try:
            with st.spinner("Extracting, embedding, and indexing locally..."):
                result = container.ingest_document_use_case().execute(
                    IngestDocumentRequest(
                        original_filename=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                    ),
                )
            st.success(f"Added {result.document.title}.")
            st.rerun()
        except (RuntimeError, ValueError) as error:
            st.error(str(error))

    if not documents:
        st.info("Upload a PDF to start studying locally.")
        return

    for document in documents:
        left_column, right_column = st.columns([5, 1])
        left_column.write(
            f"**{document.title}** — {document.page_count} pages — {document.source}",
        )
        if right_column.button("Delete", key=f"delete-{document.identifier.value}"):
            container.delete_document_use_case().execute(
                DeleteDocumentRequest(document.identifier),
            )
            st.rerun()


def _render_question_answering(
    container: Container, documents: tuple[Document, ...]
) -> None:
    st.subheader("Ask your PDFs")
    if not _require_documents(documents):
        return
    selected_documents = st.multiselect(
        "Search these documents",
        documents,
        default=list(documents),
        format_func=_document_label,
    )
    question = st.text_area("Question")
    if st.button("Answer"):
        try:
            with st.spinner("Searching local sources and generating an answer..."):
                result = container.answer_question_use_case().execute(
                    AnswerQuestionRequest(
                        question=question,
                        document_ids=tuple(
                            document.identifier for document in selected_documents
                        ),
                    ),
                )
            st.write(result.answer)
            _render_citations(result.citations)
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_study_agent(container: Container, documents: tuple[Document, ...]) -> None:
    st.subheader("Exam preparation agent")
    if not _require_documents(documents):
        return
    selected_documents = st.multiselect(
        "Study these documents",
        documents,
        default=list(documents),
        format_func=_document_label,
    )
    goal = st.text_area(
        "Study goal",
        value="Prepare me for an exam on the most important concepts.",
    )
    question_count = st.slider("Quiz questions", min_value=1, max_value=10, value=5)
    if st.button("Run study agent"):
        if not selected_documents:
            st.error("Select at least one document.")
            return
        try:
            with st.spinner("Planning a study session and running local tools..."):
                result = container.prepare_study_session_use_case().execute(
                    PrepareStudySessionRequest(
                        goal=goal,
                        document_ids=tuple(
                            document.identifier for document in selected_documents
                        ),
                        question_count=question_count,
                    ),
                )
            st.markdown("### Agent plan")
            for index, step in enumerate(result.plan, start=1):
                st.write(f"{index}. **{step.tool_name}** — {step.description}")
            st.markdown("### Study summary")
            st.write(result.summary or "No summary was produced.")
            st.markdown("### Quiz")
            for index, question in enumerate(result.quiz, start=1):
                with st.expander(f"Question {index}: {question.prompt}"):
                    st.write(question.answer)
            st.markdown("### Recommendations")
            for recommendation in result.recommendations:
                st.write(f"- {recommendation}")
            _render_citations(result.citations)
            if result.errors:
                st.warning("Some optional steps were unavailable:")
                for error in result.errors:
                    st.write(f"- {error}")
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_summary(container: Container, documents: tuple[Document, ...]) -> None:
    st.subheader("Summarize a PDF")
    document = _select_document(documents, "Document to summarize")
    if document is not None and st.button("Generate summary"):
        try:
            with st.spinner("Generating a local hierarchical summary..."):
                result = container.summarize_document_use_case().execute(
                    SummarizeDocumentRequest(document.identifier),
                )
            st.write(result.summary)
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_quiz(container: Container, documents: tuple[Document, ...]) -> None:
    st.subheader("Generate a quiz")
    document = _select_document(documents, "Document for the quiz")
    question_count = st.slider("Questions", min_value=1, max_value=10, value=5)
    if document is not None and st.button("Generate quiz"):
        try:
            with st.spinner("Generating structured questions locally..."):
                result = container.generate_quiz_use_case().execute(
                    GenerateQuizRequest(document.identifier, question_count),
                )
            for index, question in enumerate(result.questions, start=1):
                with st.expander(f"Question {index}: {question.prompt}"):
                    st.write(question.answer)
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_flashcards(container: Container, documents: tuple[Document, ...]) -> None:
    st.subheader("Generate flashcards")
    document = _select_document(documents, "Document for flashcards")
    card_count = st.slider("Cards", min_value=1, max_value=20, value=10)
    if document is not None and st.button("Generate flashcards"):
        try:
            with st.spinner("Generating structured flashcards locally..."):
                result = container.generate_flashcards_use_case().execute(
                    GenerateFlashcardsRequest(document.identifier, card_count),
                )
            for card in result.cards:
                with st.expander(card.front):
                    st.write(card.back)
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_comparison(container: Container, documents: tuple[Document, ...]) -> None:
    st.subheader("Compare two PDFs")
    if len(documents) < 2:
        st.info("Add at least two PDFs to compare them.")
        return
    first_document = st.selectbox(
        "First document", documents, format_func=_document_label
    )
    second_document = st.selectbox(
        "Second document",
        documents,
        index=1,
        format_func=_document_label,
    )
    if st.button("Compare"):
        try:
            with st.spinner("Retrieving local evidence for both documents..."):
                result = container.compare_documents_use_case().execute(
                    CompareDocumentsRequest(
                        first_document_id=first_document.identifier,
                        second_document_id=second_document.identifier,
                    ),
                )
            st.write(result.comparison)
            _render_citations(result.citations)
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _select_document(
    documents: tuple[Document, ...],
    label: str,
) -> Document | None:
    if not _require_documents(documents):
        return None
    return st.selectbox(label, documents, format_func=_document_label)


def _require_documents(documents: tuple[Document, ...]) -> bool:
    if documents:
        return True
    st.info("Upload a PDF in the Library first.")
    return False


def _document_label(document: Document) -> str:
    return f"{document.title} ({document.page_count} pages)"


def _render_citations(citations: tuple[RetrievedChunk, ...]) -> None:
    if not citations:
        return
    st.caption("Citations")
    for citation in citations:
        st.write(
            f"- {citation.document_id.value}, page {citation.page_number}, "
            f"chunk {citation.chunk_id}, score {citation.similarity_score:.3f}",
        )


if __name__ == "__main__":
    main()
