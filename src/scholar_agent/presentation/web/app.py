"""Streamlit user interface for the local study library."""

import subprocess
import sys
from pathlib import Path

import streamlit as st

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
    StudyAgentAnswerResult,
    StudyAgentFlashcardsResult,
    StudyAgentQuizResult,
    StudyAgentSummaryResult,
    StudyAgentTaskResult,
)
from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    IngestDocumentRequest,
)
from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.dtos.tutor import (
    ContinueStudySessionRequest,
    StartStudySessionRequest,
    TutorActivity,
)
from scholar_agent.config.settings import Settings
from scholar_agent.domain.entities.document import Document
from scholar_agent.domain.entities.study_session import (
    LearnerLevel,
    SourceReference,
    StudyMode,
)
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
            "Navigation",
            ("Library", "Adaptive Tutor", "Ask Study Agent"),
        )

    st.title("ScholarAgent")
    if page == "Library":
        _render_library(container, documents)
    elif page == "Ask Study Agent":
        _render_ask_study_agent(container, documents)
    else:
        _render_adaptive_tutor(container, documents)


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


def _render_ask_study_agent(
    container: Container,
    documents: tuple[Document, ...],
) -> None:
    st.subheader("Ask Study Agent")
    document = _select_document(documents, "Study this document")
    if document is None:
        return
    prompt = st.text_area(
        "What would you like to do?",
        placeholder=(
            "Ask a question, request a summary, create a quiz or flashcards, "
            "or describe a broader study goal."
        ),
    )
    if st.button("Ask Study Agent"):
        try:
            with st.spinner("Choosing and running the right study tools..."):
                result = container.ask_study_agent_use_case().execute(
                    AskStudyAgentRequest(
                        prompt=prompt,
                        document_id=document.identifier,
                    )
                )
            _render_agent_response(result)
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_adaptive_tutor(
    container: Container,
    documents: tuple[Document, ...],
) -> None:
    st.subheader("Adaptive Document Tutor")
    st.caption("A persistent, cited learning path that adapts to demonstrated mastery.")
    session_id = st.session_state.get("adaptive_tutor_session_id")
    if not isinstance(session_id, str):
        _render_start_session(container, documents)
        return
    try:
        result = container.get_study_session_use_case().execute(session_id)
    except ValueError:
        st.session_state.pop("adaptive_tutor_session_id", None)
        st.warning("That session is no longer available.")
        return

    map_column, tutor_column = st.columns([2, 3])
    with map_column:
        _render_document_map(result.session.brief)
        st.markdown("### Mastery")
        progress_by_id = {item.objective_id: item for item in result.progress}
        for objective in result.session.brief.objectives:
            progress = progress_by_id[objective.identifier]
            st.write(f"**{objective.title}** · {progress.label.value.title()}")
            st.progress(progress.percentage)
        if st.button("End and delete session"):
            container.delete_study_session_use_case().execute(session_id)
            st.session_state.pop("adaptive_tutor_session_id", None)
            st.session_state.pop("adaptive_tutor_activity", None)
            st.rerun()

    with tutor_column:
        st.markdown(f"### {result.session.mode.value.title()} session")
        st.caption(result.session.goal)
        for turn in result.session.turns:
            with st.chat_message("user"):
                st.write(turn.learner_message)
            with st.chat_message("assistant"):
                st.markdown(turn.tutor_message)
                _render_source_references(turn.citations)
        pending = st.session_state.get("adaptive_tutor_activity")
        if isinstance(pending, TutorActivity):
            with st.chat_message("assistant"):
                st.markdown(pending.message)
                _render_source_references(pending.citations)
            st.session_state.pop("adaptive_tutor_activity", None)
        message = st.chat_input(
            "Answer, ask for a hint, request an explanation, or ask for a recap"
        )
        if message:
            try:
                with st.spinner("Assessing, verifying evidence, and adapting..."):
                    turn_result = container.continue_study_session_use_case().execute(
                        ContinueStudySessionRequest(session_id, message)
                    )
                st.session_state["adaptive_tutor_activity"] = turn_result.activity
                st.rerun()
            except (RuntimeError, ValueError) as error:
                st.error(str(error))


def _render_start_session(
    container: Container,
    documents: tuple[Document, ...],
) -> None:
    document = _select_document(documents, "Tutor document")
    if document is None:
        return
    with st.form("start-adaptive-session"):
        goal = st.text_input(
            "Learning goal",
            value="Understand the document and retain its key ideas.",
        )
        left, middle, right = st.columns(3)
        level = left.selectbox(
            "Level",
            tuple(LearnerLevel),
            index=1,
            format_func=lambda value: value.value.title(),
        )
        mode = middle.selectbox(
            "Mode",
            tuple(StudyMode),
            format_func=lambda value: value.value.title(),
        )
        minutes = right.number_input(
            "Minutes",
            min_value=5,
            max_value=240,
            value=30,
            step=5,
        )
        submitted = st.form_submit_button("Build cited learning path")
    if submitted:
        try:
            with st.spinner("Mapping concepts, objectives, and source evidence..."):
                result = container.start_study_session_use_case().execute(
                    StartStudySessionRequest(
                        document_id=document.identifier,
                        goal=goal,
                        learner_level=level,
                        mode=mode,
                        target_minutes=int(minutes),
                    )
                )
            st.session_state["adaptive_tutor_session_id"] = result.session.identifier
            st.session_state["adaptive_tutor_activity"] = result.activity
            st.rerun()
        except (RuntimeError, ValueError) as error:
            st.error(str(error))


def _render_document_map(brief: object) -> None:
    from scholar_agent.domain.entities.study_session import DocumentBrief

    if not isinstance(brief, DocumentBrief):
        return
    st.markdown("### Document map")
    st.write(brief.synopsis)
    graph_lines = ["digraph concepts {", 'rankdir="LR";']
    for concept in brief.concepts:
        safe_id = concept.identifier.replace('"', "")
        safe_label = concept.label.replace('"', "'")
        graph_lines.append(f'"{safe_id}" [label="{safe_label}"];')
        for prerequisite in concept.prerequisite_ids:
            safe_prerequisite = prerequisite.replace('"', "")
            graph_lines.append(f'"{safe_prerequisite}" -> "{safe_id}";')
    graph_lines.append("}")
    st.graphviz_chart("\n".join(graph_lines))
    with st.expander("Learning objectives"):
        for objective in brief.objectives:
            st.markdown(f"**{objective.title}** — {objective.description}")
            _render_source_references(objective.citations)
    with st.expander("Glossary and misconceptions"):
        for term in brief.glossary:
            st.markdown(f"**{term.term}:** {term.definition}")
        if brief.misconceptions:
            st.markdown("**Watch for:**")
            for misconception in brief.misconceptions:
                st.write(f"- {misconception}")


def _render_source_references(
    references: tuple[SourceReference, ...],
) -> None:
    if not references:
        return
    with st.expander(f"Evidence ({len(references)})"):
        for reference in references:
            st.caption(f"Page {reference.page_number} · chunk {reference.chunk_id}")
            st.write(reference.excerpt)


def _render_agent_response(result: AskStudyAgentResult) -> None:
    if result.plan:
        selected_tasks = ", ".join(
            step.task.value.replace("_", " ").title() for step in result.plan
        )
        st.caption(f"Agent selected: {selected_tasks}")
    if result.message:
        st.info(result.message)
    for notice in result.notices:
        st.info(notice)
    for task_result in result.results:
        _render_task_result(task_result)
    if result.errors:
        st.warning("Some selected tasks could not be completed:")
        for error in result.errors:
            task_name = error.task.value.replace("_", " ").title()
            st.write(f"- **{task_name}:** {error.message}")


def _render_task_result(result: StudyAgentTaskResult) -> None:
    if isinstance(result, StudyAgentAnswerResult):
        st.markdown("### Answer")
        st.write(result.answer)
        _render_citations(result.citations)
    elif isinstance(result, StudyAgentSummaryResult):
        st.markdown("### Summary")
        st.write(result.summary)
    elif isinstance(result, StudyAgentQuizResult):
        st.markdown("### Quiz")
        for index, question in enumerate(result.questions, start=1):
            with st.expander(f"Question {index}: {question.prompt}"):
                st.write(question.answer)
    elif isinstance(result, StudyAgentFlashcardsResult):
        st.markdown("### Flashcards")
        for card in result.cards:
            with st.expander(card.front):
                st.write(card.back)


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
