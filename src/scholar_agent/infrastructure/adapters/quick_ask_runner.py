"""Stateless Quick Ask routing without a model-selected document identifier."""

from collections.abc import Mapping

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
    StudyAgentAnswerResult,
    StudyAgentFlashcardsResult,
    StudyAgentPlanStep,
    StudyAgentQuizResult,
    StudyAgentStatus,
    StudyAgentSummaryResult,
    StudyAgentTaskError,
    StudyAgentTaskResult,
    StudyTask,
)
from scholar_agent.application.dtos.retrieval import RetrievedChunk
from scholar_agent.application.output_ports.agent_runner import IAgentRunner
from scholar_agent.application.output_ports.tool_executor import IToolExecutor
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference


class QuickAskRunner(IAgentRunner):
    """Route one ephemeral request through the bounded mission capabilities."""

    def __init__(self, tool_executor: IToolExecutor) -> None:
        self._tool_executor = tool_executor

    def run(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        prompt = request.prompt.strip()
        lowered = prompt.casefold()
        if any(
            word in lowered for word in ("web", "internet", "another pdf", "compare")
        ):
            return AskStudyAgentResult(
                status=StudyAgentStatus.NEEDS_CLARIFICATION,
                plan=(),
                results=(),
                message=(
                    "Quick Ask is grounded in one selected document and cannot "
                    "browse or compare sources."
                ),
            )
        try:
            if self._is_question(prompt):
                return self._question(request)
            return self._materials(request)
        except (RuntimeError, ValueError) as error:
            return AskStudyAgentResult(
                status=StudyAgentStatus.FAILED,
                plan=(),
                results=(),
                errors=(StudyAgentTaskError(StudyTask.BUILD_DOCUMENT_MAP, str(error)),),
                message=str(error),
            )

    def _question(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        document_id = request.document_id.value
        search = self._tool_executor.execute(
            StudyTask.SEMANTIC_SEARCH.value,
            {"document_id": document_id, "query": request.prompt, "limit": 4},
        )
        chunks = _retrieved_chunks(search.get("chunks"), request.document_id)
        if not chunks:
            return AskStudyAgentResult(
                status=StudyAgentStatus.NEEDS_CLARIFICATION,
                plan=(),
                results=(),
                message=(
                    "The selected document does not provide enough evidence for "
                    "that question."
                ),
            )
        brief = self._tool_executor.execute(
            StudyTask.BUILD_DOCUMENT_MAP.value,
            {"document_id": document_id},
        )
        objectives = brief.get("objectives")
        if not isinstance(objectives, list) or not objectives:
            raise ValueError("The selected document map has no objective.")
        first = objectives[0]
        if not isinstance(first, Mapping) or not isinstance(first.get("id"), str):
            raise ValueError("The selected document map returned an invalid objective.")
        explanation = self._tool_executor.execute(
            StudyTask.EXPLAIN_CONCEPT.value,
            {
                "document_id": document_id,
                "objective_id": first["id"],
                "source_chunk_ids": [chunk.chunk_id for chunk in chunks],
                "learner_question": request.prompt,
                "style": "quick answer",
            },
        )
        answer = explanation.get("explanation")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Quick Ask explanation was empty.")
        return AskStudyAgentResult(
            status=StudyAgentStatus.COMPLETED,
            plan=(
                StudyAgentPlanStep(StudyTask.SEMANTIC_SEARCH, "Find cited evidence."),
                StudyAgentPlanStep(
                    StudyTask.BUILD_DOCUMENT_MAP, "Use the cited document map."
                ),
                StudyAgentPlanStep(
                    StudyTask.EXPLAIN_CONCEPT, "Explain the current objective."
                ),
            ),
            results=(StudyAgentAnswerResult(answer.strip(), chunks),),
        )

    def _materials(self, request: AskStudyAgentRequest) -> AskStudyAgentResult:
        lowered = request.prompt.casefold()
        actions: list[tuple[StudyTask, dict[str, object], str]] = []
        if "summary" in lowered or "summarize" in lowered or "overview" in lowered:
            actions.append(
                (StudyTask.SUMMARIZE_DOCUMENT, {}, "Create a cited summary.")
            )
        if "flashcard" in lowered:
            actions.append(
                (
                    StudyTask.GENERATE_FLASHCARDS,
                    {"card_count": 10},
                    "Create cited flashcards.",
                )
            )
        if "quiz" in lowered or "test" in lowered or "exam" in lowered:
            actions.append(
                (
                    StudyTask.GENERATE_QUIZ,
                    {"question_count": request.quiz_count_default},
                    "Create a cited quiz.",
                )
            )
        if not actions:
            actions = [
                (StudyTask.SUMMARIZE_DOCUMENT, {}, "Create a cited summary."),
                (
                    StudyTask.GENERATE_FLASHCARDS,
                    {"card_count": 10},
                    "Create cited flashcards.",
                ),
            ]
        results: list[StudyAgentTaskResult] = []
        errors = []
        notices = []
        for task, arguments, _description in actions:
            try:
                payload = self._tool_executor.execute(
                    task.value,
                    {**arguments, "document_id": request.document_id.value},
                )
                if task is StudyTask.SUMMARIZE_DOCUMENT:
                    results.append(
                        StudyAgentSummaryResult(
                            _text(payload, "summary"),
                            _source_references(payload.get("citations")),
                        )
                    )
                elif task is StudyTask.GENERATE_QUIZ:
                    results.append(_quiz_result(payload))
                else:
                    results.append(_flashcard_result(payload))
                notice = payload.get("notice")
                if isinstance(notice, str) and notice:
                    notices.append(notice)
            except (RuntimeError, ValueError) as error:
                errors.append(StudyAgentTaskError(task, str(error)))
        status = (
            StudyAgentStatus.PARTIAL
            if results and errors
            else StudyAgentStatus.COMPLETED
            if results
            else StudyAgentStatus.FAILED
        )
        return AskStudyAgentResult(
            status=status,
            plan=tuple(
                StudyAgentPlanStep(task, description)
                for task, _, description in actions
            ),
            results=tuple(results),
            notices=tuple(notices),
            errors=tuple(errors),
        )

    @staticmethod
    def _is_question(prompt: str) -> bool:
        lowered = prompt.casefold()
        return prompt.rstrip().endswith("?") or any(
            phrase in lowered for phrase in ("what is", "why ", "how ", "explain")
        )


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Quick Ask field '{key}' must be non-blank.")
    return value.strip()


def _retrieved_chunks(
    value: object, document_id: DocumentId
) -> tuple[RetrievedChunk, ...]:
    if not isinstance(value, list):
        raise ValueError("Semantic search returned an invalid chunk list.")
    chunks: list[RetrievedChunk] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Semantic search returned an invalid chunk.")
        if item.get("document_id") != document_id.value:
            raise ValueError("Semantic search returned a different document.")
        page = item.get("page_number")
        score = item.get("similarity_score")
        if (
            not isinstance(item.get("chunk_id"), str)
            or not isinstance(item.get("content"), str)
            or (page is not None and not isinstance(page, int))
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise ValueError("Semantic search returned invalid citation fields.")
        chunks.append(
            RetrievedChunk(
                document_id=document_id,
                content=item["content"],
                page_number=page,
                section=None,
                chunk_id=item["chunk_id"],
                similarity_score=float(score),
            )
        )
    return tuple(chunks)


def _source_references(value: object) -> tuple[SourceReference, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Material citations must be a list.")
    return tuple(_source_reference(item) for item in value)


def _source_reference(value: object) -> SourceReference:
    if not isinstance(value, Mapping):
        raise ValueError("Material citation must be an object.")
    document_id = value.get("document_id")
    chunk_id = value.get("chunk_id")
    page_number = value.get("page_number")
    excerpt = value.get("excerpt")
    if (
        not isinstance(document_id, str)
        or not isinstance(chunk_id, str)
        or not isinstance(excerpt, str)
        or (page_number is not None and not isinstance(page_number, int))
    ):
        raise ValueError("Material citation has invalid fields.")
    return SourceReference(DocumentId(document_id), chunk_id, page_number, excerpt)


def _quiz_result(payload: Mapping[str, object]) -> StudyAgentQuizResult:
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Quiz result has no questions.")
    from scholar_agent.application.dtos.study_results import QuizQuestion

    parsed = tuple(
        QuizQuestion(
            _text(item, "prompt"),
            _text(item, "answer"),
            _source_references(item.get("citations")),
        )
        for item in questions
        if isinstance(item, Mapping)
    )
    return StudyAgentQuizResult(
        parsed,
        _positive_int(payload, "requested_count"),
        _positive_int(payload, "effective_count"),
        _positive_int(payload, "maximum_count"),
    )


def _flashcard_result(payload: Mapping[str, object]) -> StudyAgentFlashcardsResult:
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise ValueError("Flashcard result has no cards.")
    from scholar_agent.application.dtos.study_results import Flashcard

    parsed = tuple(
        Flashcard(
            _text(item, "front"),
            _text(item, "back"),
            _source_references(item.get("citations")),
        )
        for item in cards
        if isinstance(item, Mapping)
    )
    return StudyAgentFlashcardsResult(
        parsed,
        _positive_int(payload, "requested_count"),
        _positive_int(payload, "effective_count"),
        _positive_int(payload, "maximum_count"),
    )


def _positive_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Quick Ask field '{key}' must be positive.")
    return value
