"""Study-assistance endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from scholar_agent.application.dtos.agent import (
    AskStudyAgentRequest,
    AskStudyAgentResult,
    StudyAgentAnswerResult,
    StudyAgentQuizResult,
    StudyAgentSummaryResult,
    StudyAgentTaskResult,
)
from scholar_agent.application.dtos.study_requests import (
    AnswerQuestionRequest,
    GenerateFlashcardsRequest,
    GenerateQuizRequest,
    SummarizeDocumentRequest,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.domain.value_objects.source_reference import SourceReference
from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import (
    AgentAnswerResultResponse,
    AgentFlashcardsResultResponse,
    AgentPlanStepResponse,
    AgentQuizQuestionResponse,
    AgentQuizResultResponse,
    AgentResultResponse,
    AgentSummaryResultResponse,
    AgentTaskErrorResponse,
    AnswerQuestionRequestModel,
    AnswerQuestionResponse,
    AskStudyAgentRequestModel,
    AskStudyAgentResponse,
    FlashcardResponse,
    GenerateFlashcardsRequestModel,
    GenerateFlashcardsResponse,
    GenerateQuizRequestModel,
    GenerateQuizResponse,
    LegacyAgentPlanStepResponse,
    PrepareStudySessionRequestModel,
    PrepareStudySessionResponse,
    QuizQuestionResponse,
    SourceReferenceResponse,
    SummarizeDocumentResponse,
)
from scholar_agent.presentation.api.serializers import citation_response

router = APIRouter(tags=["study"])


@router.post("/agent/requests", response_model=AskStudyAgentResponse)
def ask_study_agent(
    request: AskStudyAgentRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> AskStudyAgentResponse:
    """Plan and execute a free-form request for one selected document."""
    result = _run_agent(
        container,
        AskStudyAgentRequest(
            prompt=request.prompt,
            document_id=DocumentId(request.document_id),
        ),
    )
    return _agent_response(result)


@router.post("/agent/study", response_model=PrepareStudySessionResponse)
def prepare_study_session(
    request: PrepareStudySessionRequestModel,
    response: Response,
    container: Annotated[Container, Depends(get_container)],
) -> PrepareStudySessionResponse:
    """Delegate the deprecated study endpoint to the unified agent."""
    if len(request.document_ids) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly one document is required.",
        )
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</agent/requests>; rel="successor-version"'
    result = _run_agent(
        container,
        AskStudyAgentRequest(
            prompt=request.goal,
            document_id=DocumentId(request.document_ids[0]),
            quiz_count_default=request.question_count,
        ),
    )
    return _legacy_agent_response(result)


@router.post("/questions", response_model=AnswerQuestionResponse)
def answer_question(
    request: AnswerQuestionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> AnswerQuestionResponse:
    """Answer a question using one locally indexed document."""
    try:
        result = container.answer_question_use_case().execute(
            AnswerQuestionRequest(
                question=request.question,
                document_id=DocumentId(request.document_id),
            ),
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return AnswerQuestionResponse(
        answer=result.answer,
        citations=[citation_response(chunk) for chunk in result.citations],
    )


@router.post(
    "/documents/{document_id}/summary",
    response_model=SummarizeDocumentResponse,
)
def summarize_document(
    document_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> SummarizeDocumentResponse:
    """Summarize one local document."""
    try:
        result = container.summarize_document_use_case().execute(
            SummarizeDocumentRequest(DocumentId(document_id)),
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return SummarizeDocumentResponse(
        summary=result.summary,
        citations=[_source_reference_response(item) for item in result.citations],
    )


@router.post(
    "/documents/{document_id}/quiz",
    response_model=GenerateQuizResponse,
)
def generate_quiz(
    document_id: str,
    request: GenerateQuizRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> GenerateQuizResponse:
    """Generate a bounded, structured quiz from one local document."""
    try:
        result = container.generate_quiz_use_case().execute(
            GenerateQuizRequest(DocumentId(document_id), request.question_count),
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return GenerateQuizResponse(
        questions=[
            QuizQuestionResponse(
                prompt=item.prompt,
                answer=item.answer,
                citations=[_source_reference_response(ref) for ref in item.citations],
            )
            for item in result.questions
        ],
        requested_count=result.requested_count,
        effective_count=result.effective_count,
        generated_count=len(result.questions),
        maximum_count=result.maximum_count,
        notice=result.notice,
    )


@router.post(
    "/documents/{document_id}/flashcards",
    response_model=GenerateFlashcardsResponse,
)
def generate_flashcards(
    document_id: str,
    request: GenerateFlashcardsRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> GenerateFlashcardsResponse:
    """Generate bounded, structured flashcards from one local document."""
    try:
        result = container.generate_flashcards_use_case().execute(
            GenerateFlashcardsRequest(DocumentId(document_id), request.card_count),
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return GenerateFlashcardsResponse(
        cards=[
            FlashcardResponse(
                front=item.front,
                back=item.back,
                citations=[_source_reference_response(ref) for ref in item.citations],
            )
            for item in result.cards
        ],
        requested_count=result.requested_count,
        effective_count=result.effective_count,
        generated_count=len(result.cards),
        maximum_count=result.maximum_count,
        notice=result.notice,
    )


def _run_agent(
    container: Container,
    request: AskStudyAgentRequest,
) -> AskStudyAgentResult:
    try:
        return container.ask_study_agent_use_case().execute(request)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error


def _agent_response(result: AskStudyAgentResult) -> AskStudyAgentResponse:
    return AskStudyAgentResponse(
        status=result.status.value,
        plan=[
            AgentPlanStepResponse(
                task=step.task.value,
                description=step.description,
            )
            for step in result.plan
        ],
        results=[_result_response(item) for item in result.results],
        notices=list(result.notices),
        errors=[
            AgentTaskErrorResponse(
                task=error.task.value,
                message=error.message,
            )
            for error in result.errors
        ],
        message=result.message,
    )


def _result_response(result: StudyAgentTaskResult) -> AgentResultResponse:
    if isinstance(result, StudyAgentAnswerResult):
        return AgentAnswerResultResponse(
            task="answer_question",
            answer=result.answer,
            citations=[citation_response(item) for item in result.citations],
        )
    if isinstance(result, StudyAgentSummaryResult):
        return AgentSummaryResultResponse(
            task="summarize_document",
            summary=result.summary,
            citations=[_source_reference_response(ref) for ref in result.citations],
        )
    if isinstance(result, StudyAgentQuizResult):
        return AgentQuizResultResponse(
            task="generate_quiz",
            questions=[
                QuizQuestionResponse(
                    prompt=item.prompt,
                    answer=item.answer,
                    citations=[
                        _source_reference_response(ref) for ref in item.citations
                    ],
                )
                for item in result.questions
            ],
            requested_count=result.requested_count,
            effective_count=result.effective_count,
            generated_count=len(result.questions),
            maximum_count=result.maximum_count,
        )
    return AgentFlashcardsResultResponse(
        task="generate_flashcards",
        cards=[
            FlashcardResponse(
                front=item.front,
                back=item.back,
                citations=[_source_reference_response(ref) for ref in item.citations],
            )
            for item in result.cards
        ],
        requested_count=result.requested_count,
        effective_count=result.effective_count,
        generated_count=len(result.cards),
        maximum_count=result.maximum_count,
    )


def _legacy_agent_response(
    result: AskStudyAgentResult,
) -> PrepareStudySessionResponse:
    summaries = [
        item.summary
        for item in result.results
        if isinstance(item, StudyAgentSummaryResult)
    ]
    quiz_questions = [
        question
        for item in result.results
        if isinstance(item, StudyAgentQuizResult)
        for question in item.questions
    ]
    citations = [
        citation
        for item in result.results
        if isinstance(item, StudyAgentAnswerResult)
        for citation in item.citations
    ]
    recommendations = list(result.notices)
    if result.message:
        recommendations.append(result.message)
    return PrepareStudySessionResponse(
        plan=[
            LegacyAgentPlanStepResponse(
                tool_name=step.task.value,
                description=step.description,
            )
            for step in result.plan
        ],
        summary="\n\n".join(summaries),
        quiz=[
            AgentQuizQuestionResponse(prompt=item.prompt, answer=item.answer)
            for item in quiz_questions
        ],
        recommendations=recommendations,
        completed_tools=[item.task.value for item in result.results],
        citations=[citation_response(item) for item in citations],
        errors=[f"{error.task.value}: {error.message}" for error in result.errors],
        results=[_result_response(item) for item in result.results],
        notices=list(result.notices),
        status=result.status.value,
        message=result.message,
    )


def _source_reference_response(
    reference: SourceReference,
) -> SourceReferenceResponse:
    return SourceReferenceResponse(
        document_id=reference.document_id.value,
        chunk_id=reference.chunk_id,
        page_number=reference.page_number,
        excerpt=reference.excerpt,
    )
