"""Study-assistance endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from scholar_agent.application.dtos.agent import PrepareStudySessionRequest
from scholar_agent.application.dtos.study_requests import (
    AnswerQuestionRequest,
    CompareDocumentsRequest,
    GenerateFlashcardsRequest,
    GenerateQuizRequest,
    SummarizeDocumentRequest,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import (
    AgentPlanStepResponse,
    AgentQuizQuestionResponse,
    AnswerQuestionRequestModel,
    AnswerQuestionResponse,
    CompareDocumentsRequestModel,
    CompareDocumentsResponse,
    FlashcardResponse,
    GenerateFlashcardsRequestModel,
    GenerateFlashcardsResponse,
    GenerateQuizRequestModel,
    GenerateQuizResponse,
    PrepareStudySessionRequestModel,
    PrepareStudySessionResponse,
    QuizQuestionResponse,
    SummarizeDocumentResponse,
)
from scholar_agent.presentation.api.serializers import citation_response

router = APIRouter(tags=["study"])


@router.post("/agent/study", response_model=PrepareStudySessionResponse)
def prepare_study_session(
    request: PrepareStudySessionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> PrepareStudySessionResponse:
    """Run the goal-oriented local study agent."""
    try:
        result = container.prepare_study_session_use_case().execute(
            PrepareStudySessionRequest(
                goal=request.goal,
                document_ids=tuple(DocumentId(value) for value in request.document_ids),
                question_count=request.question_count,
                session_id=request.session_id,
            ),
        )
    except (RuntimeError, ValueError) as error:
        status_code = (
            status.HTTP_400_BAD_REQUEST
            if isinstance(error, ValueError)
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    return PrepareStudySessionResponse(
        plan=[
            AgentPlanStepResponse(
                tool_name=item.tool_name,
                description=item.description,
            )
            for item in result.plan
        ],
        summary=result.summary,
        quiz=[
            AgentQuizQuestionResponse(prompt=item.prompt, answer=item.answer)
            for item in result.quiz
        ],
        recommendations=list(result.recommendations),
        completed_tools=list(result.completed_tools),
        citations=[citation_response(chunk) for chunk in result.citations],
        errors=list(result.errors),
    )


@router.post("/questions", response_model=AnswerQuestionResponse)
def answer_question(
    request: AnswerQuestionRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> AnswerQuestionResponse:
    """Answer a question using locally indexed document excerpts."""
    try:
        result = container.answer_question_use_case().execute(
            AnswerQuestionRequest(
                question=request.question,
                document_ids=tuple(DocumentId(value) for value in request.document_ids),
            ),
        )
    except (RuntimeError, ValueError) as error:
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
    return SummarizeDocumentResponse(summary=result.summary)


@router.post(
    "/documents/{document_id}/quiz",
    response_model=GenerateQuizResponse,
)
def generate_quiz(
    document_id: str,
    request: GenerateQuizRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> GenerateQuizResponse:
    """Generate a structured quiz from one local document."""
    try:
        result = container.generate_quiz_use_case().execute(
            GenerateQuizRequest(DocumentId(document_id), request.question_count),
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return GenerateQuizResponse(
        questions=[
            QuizQuestionResponse(prompt=item.prompt, answer=item.answer)
            for item in result.questions
        ],
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
    """Generate structured flashcards from one local document."""
    try:
        result = container.generate_flashcards_use_case().execute(
            GenerateFlashcardsRequest(DocumentId(document_id), request.card_count),
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return GenerateFlashcardsResponse(
        cards=[
            FlashcardResponse(front=item.front, back=item.back) for item in result.cards
        ],
    )


@router.post("/comparisons", response_model=CompareDocumentsResponse)
def compare_documents(
    request: CompareDocumentsRequestModel,
    container: Annotated[Container, Depends(get_container)],
) -> CompareDocumentsResponse:
    """Compare evidence independently retrieved from two documents."""
    try:
        result = container.compare_documents_use_case().execute(
            CompareDocumentsRequest(
                first_document_id=DocumentId(request.first_document_id),
                second_document_id=DocumentId(request.second_document_id),
            ),
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error
    return CompareDocumentsResponse(
        comparison=result.comparison,
        citations=[citation_response(chunk) for chunk in result.citations],
    )
