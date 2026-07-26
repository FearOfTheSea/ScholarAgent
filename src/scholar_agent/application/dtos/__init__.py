"""Application data transfer objects."""

from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    DeleteDocumentResult,
    IngestDocumentRequest,
    IngestDocumentResult,
    ListDocumentsResult,
)
from scholar_agent.application.dtos.retrieval import (
    DocumentChunk,
    LoadedPage,
    RetrievedChunk,
)
from scholar_agent.application.dtos.runtime import RuntimeReadinessResult
from scholar_agent.application.dtos.study_requests import (
    AnswerQuestionRequest,
    CompareDocumentsRequest,
    GenerateFlashcardsRequest,
    GenerateQuizRequest,
    SummarizeDocumentRequest,
)
from scholar_agent.application.dtos.study_results import (
    AnswerQuestionResult,
    CompareDocumentsResult,
    Flashcard,
    GenerateFlashcardsResult,
    GenerateQuizResult,
    QuizQuestion,
    SummarizeDocumentResult,
)

__all__ = [
    "AnswerQuestionRequest",
    "AnswerQuestionResult",
    "CompareDocumentsRequest",
    "CompareDocumentsResult",
    "DeleteDocumentRequest",
    "DeleteDocumentResult",
    "DocumentChunk",
    "Flashcard",
    "GenerateFlashcardsRequest",
    "GenerateFlashcardsResult",
    "GenerateQuizRequest",
    "GenerateQuizResult",
    "IngestDocumentRequest",
    "IngestDocumentResult",
    "LoadedPage",
    "ListDocumentsResult",
    "QuizQuestion",
    "RetrievedChunk",
    "RuntimeReadinessResult",
    "SummarizeDocumentRequest",
    "SummarizeDocumentResult",
]
