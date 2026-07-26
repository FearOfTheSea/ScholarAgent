"""Contracts for the study-assistance actions."""

from abc import ABC, abstractmethod

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
    GenerateFlashcardsResult,
    GenerateQuizResult,
    SummarizeDocumentResult,
)


class AnswerQuestion(ABC):
    """Answers a question based on study material."""

    @abstractmethod
    def execute(self, request: AnswerQuestionRequest) -> AnswerQuestionResult:
        """Answer the requested question."""


class SummarizeDocument(ABC):
    """Summarizes a document."""

    @abstractmethod
    def execute(self, request: SummarizeDocumentRequest) -> SummarizeDocumentResult:
        """Summarize the requested document."""


class CompareDocuments(ABC):
    """Compares two documents."""

    @abstractmethod
    def execute(self, request: CompareDocumentsRequest) -> CompareDocumentsResult:
        """Compare the requested documents."""


class GenerateQuiz(ABC):
    """Generates a quiz from a document."""

    @abstractmethod
    def execute(self, request: GenerateQuizRequest) -> GenerateQuizResult:
        """Generate a quiz."""


class GenerateFlashcards(ABC):
    """Generates flashcards from a document."""

    @abstractmethod
    def execute(
        self,
        request: GenerateFlashcardsRequest,
    ) -> GenerateFlashcardsResult:
        """Generate flashcards."""
