"""Application use-case implementations."""

from scholar_agent.application.use_cases.answer_question import AnswerQuestionUseCase
from scholar_agent.application.use_cases.ask_study_agent import AskStudyAgentUseCase
from scholar_agent.application.use_cases.check_runtime_readiness import (
    CheckRuntimeReadinessUseCase,
)
from scholar_agent.application.use_cases.delete_document import DeleteDocumentUseCase
from scholar_agent.application.use_cases.generate_flashcards import (
    GenerateFlashcardsUseCase,
)
from scholar_agent.application.use_cases.generate_quiz import GenerateQuizUseCase
from scholar_agent.application.use_cases.ingest_document import IngestDocumentUseCase
from scholar_agent.application.use_cases.list_documents import ListDocumentsUseCase
from scholar_agent.application.use_cases.summarize_document import (
    SummarizeDocumentUseCase,
)

__all__ = [
    "AnswerQuestionUseCase",
    "AskStudyAgentUseCase",
    "CheckRuntimeReadinessUseCase",
    "DeleteDocumentUseCase",
    "GenerateFlashcardsUseCase",
    "GenerateQuizUseCase",
    "IngestDocumentUseCase",
    "ListDocumentsUseCase",
    "SummarizeDocumentUseCase",
]
