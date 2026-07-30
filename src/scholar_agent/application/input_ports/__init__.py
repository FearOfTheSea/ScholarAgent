"""Input-port contracts for application actions."""

from scholar_agent.application.input_ports.document_library import (
    DeleteDocument,
    IngestDocument,
    ListDocuments,
)
from scholar_agent.application.input_ports.runtime import CheckRuntimeReadiness
from scholar_agent.application.input_ports.study_assistant import (
    AnswerQuestion,
    GenerateFlashcards,
    GenerateQuiz,
    SummarizeDocument,
)

__all__ = [
    "AnswerQuestion",
    "CheckRuntimeReadiness",
    "DeleteDocument",
    "GenerateFlashcards",
    "GenerateQuiz",
    "IngestDocument",
    "ListDocuments",
    "SummarizeDocument",
]
