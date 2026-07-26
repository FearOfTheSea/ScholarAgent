"""List-documents use case."""

from scholar_agent.application.dtos.documents import ListDocumentsResult
from scholar_agent.application.input_ports.document_library import ListDocuments
from scholar_agent.domain.repositories.document_repository import DocumentRepository


class ListDocumentsUseCase(ListDocuments):
    """Returns documents available in the local library."""

    def __init__(self, document_repository: DocumentRepository) -> None:
        self._document_repository = document_repository

    def execute(self) -> ListDocumentsResult:
        """List local documents."""
        return ListDocumentsResult(documents=self._document_repository.list_all())
