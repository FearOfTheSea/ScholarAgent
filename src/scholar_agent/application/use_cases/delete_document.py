"""Delete-document use case."""

from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    DeleteDocumentResult,
)
from scholar_agent.application.input_ports.document_library import DeleteDocument
from scholar_agent.application.output_ports.document_library import IDocumentLibrary
from scholar_agent.application.output_ports.vector_store import IVectorStore
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.repositories.document_repository import DocumentRepository


class DeleteDocumentUseCase(DeleteDocument):
    """Deletes a source PDF and all local derived data."""

    def __init__(
        self,
        document_repository: DocumentRepository,
        document_library: IDocumentLibrary,
        vector_store: IVectorStore,
    ) -> None:
        self._document_repository = document_repository
        self._document_library = document_library
        self._vector_store = vector_store

    def execute(self, request: DeleteDocumentRequest) -> DeleteDocumentResult:
        """Delete a document and report the completed action."""
        document = self._document_repository.get_by_id(request.document_id)
        if document is None:
            raise DocumentNotFoundError(request.document_id.value)
        self._vector_store.delete_document(request.document_id)
        self._document_library.delete(request.document_id)
        self._document_repository.delete(request.document_id)
        return DeleteDocumentResult(document_id=request.document_id, deleted=True)
