"""Document-library endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from scholar_agent.application.dtos.documents import (
    DeleteDocumentRequest,
    IngestDocumentRequest,
)
from scholar_agent.domain.exceptions.document_not_found_error import (
    DocumentNotFoundError,
)
from scholar_agent.domain.exceptions.document_processing_error import (
    DocumentProcessingError,
)
from scholar_agent.domain.value_objects.document_id import DocumentId
from scholar_agent.infrastructure.di.container import Container
from scholar_agent.presentation.api.dependencies import get_container
from scholar_agent.presentation.api.models import DocumentResponse
from scholar_agent.presentation.api.serializers import document_response

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: Annotated[UploadFile, File()],
    container: Annotated[Container, Depends(get_container)],
) -> DocumentResponse:
    """Store and index one local PDF."""
    try:
        result = container.ingest_document_use_case().execute(
            IngestDocumentRequest(
                original_filename=file.filename or "",
                content=await file.read(),
            ),
        )
    except DocumentProcessingError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)
        ) from error
    return document_response(result.document)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    container: Annotated[Container, Depends(get_container)],
) -> list[DocumentResponse]:
    """List PDFs in the local library."""
    result = container.list_documents_use_case().execute()
    return [document_response(document) for document in result.documents]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    container: Annotated[Container, Depends(get_container)],
) -> None:
    """Delete one local PDF and its index data."""
    try:
        container.delete_document_use_case().execute(
            DeleteDocumentRequest(document_id=DocumentId(document_id)),
        )
    except DocumentNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(error)
        ) from error
