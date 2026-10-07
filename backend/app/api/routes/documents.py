from fastapi import APIRouter, File, UploadFile, HTTPException, status
from backend.app.schemas.document import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentListItem,
    DocumentDetailResponse
)
from backend.app.services.ingestion.service import ingestion_service
from backend.app.database.document_store import document_store

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document (PDF, TXT, MD)"
)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF or text document, extracts text page-by-page, and stores metadata."""
    processed_doc = await ingestion_service.process_and_ingest(file)
    meta = processed_doc.metadata

    return DocumentUploadResponse(
        document_id=meta.document_id,
        filename=meta.filename,
        total_pages=meta.total_pages,
        file_size=meta.file_size,
        message="Document uploaded, parsed, and metadata preserved successfully"
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all ingested documents"
)
async def list_documents():
    """Returns a list of metadata summaries for all ingested documents."""
    doc_metas = document_store.list_documents()
    items = [
        DocumentListItem(
            document_id=m.document_id,
            filename=m.filename,
            content_type=m.content_type,
            file_size=m.file_size,
            total_pages=m.total_pages,
            created_at=m.created_at
        )
        for m in doc_metas
    ]
    return DocumentListResponse(total=len(items), documents=items)


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Get document details and extracted pages"
)
async def get_document(document_id: str):
    """Retrieves metadata and page-level extracted text for a specific document ID."""
    doc = document_store.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    return DocumentDetailResponse(
        metadata=doc.metadata,
        pages=doc.pages
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an ingested document"
)
async def delete_document(document_id: str):
    """Deletes an ingested document and its raw storage file."""
    deleted = document_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found."
        )
    return {"message": f"Document '{document_id}' deleted successfully"}
