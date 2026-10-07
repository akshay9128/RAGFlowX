from typing import List, Optional
from pydantic import BaseModel, Field
from backend.app.models.document import DocumentPage, DocumentMetadata


class DocumentUploadResponse(BaseModel):
    """API response schema after a successful document upload."""
    document_id: str = Field(..., description="Unique UUID assigned to the document")
    filename: str = Field(..., description="Uploaded document filename")
    total_pages: int = Field(..., description="Number of pages extracted")
    file_size: int = Field(..., description="Size of file in bytes")
    message: str = Field(default="Document uploaded and processed successfully")


class DocumentListItem(BaseModel):
    """Summary item for document listing endpoint."""
    document_id: str
    filename: str
    content_type: str
    file_size: int
    total_pages: int
    created_at: str


class DocumentListResponse(BaseModel):
    """Response containing a list of ingested documents."""
    total: int
    documents: List[DocumentListItem]


class DocumentDetailResponse(BaseModel):
    """Detailed response containing document metadata and all extracted pages."""
    metadata: DocumentMetadata
    pages: List[DocumentPage]
