from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single page extracted from a document."""
    page_number: int = Field(..., description="1-indexed page number", ge=1)
    text: str = Field(..., description="Raw text content of the page")
    character_count: int = Field(..., description="Number of characters in the page text")


class DocumentMetadata(BaseModel):
    """Metadata for an ingested document."""
    document_id: str = Field(..., description="Unique ID of the document")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type or file format")
    file_size: int = Field(..., description="Size of the file in bytes")
    total_pages: int = Field(..., description="Total number of pages extracted")
    source: str = Field(..., description="File path or storage source identifier")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProcessedDocument(BaseModel):
    """Full representation of a processed document including page contents."""
    metadata: DocumentMetadata
    pages: List[DocumentPage]
