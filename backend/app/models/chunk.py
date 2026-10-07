from typing import List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to an individual text chunk."""
    chunk_id: str = Field(..., description="Unique UUID for this text chunk")
    document_id: str = Field(..., description="ID of the parent document")
    filename: str = Field(..., description="Original filename of the parent document")
    page_number: int = Field(..., description="Exact 1-indexed page number from which the chunk originated")
    chunk_index: int = Field(..., description="0-indexed sequence position of chunk in the document")
    source: str = Field(..., description="File path reference of the parent document")
    character_count: int = Field(..., description="Total character length of the chunk text")
    start_char_idx: Optional[int] = Field(default=None, description="Start character offset in page text")
    end_char_idx: Optional[int] = Field(default=None, description="End character offset in page text")


class DocumentChunk(BaseModel):
    """Represents a single text chunk with preserved metadata."""
    text: str = Field(..., description="Text content of the chunk")
    metadata: ChunkMetadata


class ChunkingConfig(BaseModel):
    """Configuration options for the text chunker."""
    chunk_size: int = Field(default=500, description="Target character size of each chunk", gt=0)
    chunk_overlap: int = Field(default=50, description="Character overlap between consecutive chunks", ge=0)
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""],
        description="Priority list of separators to break text on"
    )
