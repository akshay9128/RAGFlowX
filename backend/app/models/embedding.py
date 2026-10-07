from typing import List
from pydantic import BaseModel, Field
from backend.app.models.chunk import DocumentChunk


class ChunkEmbedding(BaseModel):
    """Represents a text chunk paired with its generated dense embedding vector."""
    chunk: DocumentChunk
    embedding: List[float] = Field(..., description="Dense float vector embedding array")
    embedding_model: str = Field(..., description="Name of the embedding model used")
    dimension: int = Field(..., description="Vector dimensionality")
