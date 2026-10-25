from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseRerankProvider(ABC):
    """Abstract Base Class for Cross-Encoder Rerank providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name of the reranker model."""
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Reranks a list of document strings against a query.

        Args:
            query: Search query string
            documents: List of document/chunk text passages
            top_k: Optional number of top results to return

        Returns:
            List of tuples (original_index, relevance_score) sorted by score descending.
        """
        pass
