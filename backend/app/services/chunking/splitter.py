from typing import List, Tuple
from backend.app.models.chunk import ChunkingConfig


class TextSplitter:
    """
    Recursive text splitter that divides text into chunks of target size with overlap,
    respecting natural text boundaries (paragraphs, lines, sentences, spaces).
    """

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Splits text into chunks of (chunk_text, start_char_idx, end_char_idx).
        """
        if not text or not text.strip():
            return []

        chunks_with_offsets: List[Tuple[str, int, int]] = []
        raw_splits = self._recursive_split(text, self.config.separators)

        # Merge splits into chunks up to chunk_size with chunk_overlap
        current_chunk_pieces: List[str] = []
        current_length = 0

        for piece in raw_splits:
            piece_len = len(piece)

            if current_length + piece_len > self.config.chunk_size and current_chunk_pieces:
                # Flush current accumulated chunk
                chunk_str = "".join(current_chunk_pieces).strip()
                if chunk_str:
                    start_idx = text.find(chunk_str) if chunk_str in text else 0
                    end_idx = start_idx + len(chunk_str)
                    chunks_with_offsets.append((chunk_str, start_idx, end_idx))

                # Retain overlap pieces from the tail of current_chunk_pieces
                overlap_pieces: List[str] = []
                overlap_len = 0
                for p in reversed(current_chunk_pieces):
                    if overlap_len + len(p) <= self.config.chunk_overlap:
                        overlap_pieces.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break

                current_chunk_pieces = overlap_pieces
                current_length = overlap_len

            current_chunk_pieces.append(piece)
            current_length += piece_len

        # Flush final remaining piece
        if current_chunk_pieces:
            chunk_str = "".join(current_chunk_pieces).strip()
            if chunk_str:
                start_idx = text.find(chunk_str) if chunk_str in text else 0
                end_idx = start_idx + len(chunk_str)
                chunks_with_offsets.append((chunk_str, start_idx, end_idx))

        return chunks_with_offsets

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Helper method that recursively splits text by priority separators."""
        if len(text) <= self.config.chunk_size:
            return [text]

        if not separators:
            # Hard character slice fallback if no separators remain
            size = self.config.chunk_size
            return [text[i:i + size] for i in range(0, len(text), size)]

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            size = self.config.chunk_size
            return [text[i:i + size] for i in range(0, len(text), size)]

        parts = text.split(separator)
        final_pieces: List[str] = []

        for i, part in enumerate(parts):
            # Reattach the separator back (except after last part)
            piece = part + (separator if i < len(parts) - 1 else "")
            if len(piece) > self.config.chunk_size:
                # Subdivide over-large piece using next separator
                sub_pieces = self._recursive_split(piece, next_separators)
                final_pieces.extend(sub_pieces)
            else:
                if piece:
                    final_pieces.append(piece)

        return final_pieces
