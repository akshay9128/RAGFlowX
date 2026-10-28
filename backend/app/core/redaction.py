import re
import logging
from backend.app.config import settings

class LogRedactionFilter(logging.Filter):
    """Logging filter that masks API keys, Authorization headers, X-API-Key, and provider credentials."""

    PATTERNS = [
        re.compile(r'(api[-_]?key["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])', re.IGNORECASE),
        re.compile(r'(Authorization["\']?\s*[:=]\s*["\'])(Bearer\s+[^"\']+)(["\'])', re.IGNORECASE),
        re.compile(r'(X-API-Key["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])', re.IGNORECASE),
        re.compile(r'(AIzaSy[A-Za-z0-9_-]{33})'),  # Gemini API key pattern
        re.compile(r'(sk-[A-Za-z0-9]{20,})'),      # OpenAI API key pattern
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if not settings.LOG_REDACTION_ENABLED:
            return True

        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (self.redact(str(v)) if isinstance(v, str) else v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple((self.redact(str(arg)) if isinstance(arg, str) else arg) for arg in record.args)

        return True

    @classmethod
    def redact(cls, text: str) -> str:
        if not text:
            return text
        redacted = text
        for pattern in cls.PATTERNS:
            if pattern.groups == 3:
                redacted = pattern.sub(r'\1[REDACTED]\3', redacted)
            else:
                redacted = pattern.sub(r'[REDACTED]', redacted)
        return redacted


def setup_log_redaction():
    if not settings.LOG_REDACTION_ENABLED:
        return
    root_logger = logging.getLogger()
    filter_instance = LogRedactionFilter()
    root_logger.addFilter(filter_instance)
    for handler in root_logger.handlers:
        handler.addFilter(filter_instance)
