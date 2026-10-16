import logging
import sys
import re
from typing import Any
import httpx
from src.config import get_settings


class SecretRedactingFilter(logging.Filter):
    """
    Sanitizing logging filter to prevent API secrets, tokens, or raw credentials
    from being exposed in logs (particularly by httpx / httpcore request loggers).
    Strips sensitive query parameters from logged request URLs while preserving
    the provider hostname, endpoint path, HTTP method, and HTTP status code.
    """
    SENSITIVE_PARAM_PATTERN = re.compile(
        r'([?&]?(?:appid|key|api_key|token|secret|access_token)=)[^&\s"\']+',
        re.IGNORECASE,
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            new_args = []
            for arg in record.args:
                # httpx request logger passes record.args as (method, url, http_version, status_code)
                # where url is an instance of httpx.URL
                if isinstance(arg, httpx.URL):
                    # Remove query string to protect secrets while preserving host and path
                    new_args.append(arg.copy_with(query=None))
                elif isinstance(arg, str):
                    new_args.append(self.SENSITIVE_PARAM_PATTERN.sub(r'\1***', arg))
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)

        if isinstance(record.msg, str):
            record.msg = self.SENSITIVE_PARAM_PATTERN.sub(r'\1***', record.msg)

        return True


def apply_secret_redacting_filter(logger_instance: logging.Logger) -> None:
    """Helper to attach SecretRedactingFilter to a logger if not already present."""
    for existing_filter in logger_instance.filters:
        if isinstance(existing_filter, SecretRedactingFilter):
            return
    logger_instance.addFilter(SecretRedactingFilter())


def setup_logging():
    """Configure structured JSON-friendly logging for the application."""
    settings = get_settings()
    
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(SecretRedactingFilter())

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[stream_handler],
    )
    
    # Set log level for third-party libs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Attach secret redaction to httpx and httpcore loggers
    apply_secret_redacting_filter(logging.getLogger("httpx"))
    apply_secret_redacting_filter(logging.getLogger("httpcore"))
    apply_secret_redacting_filter(logging.getLogger())

    return logging.getLogger(settings.app_name)

logger = setup_logging()
