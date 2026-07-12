import logging
import sys
from src.config import get_settings

def setup_logging():
    """Configure structured JSON-friendly logging for the application."""
    settings = get_settings()
    
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set log level for third-party libs
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)

    return logging.getLogger(settings.app_name)

logger = setup_logging()
