from fastapi import Request
from fastapi.responses import JSONResponse
from src.core.logging import logger

class BhoomiMitraException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code

async def bhoomimitra_exception_handler(request: Request, exc: BhoomiMitraException):
    """Handle custom application exceptions."""
    logger.error(f"App Error: {exc.message} on path {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": "APPLICATION_ERROR",
                "message": exc.message
            }
        }
    )

async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected server errors."""
    logger.exception(f"Unhandled Server Error on path {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later."
            }
        }
    )
