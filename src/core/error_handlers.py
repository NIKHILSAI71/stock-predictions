"""
Error Handlers
Production-grade error handling that sanitizes sensitive information.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import logging
import traceback
import sys

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation errors (422 Unprocessable Entity).

    Sanitizes error details in production to avoid information disclosure.

    Args:
        request: The request that caused the error
        exc: RequestValidationError exception

    Returns:
        JSONResponse with sanitized error details
    """
    # Get environment from app state
    environment = getattr(request.app.state, "environment", "production")

    # Log the validation error
    logger.warning(
        f"Validation error: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'} "
        f"- {len(exc.errors())} error(s)"
    )

    # Detailed errors for development, sanitized for production
    if environment == "development":
        error_details = exc.errors()
    else:
        # In production, provide minimal error information
        error_details = [
            {
                "field": ".".join(str(loc) for loc in err.get("loc", [])),
                "message": err.get("msg", "Invalid value")
            }
            for err in exc.errors()
        ]

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error": "Validation Error",
            "message": "Invalid request parameters. Please check your input.",
            "details": error_details if environment != "production" else None,
            "errors": error_details,
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions.

    Prevents stack traces from being exposed to clients in production.

    Args:
        request: The request that caused the error
        exc: The uncaught exception

    Returns:
        JSONResponse with error message
    """
    # Get environment from app state
    environment = getattr(request.app.state, "environment", "production")

    # Log the full exception with traceback
    logger.error(
        f"Unhandled exception: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'} "
        f"- {type(exc).__name__}: {str(exc)}",
        exc_info=True,  # Include full traceback in logs
    )

    # Generate request ID if available
    request_id = getattr(request.state, "request_id", None)

    # Development: Return full error details
    if environment == "development":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": "Internal Server Error",
                "message": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc().split("\n"),
                "request_id": request_id,
            },
        )

    # Production: Return generic error message
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "Internal Server Error",
            "message": "An internal error occurred. Please try again later or contact support.",
            "request_id": request_id,
        },
    )


async def http_exception_handler(request: Request, exc):
    """
    Handle HTTP exceptions from FastAPI.

    Args:
        request: The request that caused the error
        exc: HTTPException

    Returns:
        JSONResponse with error details
    """
    # Log HTTP exceptions (warning level for 4xx, error level for 5xx)
    if exc.status_code >= 500:
        logger.error(
            f"HTTP {exc.status_code}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'} "
            f"- {exc.detail}"
        )
    else:
        logger.warning(
            f"HTTP {exc.status_code}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'} "
            f"- {exc.detail}"
        )

    # Format detail as consistent structure
    if isinstance(exc.detail, dict):
        content = {
            "status": "error",
            **exc.detail,
        }
    else:
        content = {
            "status": "error",
            "message": str(exc.detail),
        }

    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=exc.headers,
    )


async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle Pydantic validation errors.

    Args:
        request: The request that caused the error
        exc: Pydantic ValidationError

    Returns:
        JSONResponse with validation errors
    """
    logger.warning(
        f"Pydantic validation error: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error": "Validation Error",
            "message": "Data validation failed",
            "errors": exc.errors(),
        },
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance
    """
    from fastapi.exceptions import HTTPException

    # Request validation errors (422)
    app.add_exception_handler(RequestValidationError,
                              validation_exception_handler)

    # Pydantic validation errors
    app.add_exception_handler(
        ValidationError, pydantic_validation_exception_handler)

    # HTTP exceptions (4xx, 5xx from FastAPI)
    app.add_exception_handler(HTTPException, http_exception_handler)

    # Catch-all for uncaught exceptions
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Exception handlers registered")
