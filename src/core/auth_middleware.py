"""
API Key Authentication Middleware
Secure API key validation with constant-time comparison.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import hashlib
import hmac
import secrets
from typing import Set
import logging

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Manage API keys with secure hashing.
    Uses SHA-256 hashing to avoid storing plaintext keys in memory.
    """

    def __init__(self, api_keys: list):
        """
        Initialize with list of valid API keys.

        Args:
            api_keys: List of valid API key strings
        """
        self.valid_keys_hashed = self._hash_keys(api_keys)
        logger.info(
            f"APIKeyManager initialized with {len(api_keys)} valid keys")

    def _hash_keys(self, keys: list) -> Set[str]:
        """
        Hash API keys using SHA-256.

        Args:
            keys: List of plaintext API keys

        Returns:
            Set of hashes API key hashes
        """
        return {hashlib.sha256(key.encode()).hexdigest() for key in keys if key}

    def validate_key(self, api_key: str) -> bool:
        """
        Validate API key using constant-time comparison.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        if not api_key:
            return False

        # Hash the provided key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Constant-time comparison to prevent timing attacks
        return any(hmac.compare_digest(key_hash, valid_hash) for valid_hash in self.valid_keys_hashed)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API key validation.

    Checks X-API-Key header on all protected routes.
    Public routes (health, docs) are exempted.
    """

    # Routes that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }

    def __init__(self, app, key_manager: APIKeyManager, api_key_header: str = "X-API-Key"):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application
            key_manager: APIKeyManager instance
            api_key_header: Header name for API key (default: X-API-Key)
        """
        super().__init__(app)
        self.key_manager = key_manager
        self.api_key_header = api_key_header
        logger.info(f"AuthMiddleware initialized (header: {api_key_header})")

    async def dispatch(self, request: Request, call_next):
        """
        Process each request and validate API key.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware or error response
        """
        # Skip authentication for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Extract API key from header or query parameter (for SSE)
        api_key = request.headers.get(self.api_key_header)

        # Fallback to query parameter for EventSource (SSE) endpoints
        # EventSource doesn't support custom headers, so we allow query params for streaming endpoints
        if not api_key and ("-stream/" in request.url.path or request.url.path.endswith("-stream")):
            api_key = request.query_params.get("api_key")

        if not api_key:
            logger.warning(
                f"Missing API key: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'}"
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": f"API key required. Include '{self.api_key_header}' header.",
                    "docs": "/docs"
                },
                headers={"WWW-Authenticate": "ApiKey"},
            )

        # Validate API key
        if not self.key_manager.validate_key(api_key):
            logger.warning(
                f"Invalid API key attempt: {request.method} {request.url.path} "
                f"from {request.client.host if request.client else 'unknown'} "
                f"(key: {api_key[:8]}...)"
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Invalid credentials",
                    "message": "The provided API key is not valid."
                },
            )

        # Add authentication context to request state
        request.state.authenticated = True
        request.state.api_key_hash = hashlib.sha256(
            api_key.encode()).hexdigest()[:16]

        # Log successful authentication (debug level)
        logger.debug(
            f"Authenticated request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Proceed to next middleware
        response = await call_next(request)

        return response


def generate_api_key() -> str:
    """
    Generate a secure random API key.

    Returns:
        URL-safe base64 encoded random string (32 bytes)
    """
    return secrets.token_urlsafe(32)


if __name__ == "__main__":
    # Generate sample API keys for development
    print("Generated API Keys (for development):")
    for i in range(3):
        key = generate_api_key()
        print(f"  Key {i+1}: {key}")
