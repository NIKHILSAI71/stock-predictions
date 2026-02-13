"""
Rate Limiting Middleware
Per-IP rate limiting with sliding window algorithm.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.

    For production with multiple instances, consider using Redis
    for distributed rate limiting across servers.
    """

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60

        # Store request timestamps per IP: {ip: [(timestamp, request_count)]}
        self.request_log: Dict[str,
                               List[Tuple[datetime, int]]] = defaultdict(list)

        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "throttled_requests": 0,
            "unique_ips": set(),
        }

        self.cleanup_task = None
        logger.info(
            f"InMemoryRateLimiter initialized: {requests_per_minute} req/min")

    async def start(self):
        """Start background cleanup task"""
        if not self.cleanup_task:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Rate limiter cleanup task started")

    async def stop(self):
        """Stop background cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Rate limiter cleanup task stopped")

    async def _cleanup_loop(self):
        """
        Background task to clean up expired entries.
        Runs every 60 seconds to prevent memory leaks.
        """
        while True:
            try:
                await asyncio.sleep(60)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limiter cleanup error: {e}")

    def _cleanup_expired(self):
        """Remove expired request entries older than 2 windows"""
        cutoff = datetime.now() - timedelta(seconds=self.window_seconds * 2)
        expired_ips = []

        for ip in list(self.request_log.keys()):
            # Filter out expired entries
            self.request_log[ip] = [
                (ts, cnt) for ts, cnt in self.request_log[ip]
                if ts > cutoff
            ]

            # Remove IP if no entries remain
            if not self.request_log[ip]:
                expired_ips.append(ip)

        for ip in expired_ips:
            del self.request_log[ip]
            self.stats["unique_ips"].discard(ip)

        if expired_ips:
            logger.debug(f"Cleaned up {len(expired_ips)} expired IP entries")

    def is_allowed(self, ip: str) -> Tuple[bool, int, int]:
        """
        Check if request is allowed for this IP.

        Uses sliding window algorithm to count requests within the time window.

        Args:
            ip: Client IP address

        Returns:
            Tuple of (allowed, current_count, limit)
            - allowed: True if request should be allowed
            - current_count: Current number of requests in window
            - limit: Maximum requests allowed
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Update statistics
        self.stats["total_requests"] += 1
        self.stats["unique_ips"].add(ip)

        # Filter requests within the current window
        self.request_log[ip] = [
            (ts, cnt) for ts, cnt in self.request_log[ip]
            if ts > cutoff
        ]

        # Count requests in current window
        current_count = len(self.request_log[ip])

        # Check if limit exceeded
        if current_count >= self.requests_per_minute:
            self.stats["throttled_requests"] += 1
            return False, current_count, self.requests_per_minute

        # Add new request timestamp
        self.request_log[ip].append((now, 1))

        return True, current_count + 1, self.requests_per_minute

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        return {
            "total_requests": self.stats["total_requests"],
            "throttled_requests": self.stats["throttled_requests"],
            "throttle_rate": (
                self.stats["throttled_requests"] /
                self.stats["total_requests"] * 100
                if self.stats["total_requests"] > 0 else 0
            ),
            "unique_ips": len(self.stats["unique_ips"]),
            "active_ips": len(self.request_log),
            "requests_per_minute_limit": self.requests_per_minute,
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Enforces per-IP rate limits and returns 429 Too Many Requests
    when limits are exceeded.
    """

    # Paths exempt from rate limiting
    EXEMPT_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }

    def __init__(self, app, rate_limiter: InMemoryRateLimiter):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            rate_limiter: InMemoryRateLimiter instance
        """
        super().__init__(app)
        self.rate_limiter = rate_limiter
        logger.info("RateLimitMiddleware initialized")

    async def dispatch(self, request: Request, call_next):
        """
        Process each request and enforce rate limits.

        Args:
            request: Incoming request
            call_next: Next middleware in chain

        Returns:
            Response from next middleware or 429 error

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Get client IP (handle X-Forwarded-For from proxies)
        client_ip = self._get_client_ip(request)

        # Check rate limit
        allowed, current, limit = self.rate_limiter.is_allowed(client_ip)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {client_ip}: "
                f"{current}/{limit} requests in last minute "
                f"(path: {request.url.path})"
            )

            # Calculate seconds until reset (approximate)
            reset_seconds = 60

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Maximum {limit} requests per minute allowed.",
                    "retry_after": reset_seconds,
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_seconds),
                    "Retry-After": str(reset_seconds),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        remaining = max(0, limit - current)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = "60"

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.

        Handles X-Forwarded-For header for proxied requests.

        Args:
            request: Incoming request

        Returns:
            Client IP address
        """
        # Check for X-Forwarded-For header (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Check for X-Real-IP header (some proxies use this)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client IP
        if request.client:
            return request.client.host

        return "unknown"
