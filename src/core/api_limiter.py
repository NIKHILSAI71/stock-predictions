"""
Centralized API Rate Limiter with Circuit Breaker Pattern

This module provides rate limiting and circuit breaker functionality to prevent
cascading failures when calling external APIs (Gemini, StockTwits, etc.).
"""

import asyncio
import time
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Too many failures, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker for external API protection.
    Opens after N failures, closes after M successes.

    Pattern:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: After failure_threshold failures, reject all requests
    - HALF_OPEN: After timeout, allow limited requests to test recovery
    """

    def __init__(self, failure_threshold=5, success_threshold=2, timeout=60):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
            success_threshold: Number of successes in half-open to close
            timeout: Seconds to wait before trying half-open state
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from func

        Raises:
            Exception: If circuit breaker is OPEN or func raises
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception(
                        f"Circuit breaker OPEN. Retry in {int(self.timeout - (time.time() - self.last_failure_time))}s")

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        """Handle successful request."""
        async with self._lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    logger.info(
                        f"Circuit breaker CLOSED after {self.success_count} successes")
                    self.state = CircuitState.CLOSED
                    self.success_count = 0

    async def _on_failure(self):
        """Handle failed request."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                logger.warning(
                    f"Circuit breaker reopening after failure in HALF_OPEN state")
                self.state = CircuitState.OPEN
                self.failure_count = 0
            elif self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit breaker OPEN after {self.failure_count} failures")
                self.state = CircuitState.OPEN


class TokenBucketLimiter:
    """
    Token bucket rate limiter for API throttling.

    Tokens refill at a constant rate. Each request consumes tokens.
    Smoother than fixed-window rate limiting.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket limiter.

        Args:
            rate: Tokens per second refill rate
            capacity: Maximum tokens in bucket
        """
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens=1) -> bool:
        """
        Try to acquire tokens (non-blocking).

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            # Refill tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_token(self, tokens=1, timeout=30.0) -> bool:
        """
        Wait for tokens to be available (blocking with timeout).

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum seconds to wait

        Returns:
            True if tokens acquired, False if timed out
        """
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire(tokens):
                return True
            # Calculate sleep time based on refill rate
            sleep_time = min(tokens / self.rate, 1.0)
            await asyncio.sleep(sleep_time)
        return False


class APICoordinator:
    """
    Centralized coordinator for all external API calls.
    Manages rate limits and circuit breakers per service.

    Usage:
        coordinator = get_api_coordinator()
        result = await coordinator.call_gemini_api(my_async_func, arg1, arg2)
    """

    def __init__(self):
        """Initialize API coordinator with service-specific limiters."""
        # Gemini API: 15 RPM free tier (Google Vertex AI)
        # Rate: 15 requests / 60 seconds = 0.25 requests/second
        self.gemini_limiter = TokenBucketLimiter(rate=15/60, capacity=5)
        self.gemini_breaker = CircuitBreaker(failure_threshold=5, timeout=120)
        self.gemini_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent calls

        # StockTwits: 200 RPH public tier
        self.stocktwits_limiter = TokenBucketLimiter(
            rate=200/3600, capacity=10)

        # Analysis endpoint queue (prevent server overload)
        self.analysis_semaphore = asyncio.Semaphore(
            3)  # Max 3 concurrent analyses

        # Statistics
        self.stats = {
            "gemini_calls": 0,
            "gemini_throttled": 0,
            "gemini_circuit_open": 0,
            "gemini_successes": 0,
            "gemini_failures": 0
        }

    async def call_gemini_api(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute Gemini API call with rate limiting and circuit breaker.

        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from func

        Raises:
            Exception: If rate limited, circuit breaker open, or func fails
        """
        async with self.gemini_semaphore:
            self.stats["gemini_calls"] += 1

            # Wait for rate limit token
            if not await self.gemini_limiter.wait_for_token(timeout=30):
                self.stats["gemini_throttled"] += 1
                logger.warning(
                    "Gemini API rate limit exceeded, request throttled")
                raise Exception(
                    "Gemini API rate limit exceeded. Please try again in 30 seconds.")

            # Execute with circuit breaker
            try:
                result = await self.gemini_breaker.call(func, *args, **kwargs)
                self.stats["gemini_successes"] += 1
                return result
            except Exception as e:
                self.stats["gemini_failures"] += 1
                if "Circuit breaker OPEN" in str(e):
                    self.stats["gemini_circuit_open"] += 1
                    logger.error("Gemini circuit breaker is OPEN")
                raise e

    async def call_stocktwits_api(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute StockTwits API call with rate limiting.

        Args:
            func: Async function to call
            *args, **kwargs: Arguments to pass to func

        Returns:
            Result from func

        Raises:
            Exception: If rate limited or func fails
        """
        if not await self.stocktwits_limiter.wait_for_token(timeout=10):
            logger.warning("StockTwits API rate limit exceeded")
            raise Exception("StockTwits API rate limit exceeded")

        return await func(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get coordinator statistics for monitoring.

        Returns:
            Dictionary with call counts, throttling stats, circuit state
        """
        return {
            **self.stats,
            "gemini_circuit": self.gemini_breaker.state.value,
            "gemini_tokens": self.gemini_limiter.tokens,
            "gemini_failure_count": self.gemini_breaker.failure_count,
            "timestamp": datetime.now().isoformat()
        }


# Global singleton
_coordinator: Optional[APICoordinator] = None


def get_api_coordinator() -> APICoordinator:
    """
    Get or create the global API coordinator singleton.

    Returns:
        APICoordinator instance
    """
    global _coordinator
    if _coordinator is None:
        _coordinator = APICoordinator()
        logger.info("API Coordinator initialized")
    return _coordinator
