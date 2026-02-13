"""
Stock Algorithms - API Server
FastAPI backend for stock analysis application with production-grade security.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager

# Import production configuration and security
from src.core.config_validator import load_config, print_config_summary
from src.core.logging_config import setup_logging
from src.core.auth_middleware import AuthMiddleware, APIKeyManager
from src.core.rate_limiter import RateLimitMiddleware, InMemoryRateLimiter
from src.core.error_handlers import register_exception_handlers

from src.api.endpoints import (
    stock, technical, fundamental, quantitative, market, ai, signals, accuracy, sentiment
)

# Load and validate configuration
try:
    config = load_config()
except Exception as e:
    print(f"\n❌ FATAL: Configuration validation failed")
    print(f"Error: {str(e)}\n")
    print("Please fix your .env file before starting the server.")
    exit(1)

# Setup logging early
setup_logging(
    log_level=config.LOG_LEVEL,
    log_file=config.LOG_FILE_PATH,
    environment=config.ENVIRONMENT
)

logger = logging.getLogger(__name__)

# Initialize security components
api_key_manager = APIKeyManager(config.api_keys_list)
rate_limiter = InMemoryRateLimiter(config.RATE_LIMIT_PER_MINUTE)


# ============================================
# FASTAPI APPLICATION
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan handler for startup and shutdown.

    Startup:
    - Print configuration summary
    - Start rate limiter cleanup task
    - Validate API keys

    Shutdown:
    - Stop rate limiter cleanup task
    """
    logger.info("=" * 60)
    logger.info("Stock Analysis API Starting Up")
    logger.info("=" * 60)

    # Print configuration summary
    print_config_summary(config)

    # Store config in app state for access by error handlers
    app.state.config = config
    app.state.environment = config.ENVIRONMENT

    # Start rate limiter background tasks
    await rate_limiter.start()
    logger.info("Rate limiter started")

    logger.info("✓ All systems operational")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down gracefully...")
    await rate_limiter.stop()
    logger.info("✓ Shutdown complete")


app = FastAPI(
    title="Stock Analysis API",
    description="Comprehensive Stock Analysis & AI Prediction API with production-grade security",
    version="2.0.0",
    lifespan=lifespan,
    # Disable docs in production
    docs_url="/docs" if config.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if config.ENVIRONMENT != "production" else None,
)

# Register exception handlers FIRST
register_exception_handlers(app)

# Add Rate Limiting Middleware (BEFORE authentication)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
logger.info("Rate limiting middleware registered")

# Add Authentication Middleware (BEFORE CORS)
app.add_middleware(
    AuthMiddleware,
    key_manager=api_key_manager,
    api_key_header=config.API_KEY_HEADER
)
logger.info("Authentication middleware registered")

# CORS Configuration (AFTER authentication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins_list,
    # Disable in production
    allow_credentials=(config.ENVIRONMENT != "production"),
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", config.API_KEY_HEADER],
    max_age=600,  # Cache preflight requests for 10 minutes
)
logger.info(f"CORS configured for {len(config.allowed_origins_list)} origins")

# Include Routers
app.include_router(stock.router, prefix="/api", tags=["Stock Data"])
app.include_router(technical.router, prefix="/api",
                   tags=["Technical Analysis"])
app.include_router(fundamental.router, prefix="/api",
                   tags=["Fundamental Analysis"])
app.include_router(quantitative.router,
                   prefix="/api/quantitative", tags=["Quantitative Analysis"])
app.include_router(market.router, prefix="/api", tags=["Market Analysis"])
app.include_router(signals.router, prefix="/api", tags=["Signals"])
app.include_router(ai.router, prefix="/api", tags=["AI Analysis"])
app.include_router(accuracy.router, prefix="/api", tags=["Model Accuracy"])
app.include_router(sentiment.router, prefix="/api",
                   tags=["Sentiment Analysis"])


@app.get("/health")
async def health_check():
    """
    Health check endpoint (no authentication required).

    Returns system status and rate limiter statistics.
    """
    stats = rate_limiter.get_stats()
    return {
        "status": "healthy",
        "message": "API is running",
        "version": "2.0.0",
        "environment": config.ENVIRONMENT,
        "rate_limiter": {
            "total_requests": stats["total_requests"],
            "throttled_requests": stats["throttled_requests"],
            "active_ips": stats["active_ips"],
        }
    }


@app.get("/")
async def root():
    """
    API root endpoint (no authentication required).

    Returns basic API information and available endpoints.
    """
    return {
        "message": "Stock Analysis API",
        "version": "2.0.0",
        "environment": config.ENVIRONMENT,
        "docs": "/docs" if config.ENVIRONMENT != "production" else None,
        "health": "/health",
        "authentication": {
            "required": True,
            "header": config.API_KEY_HEADER,
            "message": "All /api/* endpoints require authentication"
        },
        "rate_limit": {
            "limit": f"{config.RATE_LIMIT_PER_MINUTE} requests per minute per IP"
        }
    }


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
    )
