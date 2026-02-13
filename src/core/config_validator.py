"""
Configuration Validator
Production-grade configuration validation using Pydantic.
Ensures all required settings are present before server starts.
Supports Docker secrets for production deployments.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional, List
import os
from dotenv import load_dotenv
from pathlib import Path


def read_secret(secret_name: str, default: str = "") -> str:
    """
    Read secret from Docker secrets or environment variable.

    Docker secrets are mounted at /run/secrets/<secret_name>.
    Falls back to environment variable if secret file doesn't exist.

    Args:
        secret_name: Name of the secret (e.g., 'gemini_api_key')
        default: Default value if neither secret nor env var exists

    Returns:
        Secret value or default
    """
    # Try Docker secret first
    secret_path = Path(f"/run/secrets/{secret_name}")
    if secret_path.exists():
        try:
            return secret_path.read_text().strip()
        except Exception as e:
            print(f"⚠️  Warning: Could not read Docker secret '{secret_name}': {e}")

    # Fallback to environment variable (uppercase with underscores)
    env_var_name = secret_name.upper()
    return os.getenv(env_var_name, default)


class AppConfig(BaseSettings):
    """Application configuration with validation"""

    # Required API Keys
    GEMINI_API_KEY: str = Field(..., min_length=10,
                                description="Gemini AI API key")

    # Optional API Keys
    SERPER_API_KEY: Optional[str] = Field(
        None, min_length=10, description="Serper search API key (optional for web search)")

    # Model Configuration
    GEMINI_MODEL: str = Field(
        "gemini-2.0-flash", description="Gemini model version to use")

    # Optional Social Media APIs
    TWITTER_API_KEY: Optional[str] = Field(
        None, description="Twitter API key (optional)")
    REDDIT_CLIENT_ID: Optional[str] = Field(
        None, description="Reddit client ID (optional)")
    REDDIT_CLIENT_SECRET: Optional[str] = Field(
        None, description="Reddit client secret (optional)")

    # Server Configuration
    HOST: str = Field("0.0.0.0", description="Server host")
    PORT: int = Field(8000, ge=1, le=65535, description="Server port")
    ENVIRONMENT: str = Field(
        "development", pattern="^(development|staging|production)$")

    # Security Settings
    API_KEY_HEADER: str = Field(
        "X-API-Key", description="Header name for API key")
    API_KEYS: str = Field(
        "", description="Comma-separated list of valid API keys")
    ALLOWED_ORIGINS: str = Field(
        "http://localhost:3000,http://localhost:8080", description="Comma-separated CORS origins")
    RATE_LIMIT_PER_MINUTE: int = Field(
        60, ge=1, le=10000, description="Rate limit per IP per minute")

    # Redis Configuration (optional for distributed rate limiting)
    REDIS_HOST: str = Field("localhost", description="Redis host")
    REDIS_PORT: int = Field(6379, ge=1, le=65535, description="Redis port")
    REDIS_ENABLED: bool = Field(
        False, description="Enable Redis for rate limiting")

    # Logging Configuration
    LOG_LEVEL: str = Field(
        "INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    LOG_FILE_PATH: str = Field("logs/api.log", description="Log file path")

    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_api_keys(cls, v, info):
        """Ensure API keys are properly configured"""
        if not v or v.startswith("your_") or v.startswith("YOUR_") or v == "":
            raise ValueError(
                f"{info.field_name} not configured properly. Check your .env file.")
        return v

    @field_validator("SERPER_API_KEY")
    @classmethod
    def validate_serper_key(cls, v):
        """Validate Serper API key if provided"""
        if v and (v.startswith("your_") or v.startswith("YOUR_")):
            print("\n⚠️  WARNING: SERPER_API_KEY not configured. Web search features may be limited.\n")
            return None
        return v

    @field_validator("API_KEYS")
    @classmethod
    def validate_client_api_keys(cls, v):
        """Validate client API keys format"""
        if not v or v.strip() == "":
            # Generate a default warning key in development
            import secrets
            default_key = secrets.token_urlsafe(32)
            print(
                f"\n⚠️  WARNING: No API_KEYS configured. Generated development key: {default_key}\n")
            return default_key
        return v

    @field_validator("ALLOWED_ORIGINS")
    @classmethod
    def validate_allowed_origins(cls, v, info):
        """Ensure production environments have restricted CORS"""
        # Note: In Pydantic v2, we can't access other field values during validation
        # This validation will be done in load_config() instead
        return v

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS into a list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def api_keys_list(self) -> List[str]:
        """Parse API_KEYS into a list"""
        return [key.strip() for key in self.API_KEYS.split(",") if key.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


def load_config() -> AppConfig:
    """
    Load and validate application configuration.
    Reads from Docker secrets (if available) or environment variables.

    Raises:
        ValidationError: If required configuration is missing or invalid

    Returns:
        AppConfig: Validated configuration object
    """
    # Load environment variables from .env file (development)
    load_dotenv()

    # Override with Docker secrets if available (production)
    secrets_to_load = {
        'GEMINI_API_KEY': 'gemini_api_key',
        'SERPER_API_KEY': 'serper_api_key',
        'API_KEYS': 'api_keys',
    }

    for env_var, secret_name in secrets_to_load.items():
        secret_value = read_secret(secret_name)
        if secret_value:
            os.environ[env_var] = secret_value

    try:
        config = AppConfig()

        # Additional validation for production CORS settings
        if config.ENVIRONMENT == "production" and ("*" in config.ALLOWED_ORIGINS or "localhost" in config.ALLOWED_ORIGINS.lower()):
            raise ValueError(
                "CRITICAL: Cannot use '*' or 'localhost' in ALLOWED_ORIGINS for production environment. "
                "Specify exact domains (e.g., 'https://yourdomain.com')"
            )

        return config
    except Exception as e:
        print(f"\n❌ Configuration Error: {str(e)}\n")
        print("Please check your .env file or Docker secrets and ensure all required variables are set.")
        print("See .env.example for reference.\n")
        raise


def print_config_summary(config: AppConfig):
    """Print configuration summary at startup"""
    print("\n" + "="*60)
    print("   Stock Analysis API - Configuration Summary")
    print("="*60)
    print(f"Environment:        {config.ENVIRONMENT}")
    print(f"Host:               {config.HOST}:{config.PORT}")
    print(f"CORS Origins:       {len(config.allowed_origins_list)} configured")
    print(f"Rate Limit:         {config.RATE_LIMIT_PER_MINUTE} req/min per IP")
    print(f"Redis Enabled:      {config.REDIS_ENABLED}")
    print(f"Log Level:          {config.LOG_LEVEL}")
    print(
        f"API Keys:           {len(config.api_keys_list)} client keys configured")
    print(f"Gemini API:         ✓ Configured (Model: {config.GEMINI_MODEL})")
    print(f"Serper API:         {'✓ Configured' if config.SERPER_API_KEY else '✗ Not configured (web search limited)'}")

    if config.ENVIRONMENT == "production":
        print("\n⚠️  PRODUCTION MODE - Security features active")
        print("   - API key authentication required")
        print("   - Rate limiting enforced")
        print("   - CORS restricted to allowed origins")
        print("   - Error details hidden from clients")
    else:
        print(
            f"\n🔧 {config.ENVIRONMENT.upper()} MODE - Enhanced debugging enabled")

    print("="*60 + "\n")
