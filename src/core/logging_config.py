"""
Logging Configuration
Structured logging with JSON formatting for production.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Any, Dict
import os
import warnings


class JSONFormatter(logging.Formatter):
    """
    Format logs as JSON for production log parsing and analysis.

    Includes timestamp, level, logger name, message, module, function,
    and optional context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add request context if available (set by middleware)
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "symbol"):
            log_data["symbol"] = record.symbol
        if hasattr(record, "client_ip"):
            log_data["client_ip"] = record.client_ip
        if hasattr(record, "method"):
            log_data["method"] = record.method
        if hasattr(record, "path"):
            log_data["path"] = record.path

        return json.dumps(log_data)


class HumanReadableFormatter(logging.Formatter):
    """
    Format logs in human-readable format for development.

    Includes colored output if terminal supports it.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors and human-readable structure.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Add color if terminal supports it
        if sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            level_colored = f"{color}{record.levelname}{reset}"
        else:
            level_colored = record.levelname

        # Format timestamp
        timestamp = datetime.fromtimestamp(
            record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Build log message
        log_message = f"{timestamp} [{level_colored}] {record.name} - {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            log_message += "\n" + self.formatException(record.exc_info)

        return log_message


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/api.log",
    environment: str = "development",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        environment: Environment name (development, staging, production)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set log level
    root_logger.setLevel(log_level.upper())

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level.upper())

    # Use JSON formatter in production, human-readable in development
    if environment == "production":
        console_formatter = JSONFormatter()
    else:
        console_formatter = HumanReadableFormatter()

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (rotating) - always JSON for parsing
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # If can't write to file, log warning but continue
        root_logger.warning(f"Could not create log file {log_file}: {e}")

    # Reduce noise from some libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("rquest").setLevel(logging.ERROR)
    logging.getLogger("primp").setLevel(logging.ERROR)  # Too verbose with HTTP requests
    logging.getLogger("src.data.news_fetcher").setLevel(logging.WARNING)  # Reduce news scraping noise

    # Suppress TensorFlow retracing warnings (informational only, not errors)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    # Suppress Keras warnings about input_shape (deprecated warning)
    logging.getLogger("keras").setLevel(logging.ERROR)

    # Suppress Python warnings from TensorFlow/Keras using warnings module
    warnings.filterwarnings('ignore', category=UserWarning, module='keras')
    warnings.filterwarnings(
        'ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', message='.*input_shape.*')
    warnings.filterwarnings('ignore', message='.*tf.function.*retracing.*')

    # Log startup message
    root_logger.info(
        f"Logging configured: level={log_level}, "
        f"environment={environment}, "
        f"file={log_file}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class RequestContextFilter(logging.Filter):
    """
    Add request context to log records.

    Can be used by middleware to inject request-specific information.
    """

    def __init__(self, request_id: str = None, client_ip: str = None):
        super().__init__()
        self.request_id = request_id
        self.client_ip = client_ip

    def filter(self, record: logging.LogRecord) -> bool:
        if self.request_id:
            record.request_id = self.request_id
        if self.client_ip:
            record.client_ip = self.client_ip
        return True
