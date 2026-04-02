"""Logging configuration for JetsonZoom."""

import logging
import sys
import os
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",       # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        level_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        log_file: Optional file path for file logging

    Returns:
        Configured logger instance
    """
    env_level = os.getenv("LOG_LEVEL", "").strip().upper()
    if env_level and level == logging.INFO:
        level = getattr(logging, env_level, level)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
