"""
Logging utility module

Provides centralized logging configuration for the RopeJumpCounter application
with support for console and file logging, log rotation, and date-based naming.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(name: str, log_dir: Path | None = None) -> logging.Logger:
    """Configure application logger with console and optional file output

    Sets up a logger with formatted output to console and optionally to
    rotating log files. Provides different log levels for console (INFO)
    and file (DEBUG) output.

    Args:
        name: Logger name, typically the application or module name
        log_dir: Optional directory for log files. If None, only console logging

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create formatter for consistent log message format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log directory is specified)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate date-based log file name
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f"{name}_{date_str}.log"

        # Set up rotating file handler to manage log file size
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB per file
            backupCount=5           # Keep 5 backup files
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger