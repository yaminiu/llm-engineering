"""Logging configuration utilities."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name (defaults to root logger)
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


class LogContext:
    """Context manager for temporary logging level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        """Initialize with logger and desired level."""
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter context, set new level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, *args):
        """Exit context, restore old level."""
        self.logger.setLevel(self.old_level)
