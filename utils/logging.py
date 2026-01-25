"""
Structured logging configuration for the AI Reliability Engine.

This module provides production-ready logging with structured output,
correlation IDs, and performance tracking integration.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

# Configure structlog for production use
def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


class LoggingContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger, **context: Any):
        """
        Initialize logging context.
        
        Args:
            logger: Logger instance
            **context: Key-value pairs to add to log context
        """
        self.logger = logger
        self.context = context
        self.bound_logger = None
    
    def __enter__(self) -> structlog.stdlib.BoundLogger:
        """Enter context and bind logger."""
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind logger."""
        if exc_type is not None:
            self.bound_logger.error(
                "exception_in_context",
                exc_info=(exc_type, exc_val, exc_tb)
            )


def log_performance_metrics(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    duration_ms: float,
    metadata: Dict[str, Any] | None = None
) -> None:
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        metadata: Additional metadata to include
    """
    log_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "performance_category": categorize_performance(duration_ms),
    }
    
    if metadata:
        log_data.update(metadata)
    
    logger.info("performance_metric", **log_data)


def categorize_performance(duration_ms: float) -> str:
    """
    Categorize performance based on duration.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        Performance category
    """
    if duration_ms < 10:
        return "excellent"
    elif duration_ms < 25:
        return "good"
    elif duration_ms < 50:
        return "acceptable"
    elif duration_ms < 100:
        return "slow"
    else:
        return "very_slow"


def log_reliability_result(
    logger: structlog.stdlib.BoundLogger,
    result: Dict[str, Any],
    response_length: int,
    processing_time_ms: float
) -> None:
    """
    Log reliability evaluation results.
    
    Args:
        logger: Logger instance
        result: Reliability result data
        response_length: Length of evaluated response
        processing_time_ms: Total processing time
    """
    logger.info(
        "reliability_evaluation_completed",
        score=result.get("score"),
        grounding=result.get("grounding"),
        decision=result.get("decision"),
        response_length=response_length,
        processing_time_ms=processing_time_ms,
        sentence_count=result.get("sentence_count"),
        evidence_count=result.get("evidence_count"),
    )
