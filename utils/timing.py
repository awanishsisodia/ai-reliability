"""
Timing and performance measurement utilities.

This module provides tools for measuring and enforcing performance
requirements, particularly the 50ms latency budget for real-time grounding.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class Timer:
    """High-precision timer for performance measurement."""
    
    def __init__(self, name: str = "operation"):
        """
        Initialize timer.
        
        Args:
            name: Name for the timer operation
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
        self.duration_ms = None
    
    def stop(self) -> float:
        """
        Stop the timer and return duration in milliseconds.
        
        Returns:
            Duration in milliseconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        return self.duration_ms
    
    def __enter__(self) -> Timer:
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
        logger.info(
            "timer_completed",
            operation=self.name,
            duration_ms=self.duration_ms
        )


class LatencyBudgetExceeded(Exception):
    """Raised when operation exceeds latency budget."""
    
    def __init__(self, operation: str, budget_ms: float, actual_ms: float):
        super().__init__(
            f"Operation '{operation}' exceeded latency budget: "
            f"{actual_ms:.2f}ms > {budget_ms:.2f}ms"
        )
        self.operation = operation
        self.budget_ms = budget_ms
        self.actual_ms = actual_ms


@contextmanager
def latency_budget(budget_ms: float, operation: str = "operation"):
    """
    Context manager that enforces latency budget.
    
    Args:
        budget_ms: Maximum allowed time in milliseconds
        operation: Name of the operation being timed
        
    Yields:
        Timer instance
        
    Raises:
        LatencyBudgetExceeded: If budget is exceeded
    """
    timer = Timer(operation)
    with timer:
        yield timer
    
    if timer.duration_ms and timer.duration_ms > budget_ms:
        raise LatencyBudgetExceeded(operation, budget_ms, timer.duration_ms)


def measure_performance(operation_name: str) -> Callable:
    """
    Decorator for measuring function performance.
    
    Args:
        operation_name: Name for the operation in logs
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            timer = Timer(operation_name)
            with timer:
                result = func(*args, **kwargs)
            
            logger.info(
                "function_performance",
                operation=operation_name,
                duration_ms=timer.duration_ms,
                function=func.__name__
            )
            
            return result
        return wrapper
    return decorator


class PerformanceTracker:
    """Tracks performance metrics across multiple operations."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize performance tracker.
        
        Args:
            window_size: Number of recent operations to track
        """
        self.window_size = window_size
        self.measurements: Dict[str, list] = {}
    
    def record_measurement(self, operation: str, duration_ms: float) -> None:
        """
        Record a performance measurement.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
        """
        if operation not in self.measurements:
            self.measurements[operation] = []
        
        self.measurements[operation].append(duration_ms)
        
        # Keep only recent measurements
        if len(self.measurements[operation]) > self.window_size:
            self.measurements[operation] = self.measurements[operation][-self.window_size:]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """
        Get performance statistics for an operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Dictionary with statistics
        """
        if operation not in self.measurements or not self.measurements[operation]:
            return {}
        
        durations = self.measurements[operation]
        return {
            "count": len(durations),
            "mean_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": sorted(durations)[len(durations) // 2],
            "p95_ms": sorted(durations)[int(len(durations) * 0.95)],
            "p99_ms": sorted(durations)[int(len(durations) * 0.99)],
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all tracked operations.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {op: self.get_stats(op) for op in self.measurements.keys()}


# Global performance tracker instance
performance_tracker = PerformanceTracker()
