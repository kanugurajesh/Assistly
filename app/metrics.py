"""
Metrics collection system for monitoring RAG pipeline performance and costs.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from threading import Lock
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    timestamp: datetime
    response_time: float  # seconds
    tokens_used: int
    search_method: str  # 'vector', 'hybrid', 'keyword'
    num_results: int
    error: bool = False
    error_type: Optional[str] = None
    cached: bool = False


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple queries."""
    total_queries: int = 0
    total_errors: int = 0
    total_tokens: int = 0
    total_response_time: float = 0.0

    # Response times
    min_response_time: float = float('inf')
    max_response_time: float = 0.0

    # Search methods
    search_method_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Errors by type
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0

    # Cost estimation (rough)
    estimated_cost: float = 0.0  # USD

    def avg_response_time(self) -> float:
        """Calculate average response time."""
        return self.total_response_time / self.total_queries if self.total_queries > 0 else 0.0

    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        return (self.total_errors / self.total_queries * 100) if self.total_queries > 0 else 0.0

    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0


class MetricsCollector:
    """
    Collects and aggregates metrics for the RAG pipeline.
    Thread-safe implementation for concurrent requests.
    """

    # Cost estimates (rough approximations in USD)
    COST_PER_1K_TOKENS = {
        'gpt-4o': 0.03,  # Input tokens
        'gpt-4o-mini': 0.015,
        'embedding': 0.0001,
    }

    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum number of query metrics to keep in memory
        """
        self.max_history = max_history
        self.query_history: List[QueryMetrics] = []
        self.aggregate = AggregateMetrics()
        self.lock = Lock()

        logger.info(f"Metrics collector initialized (max history: {max_history})")

    def record_query(
        self,
        query_id: str,
        response_time: float,
        tokens_used: int,
        search_method: str,
        num_results: int,
        error: bool = False,
        error_type: Optional[str] = None,
        cached: bool = False,
        model: str = 'gpt-4o'
    ) -> None:
        """
        Record metrics for a single query.

        Args:
            query_id: Unique identifier for the query
            response_time: Time taken to process query in seconds
            tokens_used: Number of tokens consumed
            search_method: Search method used
            num_results: Number of results returned
            error: Whether an error occurred
            error_type: Type of error if applicable
            cached: Whether result was cached
            model: Model used for the query
        """
        with self.lock:
            # Create query metrics
            metrics = QueryMetrics(
                query_id=query_id,
                timestamp=datetime.now(),
                response_time=response_time,
                tokens_used=tokens_used,
                search_method=search_method,
                num_results=num_results,
                error=error,
                error_type=error_type,
                cached=cached
            )

            # Add to history
            self.query_history.append(metrics)

            # Trim history if needed
            if len(self.query_history) > self.max_history:
                self.query_history.pop(0)

            # Update aggregates
            self.aggregate.total_queries += 1
            self.aggregate.total_tokens += tokens_used
            self.aggregate.total_response_time += response_time

            # Update min/max response times
            self.aggregate.min_response_time = min(
                self.aggregate.min_response_time,
                response_time
            )
            self.aggregate.max_response_time = max(
                self.aggregate.max_response_time,
                response_time
            )

            # Update search method counts
            self.aggregate.search_method_counts[search_method] += 1

            # Update cache statistics
            if cached:
                self.aggregate.cache_hits += 1
            else:
                self.aggregate.cache_misses += 1

            # Update error statistics
            if error:
                self.aggregate.total_errors += 1
                if error_type:
                    self.aggregate.error_counts[error_type] += 1

            # Estimate cost
            cost_per_token = self.COST_PER_1K_TOKENS.get(model, 0.02)
            query_cost = (tokens_used / 1000.0) * cost_per_token
            self.aggregate.estimated_cost += query_cost

            logger.debug(
                f"Recorded metrics for query {query_id}: "
                f"{response_time:.2f}s, {tokens_used} tokens, "
                f"{search_method}, cached={cached}"
            )

    def get_aggregate_metrics(self) -> Dict:
        """
        Get aggregated metrics as a dictionary.

        Returns:
            Dictionary with all aggregate metrics
        """
        with self.lock:
            return {
                "summary": {
                    "total_queries": self.aggregate.total_queries,
                    "total_errors": self.aggregate.total_errors,
                    "error_rate": f"{self.aggregate.error_rate():.2f}%",
                    "total_tokens": self.aggregate.total_tokens,
                    "estimated_cost_usd": f"${self.aggregate.estimated_cost:.4f}",
                },
                "performance": {
                    "avg_response_time": f"{self.aggregate.avg_response_time():.2f}s",
                    "min_response_time": f"{self.aggregate.min_response_time:.2f}s",
                    "max_response_time": f"{self.aggregate.max_response_time:.2f}s",
                },
                "search_methods": dict(self.aggregate.search_method_counts),
                "cache": {
                    "hits": self.aggregate.cache_hits,
                    "misses": self.aggregate.cache_misses,
                    "hit_rate": f"{self.aggregate.cache_hit_rate():.2f}%",
                },
                "errors": dict(self.aggregate.error_counts),
            }

    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """
        Get recent query metrics.

        Args:
            limit: Maximum number of recent queries to return

        Returns:
            List of recent query metrics as dictionaries
        """
        with self.lock:
            recent = self.query_history[-limit:]
            return [
                {
                    "query_id": q.query_id,
                    "timestamp": q.timestamp.isoformat(),
                    "response_time": f"{q.response_time:.2f}s",
                    "tokens_used": q.tokens_used,
                    "search_method": q.search_method,
                    "num_results": q.num_results,
                    "error": q.error,
                    "error_type": q.error_type,
                    "cached": q.cached,
                }
                for q in reversed(recent)
            ]

    def get_performance_percentiles(self) -> Dict[str, float]:
        """
        Calculate response time percentiles.

        Returns:
            Dictionary with P50, P95, P99 response times
        """
        with self.lock:
            if not self.query_history:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

            response_times = sorted(q.response_time for q in self.query_history)
            n = len(response_times)

            def percentile(p: float) -> float:
                idx = int(n * p / 100.0)
                return response_times[min(idx, n - 1)]

            return {
                "p50": percentile(50),
                "p95": percentile(95),
                "p99": percentile(99),
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.query_history.clear()
            self.aggregate = AggregateMetrics()
            logger.info("Metrics reset")


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        logger.debug(f"{self.name} took {self.elapsed:.3f}s")
        return False  # Don't suppress exceptions

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.elapsed is None:
            return time.time() - self.start_time if self.start_time else 0.0
        return self.elapsed


# Global metrics collector instance (singleton pattern)
_global_metrics = None
_global_metrics_lock = Lock()


def get_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance (singleton).
    Thread-safe implementation using double-checked locking.

    Returns:
        MetricsCollector instance
    """
    global _global_metrics

    # Double-checked locking pattern for thread-safe singleton
    if _global_metrics is None:
        with _global_metrics_lock:
            if _global_metrics is None:
                _global_metrics = MetricsCollector()
                logger.info("Global metrics collector instance created")

    return _global_metrics


def reset_metrics() -> None:
    """Reset the global metrics collector."""
    global _global_metrics
    if _global_metrics:
        _global_metrics.reset()
    _global_metrics = None
