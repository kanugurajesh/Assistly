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
class ConnectionPoolMetrics:
    """Metrics for connection pool health."""
    service: str  # 'mongodb', 'qdrant', 'openai'
    timestamp: datetime
    active_connections: Optional[int] = None
    pool_size: Optional[int] = None
    wait_time_ms: Optional[float] = None
    connection_errors: int = 0
    slow_operations: int = 0  # Operations slower than threshold


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
        self.connection_pool_history: List[ConnectionPoolMetrics] = []
        self.aggregate = AggregateMetrics()
        self.lock = Lock()

        # Connection pool monitoring thresholds
        self.slow_operation_threshold = 5.0  # seconds

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

    def record_connection_pool_metrics(
        self,
        service: str,
        active_connections: Optional[int] = None,
        pool_size: Optional[int] = None,
        wait_time_ms: Optional[float] = None,
        connection_errors: int = 0,
        response_time: Optional[float] = None
    ) -> None:
        """
        Record connection pool metrics for monitoring.

        Args:
            service: Service name ('mongodb', 'qdrant', 'openai')
            active_connections: Number of active connections
            pool_size: Maximum pool size
            wait_time_ms: Time waiting for connection in milliseconds
            connection_errors: Number of connection errors
            response_time: Response time in seconds (for slow operation detection)
        """
        with self.lock:
            # Detect slow operations
            slow_operations = 0
            if response_time and response_time > self.slow_operation_threshold:
                slow_operations = 1
                logger.warning(
                    f"Slow {service} operation detected: {response_time:.2f}s "
                    f"(threshold: {self.slow_operation_threshold}s) - possible connection pool issue"
                )

            # Create connection pool metrics
            metrics = ConnectionPoolMetrics(
                service=service,
                timestamp=datetime.now(),
                active_connections=active_connections,
                pool_size=pool_size,
                wait_time_ms=wait_time_ms,
                connection_errors=connection_errors,
                slow_operations=slow_operations
            )

            # Add to history
            self.connection_pool_history.append(metrics)

            # Trim history
            if len(self.connection_pool_history) > self.max_history:
                self.connection_pool_history.pop(0)

            # Log warnings for connection pool issues
            if active_connections and pool_size:
                utilization = (active_connections / pool_size) * 100
                if utilization > 80:
                    logger.warning(
                        f"{service} connection pool utilization high: {utilization:.1f}% "
                        f"({active_connections}/{pool_size})"
                    )

            if wait_time_ms and wait_time_ms > 100:
                logger.warning(
                    f"{service} connection wait time high: {wait_time_ms:.1f}ms - "
                    f"consider increasing pool size"
                )

            if connection_errors > 0:
                logger.error(
                    f"{service} connection errors: {connection_errors} - "
                    f"check service health and network connectivity"
                )

    def get_connection_pool_stats(self, service: Optional[str] = None) -> Dict:
        """
        Get connection pool statistics.

        Args:
            service: Optional service name to filter by

        Returns:
            Dictionary with connection pool statistics
        """
        with self.lock:
            if service:
                metrics_list = [m for m in self.connection_pool_history if m.service == service]
            else:
                metrics_list = self.connection_pool_history

            if not metrics_list:
                return {"error": "No connection pool metrics available"}

            # Calculate statistics by service
            stats_by_service = defaultdict(lambda: {
                'total_operations': 0,
                'connection_errors': 0,
                'slow_operations': 0,
                'avg_wait_time_ms': 0.0,
                'max_wait_time_ms': 0.0,
                'samples': []
            })

            for metric in metrics_list:
                svc = metric.service
                stats = stats_by_service[svc]

                stats['total_operations'] += 1
                stats['connection_errors'] += metric.connection_errors
                stats['slow_operations'] += metric.slow_operations

                if metric.wait_time_ms:
                    stats['samples'].append(metric.wait_time_ms)
                    stats['max_wait_time_ms'] = max(stats['max_wait_time_ms'], metric.wait_time_ms)

            # Calculate averages
            result = {}
            for svc, stats in stats_by_service.items():
                if stats['samples']:
                    stats['avg_wait_time_ms'] = sum(stats['samples']) / len(stats['samples'])
                del stats['samples']  # Remove raw samples from output
                result[svc] = stats

            return result

    def reset(self) -> None:
        """Reset all metrics."""
        with self.lock:
            self.query_history.clear()
            self.connection_pool_history.clear()
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
