"""
Rate limiting for API calls to prevent quota exhaustion and ensure fair resource usage.
"""
import time
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Tuple, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Thread-safe implementation for multi-user environments.
    """

    def __init__(self, max_calls: int, time_window: int, name: str = "default"):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
            name: Name of this rate limiter (for logging)
        """
        self.max_calls = max_calls
        self.time_window = timedelta(seconds=time_window)
        self.name = name
        self.calls = deque()
        self.lock = Lock()  # Thread-safe

        logger.info(
            f"Rate limiter '{name}' initialized: {max_calls} calls per {time_window}s"
        )

    def allow_request(self) -> Tuple[bool, Optional[int]]:
        """
        Check if a request is allowed under rate limits.

        Returns:
            Tuple of (allowed, wait_time_seconds)
            - If allowed=True, request can proceed, wait_time=None
            - If allowed=False, request blocked, wait_time=seconds until next slot
        """
        with self.lock:
            now = datetime.now()

            # Remove expired calls from the window
            while self.calls and now - self.calls[0] > self.time_window:
                self.calls.popleft()

            # Check if we're under the limit
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True, None

            # Rate limit exceeded - calculate wait time
            oldest_call = self.calls[0]
            wait_until = oldest_call + self.time_window
            wait_seconds = int((wait_until - now).total_seconds()) + 1

            logger.warning(
                f"Rate limit exceeded for '{self.name}': "
                f"{len(self.calls)}/{self.max_calls} calls. Wait {wait_seconds}s"
            )

            return False, wait_seconds

    def get_current_usage(self) -> Tuple[int, int]:
        """
        Get current rate limit usage.

        Returns:
            Tuple of (current_calls, max_calls)
        """
        with self.lock:
            now = datetime.now()
            # Remove expired calls
            while self.calls and now - self.calls[0] > self.time_window:
                self.calls.popleft()

            return len(self.calls), self.max_calls

    def reset(self) -> None:
        """Reset the rate limiter (clear all tracked calls)."""
        with self.lock:
            self.calls.clear()
            logger.info(f"Rate limiter '{self.name}' reset")


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter with separate limits for different operations.
    """

    def __init__(self):
        """Initialize multi-tier rate limiter with different limits."""
        # OpenAI API limits (conservative estimates)
        self.openai_limiter = RateLimiter(
            max_calls=50,  # 50 calls per minute
            time_window=60,
            name="openai"
        )

        # Qdrant limits (more generous)
        self.qdrant_limiter = RateLimiter(
            max_calls=100,  # 100 searches per minute
            time_window=60,
            name="qdrant"
        )

        # Overall query limit (prevents abuse)
        self.query_limiter = RateLimiter(
            max_calls=30,  # 30 queries per minute per instance
            time_window=60,
            name="query"
        )

        # Classification limit (separate from queries)
        self.classification_limiter = RateLimiter(
            max_calls=40,  # 40 classifications per minute
            time_window=60,
            name="classification"
        )

    def check_openai_limit(self) -> Tuple[bool, Optional[int]]:
        """Check if OpenAI API call is allowed."""
        return self.openai_limiter.allow_request()

    def check_qdrant_limit(self) -> Tuple[bool, Optional[int]]:
        """Check if Qdrant search is allowed."""
        return self.qdrant_limiter.allow_request()

    def check_query_limit(self) -> Tuple[bool, Optional[int]]:
        """Check if query is allowed."""
        return self.query_limiter.allow_request()

    def check_classification_limit(self) -> Tuple[bool, Optional[int]]:
        """Check if classification is allowed."""
        return self.classification_limiter.allow_request()

    def get_status(self) -> dict:
        """
        Get status of all rate limiters.

        Returns:
            Dictionary with current usage for each limiter
        """
        return {
            "openai": {
                "current": self.openai_limiter.get_current_usage()[0],
                "max": self.openai_limiter.get_current_usage()[1],
            },
            "qdrant": {
                "current": self.qdrant_limiter.get_current_usage()[0],
                "max": self.qdrant_limiter.get_current_usage()[1],
            },
            "query": {
                "current": self.query_limiter.get_current_usage()[0],
                "max": self.query_limiter.get_current_usage()[1],
            },
            "classification": {
                "current": self.classification_limiter.get_current_usage()[0],
                "max": self.classification_limiter.get_current_usage()[1],
            },
        }


# Global instance (singleton pattern)
_global_rate_limiter = None
_global_rate_limiter_lock = Lock()


def get_rate_limiter() -> MultiTierRateLimiter:
    """
    Get the global rate limiter instance (singleton).
    Thread-safe implementation using double-checked locking.

    Returns:
        MultiTierRateLimiter instance
    """
    global _global_rate_limiter

    # Double-checked locking pattern for thread-safe singleton
    if _global_rate_limiter is None:
        with _global_rate_limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = MultiTierRateLimiter()
                logger.info("Global rate limiter instance created")

    return _global_rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (useful for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None
