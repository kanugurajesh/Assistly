"""
Retry logic with exponential backoff for handling transient failures.
"""
import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (ConnectionError, TimeoutError),
    on_retry: Optional[Callable] = None
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry

    Returns:
        Decorated function that retries on failure

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    # Don't retry if this was the last attempt
                    if attempt == max_retries - 1:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    # Call optional retry callback
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.error(f"Retry callback error: {callback_error}")

                    # Wait before retrying
                    time.sleep(delay)

                except Exception as e:
                    # Don't retry non-transient errors
                    logger.error(f"{func.__name__} failed with non-retryable error: {e}")
                    raise

            # This should never be reached, but just in case
            raise last_exception

        return wrapper
    return decorator


class RetryConfig:
    """Configuration for retry behavior."""

    # OpenAI API retries (connection issues, rate limits)
    OPENAI_RETRIES = 3
    OPENAI_BASE_DELAY = 1.0
    OPENAI_MAX_DELAY = 30.0
    OPENAI_EXCEPTIONS = (ConnectionError, TimeoutError, Exception)

    # Qdrant retries (connection issues)
    QDRANT_RETRIES = 3
    QDRANT_BASE_DELAY = 0.5
    QDRANT_MAX_DELAY = 10.0
    QDRANT_EXCEPTIONS = (ConnectionError, TimeoutError)

    # Embedding generation retries (model issues)
    EMBEDDING_RETRIES = 2
    EMBEDDING_BASE_DELAY = 1.0
    EMBEDDING_MAX_DELAY = 5.0
    EMBEDDING_EXCEPTIONS = (RuntimeError, ValueError, TypeError)


def retry_openai_call(func: Callable) -> Callable:
    """
    Decorator specifically for OpenAI API calls.
    Uses conservative retry settings appropriate for API calls.
    """
    return retry_with_backoff(
        max_retries=RetryConfig.OPENAI_RETRIES,
        base_delay=RetryConfig.OPENAI_BASE_DELAY,
        max_delay=RetryConfig.OPENAI_MAX_DELAY,
        exceptions=RetryConfig.OPENAI_EXCEPTIONS
    )(func)


def retry_qdrant_call(func: Callable) -> Callable:
    """
    Decorator specifically for Qdrant operations.
    Uses faster retries appropriate for database calls.
    """
    return retry_with_backoff(
        max_retries=RetryConfig.QDRANT_RETRIES,
        base_delay=RetryConfig.QDRANT_BASE_DELAY,
        max_delay=RetryConfig.QDRANT_MAX_DELAY,
        exceptions=RetryConfig.QDRANT_EXCEPTIONS
    )(func)


def retry_embedding_generation(func: Callable) -> Callable:
    """
    Decorator specifically for embedding generation.
    Uses quick retries for local model issues.
    """
    return retry_with_backoff(
        max_retries=RetryConfig.EMBEDDING_RETRIES,
        base_delay=RetryConfig.EMBEDDING_BASE_DELAY,
        max_delay=RetryConfig.EMBEDDING_MAX_DELAY,
        exceptions=RetryConfig.EMBEDDING_EXCEPTIONS
    )(func)


class RetryCounter:
    """Track retry statistics for monitoring."""

    def __init__(self):
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_after_retries = 0

    def record_attempt(self, attempt_number: int, success: bool):
        """Record a retry attempt."""
        self.total_attempts += 1
        if success and attempt_number > 0:
            self.successful_retries += 1
        elif not success:
            self.failed_after_retries += 1

    def get_stats(self) -> dict:
        """Get retry statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful_retries": self.successful_retries,
            "failed_after_retries": self.failed_after_retries,
            "retry_success_rate": (
                self.successful_retries / self.total_attempts
                if self.total_attempts > 0
                else 0.0
            )
        }


# Global retry counter
_retry_counter = RetryCounter()


def get_retry_stats() -> dict:
    """Get global retry statistics."""
    return _retry_counter.get_stats()
