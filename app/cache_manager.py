"""
Caching layer for embeddings and search results to reduce costs and improve performance.
"""
import hashlib
import logging
from typing import Any, Optional, List, Dict
from cachetools import TTLCache
from threading import Lock

logger = logging.getLogger(__name__)


class RAGCache:
    """
    Multi-level caching for RAG pipeline components.
    Thread-safe implementation with TTL (Time To Live).
    """

    def __init__(
        self,
        embedding_cache_size: int = 1000,
        embedding_ttl: int = 3600,  # 1 hour
        search_cache_size: int = 500,
        search_ttl: int = 1800,  # 30 minutes
        classification_cache_size: int = 500,
        classification_ttl: int = 3600,  # 1 hour
        response_cache_size: int = 500,
        response_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize caching layer.

        Args:
            embedding_cache_size: Maximum number of embeddings to cache
            embedding_ttl: Embedding cache TTL in seconds
            search_cache_size: Maximum number of search results to cache
            search_ttl: Search results cache TTL in seconds
            classification_cache_size: Maximum number of classifications to cache
            classification_ttl: Classification cache TTL in seconds
            response_cache_size: Maximum number of responses to cache
            response_ttl: Response cache TTL in seconds
        """
        # Embedding cache (most expensive to generate)
        self.embedding_cache = TTLCache(maxsize=embedding_cache_size, ttl=embedding_ttl)
        self.embedding_lock = Lock()

        # Search results cache (Qdrant queries)
        self.search_cache = TTLCache(maxsize=search_cache_size, ttl=search_ttl)
        self.search_lock = Lock()

        # Classification cache (OpenAI calls)
        self.classification_cache = TTLCache(maxsize=classification_cache_size, ttl=classification_ttl)
        self.classification_lock = Lock()

        # Response cache (most expensive - OpenAI response generation)
        self.response_cache = TTLCache(maxsize=response_cache_size, ttl=response_ttl)
        self.response_lock = Lock()

        # Statistics
        self.stats = {
            "embedding_hits": 0,
            "embedding_misses": 0,
            "search_hits": 0,
            "search_misses": 0,
            "classification_hits": 0,
            "classification_misses": 0,
            "response_hits": 0,
            "response_misses": 0,
        }

        logger.info(
            f"Cache initialized - Embeddings: {embedding_cache_size} ({embedding_ttl}s), "
            f"Search: {search_cache_size} ({search_ttl}s), "
            f"Classification: {classification_cache_size} ({classification_ttl}s), "
            f"Response: {response_cache_size} ({response_ttl}s)"
        )

    @staticmethod
    def _cache_key(text: str) -> str:
        """
        Generate cache key from text using SHA256 hash.

        Args:
            text: Input text

        Returns:
            SHA256 hash as hex string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @staticmethod
    def _search_cache_key(query: str, top_k: int, score_threshold: float, collection: str) -> str:
        """
        Generate cache key for search results.

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score
            collection: Collection name

        Returns:
            Cache key
        """
        key_components = f"{query}|{top_k}|{score_threshold}|{collection}"
        return hashlib.sha256(key_components.encode('utf-8')).hexdigest()

    # ========== Embedding Cache ==========

    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Cached embedding vector or None if not found
        """
        cache_key = self._cache_key(text)

        with self.embedding_lock:
            embedding = self.embedding_cache.get(cache_key)

            if embedding is not None:
                self.stats["embedding_hits"] += 1
                logger.debug(f"Embedding cache HIT for text: {text[:50]}...")
                return embedding
            else:
                self.stats["embedding_misses"] += 1
                logger.debug(f"Embedding cache MISS for text: {text[:50]}...")
                return None

    def set_cached_embedding(self, text: str, embedding: List[float]) -> None:
        """
        Cache embedding for text.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        cache_key = self._cache_key(text)

        with self.embedding_lock:
            self.embedding_cache[cache_key] = embedding
            logger.debug(f"Cached embedding for text: {text[:50]}...")

    # ========== Search Results Cache ==========

    def get_cached_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        collection: str
    ) -> Optional[List[Dict]]:
        """
        Get cached search results.

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score
            collection: Collection name

        Returns:
            Cached search results or None if not found
        """
        cache_key = self._search_cache_key(query, top_k, score_threshold, collection)

        with self.search_lock:
            results = self.search_cache.get(cache_key)

            if results is not None:
                self.stats["search_hits"] += 1
                logger.debug(f"Search cache HIT for query: {query[:50]}...")
                return results
            else:
                self.stats["search_misses"] += 1
                logger.debug(f"Search cache MISS for query: {query[:50]}...")
                return None

    def set_cached_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        collection: str,
        results: List[Dict]
    ) -> None:
        """
        Cache search results.

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum score
            collection: Collection name
            results: Search results to cache
        """
        cache_key = self._search_cache_key(query, top_k, score_threshold, collection)

        with self.search_lock:
            self.search_cache[cache_key] = results
            logger.debug(f"Cached search results for query: {query[:50]}...")

    # ========== Classification Cache ==========

    def get_cached_classification(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Get cached classification.

        Args:
            text: Ticket text (subject + body)

        Returns:
            Cached classification or None if not found
        """
        cache_key = self._cache_key(text)

        with self.classification_lock:
            classification = self.classification_cache.get(cache_key)

            if classification is not None:
                self.stats["classification_hits"] += 1
                logger.debug(f"Classification cache HIT for text: {text[:50]}...")
                return classification
            else:
                self.stats["classification_misses"] += 1
                logger.debug(f"Classification cache MISS for text: {text[:50]}...")
                return None

    def set_cached_classification(self, text: str, classification: Dict[str, Any]) -> None:
        """
        Cache classification result.

        Args:
            text: Ticket text (subject + body)
            classification: Classification result
        """
        cache_key = self._cache_key(text)

        with self.classification_lock:
            self.classification_cache[cache_key] = classification
            logger.debug(f"Cached classification for text: {text[:50]}...")

    # ========== Response Cache ==========

    def get_cached_response(self, query: str, context_hash: str) -> Optional[str]:
        """
        Get cached response for query with specific context.

        Args:
            query: User query
            context_hash: Hash of context documents (to ensure same context)

        Returns:
            Cached response or None if not found
        """
        cache_key = self._cache_key(f"{query}|{context_hash}")

        with self.response_lock:
            response = self.response_cache.get(cache_key)

            if response is not None:
                self.stats["response_hits"] += 1
                logger.debug(f"Response cache HIT for query: {query[:50]}...")
                return response
            else:
                self.stats["response_misses"] += 1
                logger.debug(f"Response cache MISS for query: {query[:50]}...")
                return None

    def set_cached_response(self, query: str, context_hash: str, response: str) -> None:
        """
        Cache response for query with specific context.

        Args:
            query: User query
            context_hash: Hash of context documents
            response: Generated response
        """
        cache_key = self._cache_key(f"{query}|{context_hash}")

        with self.response_lock:
            self.response_cache[cache_key] = response
            logger.debug(f"Cached response for query: {query[:50]}...")

    # ========== Statistics ==========

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hit/miss rates and sizes
        """
        def hit_rate(hits: int, misses: int) -> float:
            total = hits + misses
            return (hits / total * 100) if total > 0 else 0.0

        # Acquire locks in consistent order to prevent deadlocks
        # Always acquire in the same order: embedding -> search -> classification
        with self.embedding_lock:
            embedding_stats = {
                "size": len(self.embedding_cache),
                "max_size": self.embedding_cache.maxsize,
                "hits": self.stats["embedding_hits"],
                "misses": self.stats["embedding_misses"],
                "hit_rate": hit_rate(
                    self.stats["embedding_hits"],
                    self.stats["embedding_misses"]
                ),
            }

        with self.search_lock:
            search_stats = {
                "size": len(self.search_cache),
                "max_size": self.search_cache.maxsize,
                "hits": self.stats["search_hits"],
                "misses": self.stats["search_misses"],
                "hit_rate": hit_rate(
                    self.stats["search_hits"],
                    self.stats["search_misses"]
                ),
            }

        with self.classification_lock:
            classification_stats = {
                "size": len(self.classification_cache),
                "max_size": self.classification_cache.maxsize,
                "hits": self.stats["classification_hits"],
                "misses": self.stats["classification_misses"],
                "hit_rate": hit_rate(
                    self.stats["classification_hits"],
                    self.stats["classification_misses"]
                ),
            }

        with self.response_lock:
            response_stats = {
                "size": len(self.response_cache),
                "max_size": self.response_cache.maxsize,
                "hits": self.stats["response_hits"],
                "misses": self.stats["response_misses"],
                "hit_rate": hit_rate(
                    self.stats["response_hits"],
                    self.stats["response_misses"]
                ),
            }

        return {
            "embedding": embedding_stats,
            "search": search_stats,
            "classification": classification_stats,
            "response": response_stats,
        }

    def clear_all(self) -> None:
        """Clear all caches. Acquires locks in consistent order to prevent deadlocks."""
        # Acquire locks in consistent order: embedding -> search -> classification -> response
        with self.embedding_lock:
            self.embedding_cache.clear()
        with self.search_lock:
            self.search_cache.clear()
        with self.classification_lock:
            self.classification_cache.clear()
        with self.response_lock:
            self.response_cache.clear()
        logger.info("All caches cleared")

    def clear_embedding_cache(self) -> None:
        """Clear only embedding cache."""
        with self.embedding_lock:
            self.embedding_cache.clear()
            logger.info("Embedding cache cleared")

    def clear_search_cache(self) -> None:
        """Clear only search cache."""
        with self.search_lock:
            self.search_cache.clear()
            logger.info("Search cache cleared")

    def clear_classification_cache(self) -> None:
        """Clear only classification cache."""
        with self.classification_lock:
            self.classification_cache.clear()
            logger.info("Classification cache cleared")

    def clear_response_cache(self) -> None:
        """Clear only response cache."""
        with self.response_lock:
            self.response_cache.clear()
            logger.info("Response cache cleared")


# Global cache instance (singleton pattern)
_global_cache = None
_global_cache_lock = Lock()


def get_cache() -> RAGCache:
    """
    Get the global cache instance (singleton).
    Thread-safe implementation using double-checked locking.

    Returns:
        RAGCache instance
    """
    global _global_cache

    # Double-checked locking pattern for thread-safe singleton
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = RAGCache()
                logger.info("Global cache instance created")

    return _global_cache


def reset_cache() -> None:
    """Reset the global cache (useful for testing)."""
    global _global_cache
    if _global_cache:
        _global_cache.clear_all()
    _global_cache = None
