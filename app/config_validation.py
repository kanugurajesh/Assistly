"""
Configuration validation using Pydantic for type safety and validation.
"""
import logging
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict

logger = logging.getLogger(__name__)


class RAGSettings(BaseModel):
    """
    Validated settings for RAG pipeline.
    Ensures all configuration values are within acceptable ranges.
    """

    # Search parameters
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve from vector search"
    )

    score_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for search results (0.0-1.0)"
    )

    # Generation parameters
    max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum tokens for response generation"
    )

    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (0.0=deterministic, 2.0=creative)"
    )

    classification_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Temperature for classification (lower is more deterministic)"
    )

    # Model configuration
    llm_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use for generation"
    )

    # Feature toggles
    enable_query_enhancement: bool = Field(
        default=False,
        description="Enable GPT-4o query enhancement for better search"
    )

    # Collection settings
    collection_name: str = Field(
        default="atlan_docs",
        min_length=1,
        max_length=100,
        description="Qdrant collection name"
    )

    @field_validator('llm_model')
    @classmethod
    def validate_model(cls, v):
        """Validate that the model is supported."""
        allowed_models = [
            'gpt-4o',
            'gpt-4o-mini',
            'gpt-4-turbo',
            'gpt-4',
            'gpt-3.5-turbo'
        ]
        if v not in allowed_models:
            raise ValueError(
                f"Model must be one of {allowed_models}, got '{v}'"
            )
        return v

    @field_validator('temperature', 'classification_temperature')
    @classmethod
    def validate_temperature(cls, v, info):
        """Validate temperature is reasonable."""
        if v > 1.5:
            logger.warning(
                f"{info.field_name}={v} is very high. "
                "Responses may be inconsistent."
            )
        return v

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v):
        """Validate top_k is reasonable."""
        if v > 10:
            logger.warning(
                f"top_k={v} is quite high. "
                "This may increase costs and response time."
            )
        return v

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on attribute assignment
        extra='forbid',  # Don't allow extra fields
        use_enum_values=True
    )


class CacheSettings(BaseModel):
    """Settings for caching layer."""

    embedding_cache_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum embeddings to cache"
    )

    embedding_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Embedding cache TTL in seconds (1 min - 24 hours)"
    )

    search_cache_size: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Maximum search results to cache"
    )

    search_ttl: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Search cache TTL in seconds (1 min - 2 hours)"
    )

    classification_cache_size: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Maximum classifications to cache"
    )

    classification_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Classification cache TTL in seconds (1 min - 24 hours)"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )


class RateLimitSettings(BaseModel):
    """Settings for rate limiting."""

    openai_max_calls: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum OpenAI calls per minute"
    )

    qdrant_max_calls: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum Qdrant searches per minute"
    )

    query_max_calls: int = Field(
        default=30,
        ge=1,
        le=200,
        description="Maximum user queries per minute"
    )

    classification_max_calls: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum classifications per minute"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )


class ApplicationConfig(BaseModel):
    """Complete application configuration."""

    rag: RAGSettings = Field(default_factory=RAGSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)

    model_config = ConfigDict(
        validate_assignment=True
    )


# Example usage for testing
if __name__ == "__main__":
    # Valid settings
    try:
        settings = RAGSettings(
            top_k=5,
            score_threshold=0.3,
            temperature=0.7,
            llm_model="gpt-4o"
        )
        print("✅ Settings validated successfully")
        print(f"Settings: {settings.model_dump()}")

    except ValidationError as e:
        print(f"❌ Validation failed: {e}")

    # Invalid settings
    try:
        invalid_settings = RAGSettings(
            top_k=0,  # Too low
            score_threshold=2.0,  # Too high
            llm_model="invalid-model"
        )
    except ValidationError as e:
        print(f"\n❌ Expected validation error: {e}")
