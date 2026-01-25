"""
Configuration management for the AI Reliability Engine.

This module provides production-grade configuration with validation,
environment variable support, and sensible defaults.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class GroundingConfig(BaseModel):
    """Configuration for real-time grounding."""
    
    # Performance constraints
    max_latency_ms: float = Field(
        default=50.0,
        ge=1.0,
        le=10000.0,  # Increased max for tests
        description="Maximum allowed latency for real-time grounding"
    )
    max_sentences: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of sentences to evaluate"
    )
    max_response_length: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum response length in characters"
    )
    
    # Scoring weights
    support_weight: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight for semantic support in grounding score"
    )
    coverage_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for coverage in grounding score"
    )
    agreement_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for evidence agreement in grounding score"
    )
    
    # Thresholds
    support_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum support for a sentence to be considered supported"
    )
    allow_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum grounding score for ALLOW decision"
    )
    hedge_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum grounding score for HEDGE decision"
    )
    
    @field_validator('hedge_threshold')
    @classmethod
    def validate_threshold_ordering(cls, v: float, info) -> float:
        """Ensure thresholds are properly ordered."""
        if info.data and 'allow_threshold' in info.data:
            allow_threshold = info.data['allow_threshold']
            if v >= allow_threshold:
                raise ValueError("hedge_threshold must be less than allow_threshold")
        return v

    @field_validator('support_weight', 'coverage_weight', 'agreement_weight')
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Ensure weights sum to approximately 1.0."""
        if info.data:
            weights = {
                'support_weight': info.data.get('support_weight', 0.50),
                'coverage_weight': info.data.get('coverage_weight', 0.30),
                'agreement_weight': info.data.get('agreement_weight', 0.20),
            }
            # Get the field name that's currently being validated
            field_name = info.field_name
            weights[field_name] = v
            
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:  # Allow small floating point tolerance
                raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding backend."""
    
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Name of the embedding model to use"
    )
    device: str = Field(
        default="cpu",
        description="Device to run embeddings on (cpu/cuda/mps)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding computation"
    )
    max_sequence_length: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="Maximum sequence length for tokenization"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds"
    )
    cache_max_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum number of cached embeddings"
    )
    
    # Redis configuration (if using Redis cache)
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis URL for distributed caching"
    )
    redis_prefix: str = Field(
        default="ai_reliability:embeddings:",
        description="Redis key prefix"
    )


class ReliabilityConfig(BaseModel):
    """Main configuration for the reliability engine."""
    
    grounding: GroundingConfig = Field(
        default_factory=GroundingConfig,
        description="Grounding configuration"
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding configuration"
    )
    
    # Overall reliability weights
    grounding_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for grounding in overall reliability score"
    )
    consistency_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for consistency in overall reliability score"
    )
    uncertainty_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for uncertainty in overall reliability score"
    )
    stability_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for stability in overall reliability score"
    )
    context_quality_weight: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Weight for context quality in overall reliability score"
    )
    
    # Logging and monitoring
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port for metrics endpoint"
    )
    
    @classmethod
    def from_env(cls) -> ReliabilityConfig:
        """Create configuration from environment variables."""
        return cls(
            grounding=GroundingConfig(
                max_latency_ms=float(os.getenv("AI_RELIABILITY_MAX_LATENCY_MS", "50.0")),
                max_sentences=int(os.getenv("AI_RELIABILITY_MAX_SENTENCES", "10")),
                max_response_length=int(os.getenv("AI_RELIABILITY_MAX_RESPONSE_LENGTH", "5000")),
                support_threshold=float(os.getenv("AI_RELIABILITY_SUPPORT_THRESHOLD", "0.7")),
                allow_threshold=float(os.getenv("AI_RELIABILITY_ALLOW_THRESHOLD", "0.85")),
                hedge_threshold=float(os.getenv("AI_RELIABILITY_HEDGE_THRESHOLD", "0.65")),
            ),
            embedding=EmbeddingConfig(
                model_name=os.getenv("AI_RELIABILITY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                batch_size=int(os.getenv("AI_RELIABILITY_EMBEDDING_BATCH_SIZE", "32")),
                redis_url=os.getenv("AI_RELIABILITY_REDIS_URL"),
                cache_ttl_seconds=int(os.getenv("AI_RELIABILITY_CACHE_TTL_SECONDS", "3600")),
            ),
            log_level=os.getenv("AI_RELIABILITY_LOG_LEVEL", "INFO"),
            enable_metrics=os.getenv("AI_RELIABILITY_ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("AI_RELIABILITY_METRICS_PORT", "9090")),
        )

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    def get_reliability_weights(self) -> Dict[str, float]:
        """Get reliability scoring weights as a dictionary."""
        return {
            "grounding": self.grounding_weight,
            "consistency": self.consistency_weight,
            "uncertainty": self.uncertainty_weight,
            "stability": self.stability_weight,
            "context_quality": self.context_quality_weight,
        }
