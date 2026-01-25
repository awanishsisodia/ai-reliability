"""
Core data structures for reliability results and decisions.

This module defines the fundamental data types used throughout the reliability engine,
with production-grade typing, validation, and serialization support.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ReliabilityDecision(str, Enum):
    """Reliability decision enumeration."""
    
    ALLOW = "allow"
    HEDGE = "hedge" 
    BLOCK = "block"
    CLARIFY = "clarify"


class ReliabilityExplanation(BaseModel):
    """Detailed explanation for reliability scoring."""
    
    unsupported_sentences: List[str] = Field(
        default_factory=list,
        description="Sentences lacking sufficient evidence support"
    )
    low_agreement: bool = Field(
        default=False,
        description="Whether evidence sources show significant disagreement"
    )
    coverage: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of sentences with adequate support"
    )
    
    @field_validator('coverage')
    @classmethod
    def clamp_coverage(cls, v: float) -> float:
        """Clamp coverage to [0,1] range to handle floating-point precision."""
        return max(0.0, min(1.0, v))
    
    mean_support: float = Field(
        ge=0.0,
        le=1.0,
        description="Average semantic support across all sentences"
    )
    
    @field_validator('mean_support')
    @classmethod
    def clamp_mean_support(cls, v: float) -> float:
        """Clamp mean_support to [0,1] range to handle floating-point precision."""
        return max(0.0, min(1.0, v))
    
    agreement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Agreement score between evidence sources"
    )
    
    @field_validator('agreement_score')
    @classmethod
    def clamp_agreement_score(cls, v: float) -> float:
        """Clamp agreement_score to [0,1] range to handle floating-point precision."""
        return max(0.0, min(1.0, v))
    processing_time_ms: float = Field(
        ge=0.0,
        description="Total processing time in milliseconds"
    )
    sentence_scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-sentence support scores and metadata"
    )
    evidence_sources: List[str] = Field(
        default_factory=list,
        description="Sources used for evidence evaluation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings or alerts generated during evaluation"
    )

    @field_validator('coverage')
    @classmethod
    def validate_coverage(cls, v: float, info) -> float:
        """Validate coverage is consistent with sentence scores."""
        if info.data and 'sentence_scores' in info.data:
            sentence_scores = info.data['sentence_scores']
            if sentence_scores:
                supported_count = sum(
                    1 for score in sentence_scores
                    if score.get('support', 0.0) > 0.7  # Threshold for "supported"
                )
                expected_coverage = supported_count / len(sentence_scores)
                # Allow small tolerance for floating point differences
                if abs(v - expected_coverage) > 0.05:
                    raise ValueError(
                        f"Coverage {v:.3f} doesn't match expected {expected_coverage:.3f}"
                    )
        return v


class ReliabilityResult(BaseModel):
    """Simplified reliability evaluation result.
    
    Focused on primary reliability score with key supporting metrics.
    """
    
    # Primary score - the main reliability indicator
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall reliability score [0,1] - primary decision metric"
    )
    
    # Key component scores
    grounding: float = Field(
        ge=0.0,
        le=1.0,
        description="Grounding score [0,1] - evidence support"
    )
    uncertainty: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncertainty score [0,1] - higher = more uncertain"
    )
    
    # Decision and explanation
    decision: ReliabilityDecision = Field(
        description="Final reliability decision"
    )
    explanation: ReliabilityExplanation = Field(
        description="Detailed explanation and evidence"
    )
    
    # Performance metrics
    processing_time_ms: float = Field(
        ge=0.0,
        description="Total processing time in milliseconds"
    )
    
    # Optional detailed scores (for advanced users)
    consistency: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Consistency score [0,1] - internal coherence"
    )
    stability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Stability score [0,1] - response consistency"
    )
    
    # Metadata
    response_length: int = Field(
        ge=0,
        description="Length of evaluated response"
    )
    evidence_count: int = Field(
        ge=0,
        description="Number of evidence sources used"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp of evaluation"
    )

    model_config = ConfigDict(
        extra="forbid"
    )

    @field_validator('decision')
    @classmethod
    def validate_decision_type(cls, v: ReliabilityDecision, info) -> ReliabilityDecision:
        """Ensure decision is a valid ReliabilityDecision enum value."""
        if not isinstance(v, ReliabilityDecision):
            raise ValueError(f"Decision must be a ReliabilityDecision enum, got {type(v)}")
        return v

    def is_safe_to_show(self) -> bool:
        """Check if response is safe to display to user."""
        return self.decision in (ReliabilityDecision.ALLOW, ReliabilityDecision.HEDGE)

    def requires_human_review(self) -> bool:
        """Check if response requires human review."""
        return self.decision in (ReliabilityDecision.BLOCK, ReliabilityDecision.CLARIFY)

    def get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        if self.score >= 0.9:
            return "very_high"
        elif self.score >= 0.75:
            return "high"
        elif self.score >= 0.6:
            return "medium"
        elif self.score >= 0.4:
            return "low"
        else:
            return "very_low"


# Type aliases for convenience
ReliabilityScore = Union[float, ReliabilityResult]
EvidenceSource = Union[str, Dict[str, Any]]
