"""
Core data structures for reliability results and decisions.

This module defines the fundamental data types used throughout the reliability engine,
with production-grade typing, validation, and serialization support.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


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
    mean_support: float = Field(
        ge=0.0,
        le=1.0,
        description="Average semantic support across all sentences"
    )
    agreement_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Agreement score between evidence sources"
    )
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

    # Single pre-validation clamp for all unit floats — must run before ge/le constraints.
    @field_validator('coverage', 'mean_support', 'agreement_score', mode='before')
    @classmethod
    def clamp_unit_float(cls, v: float) -> float:
        """Clamp to [0,1] before constraint checking to absorb floating-point drift."""
        return max(0.0, min(1.0, float(v)))

    @model_validator(mode='after')
    def validate_coverage_consistency(self) -> 'ReliabilityExplanation':
        """Cross-check coverage against sentence_scores after all fields are set."""
        if self.sentence_scores:
            supported_count = sum(
                1 for s in self.sentence_scores
                if s.get('support', 0.0) > 0.7
            )
            expected = supported_count / len(self.sentence_scores)
            if abs(self.coverage - expected) > 0.05:
                raise ValueError(
                    f"Coverage {self.coverage:.3f} doesn't match sentence_scores "
                    f"(expected {expected:.3f})"
                )
        return self


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

    # No custom decision validator needed: Pydantic coerces strings to ReliabilityDecision
    # and rejects unknown values before any field_validator runs.

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
