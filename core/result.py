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
    """Complete reliability evaluation result."""
    
    # Primary scores
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall reliability score [0,1]"
    )
    grounding: float = Field(
        ge=0.0,
        le=1.0,
        description="Grounding score [0,1]"
    )
    consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Consistency score [0,1]"
    )
    uncertainty: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncertainty score [0,1], higher = more uncertain"
    )
    stability: float = Field(
        ge=0.0,
        le=1.0,
        description="Stability score [0,1]"
    )
    
    # Decision and explanation
    decision: ReliabilityDecision = Field(
        description="Recommended action based on reliability"
    )
    explanation: ReliabilityExplanation = Field(
        description="Detailed explanation of the scoring"
    )
    
    # Metadata
    response_length: int = Field(
        ge=0,
        description="Length of the evaluated response in characters"
    )
    sentence_count: int = Field(
        ge=0,
        description="Number of sentences evaluated"
    )
    evidence_count: int = Field(
        ge=0,
        description="Number of evidence items used"
    )
    model_version: str = Field(
        default="0.1.0",
        description="Version of the reliability model used"
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO timestamp of evaluation"
    )
    
    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
        json_encoders={
            ReliabilityDecision: lambda v: v.value
        }
    )

    @field_validator('decision')
    @classmethod
    def validate_decision_consistency(cls, v: ReliabilityDecision, info) -> ReliabilityDecision:
        """Ensure decision aligns with grounding score."""
        if info.data and 'grounding' in info.data:
            grounding = info.data['grounding']
            if grounding >= 0.85 and v != ReliabilityDecision.ALLOW:
                raise ValueError(
                    f"High grounding ({grounding:.3f}) should result in ALLOW decision"
                )
            elif grounding < 0.65 and v not in (ReliabilityDecision.BLOCK, ReliabilityDecision.CLARIFY):
                raise ValueError(
                    f"Low grounding ({grounding:.3f}) should result in BLOCK or CLARIFY decision"
                )
            elif 0.65 <= grounding < 0.85 and v not in (ReliabilityDecision.HEDGE, ReliabilityDecision.ALLOW):
                raise ValueError(
                    f"Medium grounding ({grounding:.3f}) should result in HEDGE or ALLOW decision"
                )
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
