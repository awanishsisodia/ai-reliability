"""
AI Reliability Engine - Production-grade framework for computing per-response reliability
for modern AI systems using real-time grounding + async refinement.

This package provides:
- Model-agnostic reliability scoring
- Real-time grounding with ≤50ms latency
- Explainable reliability decisions
- Production-ready API and monitoring
"""

__version__ = "0.2.0"
__author__ = "AI Reliability Team"

# Import using absolute imports for the main package (to avoid circular dependency)
from ai_reliability.core.engine import ReliabilityEngine
from ai_reliability.core.result import ReliabilityResult, ReliabilityDecision

__all__ = [
    "ReliabilityEngine",
    "ReliabilityResult", 
    "ReliabilityDecision",
]
