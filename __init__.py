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

# Auto-register the package if it's not already registered
import sys
import os

# Check if the package is already registered
if 'ai_reliability' not in sys.modules:
    # Find the package path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the package path to sys.path if not already there
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Register the package
    import importlib.util
    spec = importlib.util.spec_from_file_location('ai_reliability', __file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ai_reliability'] = module

# Import from core modules using absolute imports
from ai_reliability.core.engine import ReliabilityEngine
from ai_reliability.core.result import ReliabilityResult, ReliabilityDecision

__all__ = [
    "ReliabilityEngine",
    "ReliabilityResult", 
    "ReliabilityDecision",
]
