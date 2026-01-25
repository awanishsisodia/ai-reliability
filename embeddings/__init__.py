"""
Embedding backend for the AI Reliability Engine.
"""

from .encoder import EmbeddingEncoder
from .cache import EmbeddingCache

__all__ = [
    "EmbeddingEncoder",
    "EmbeddingCache",
]
