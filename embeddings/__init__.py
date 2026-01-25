"""
Embedding backend for the AI Reliability Engine.
"""

# Import using absolute imports
from ai_reliability.embeddings.encoder import EmbeddingEncoder
from ai_reliability.embeddings.cache import EmbeddingCache

__all__ = [
    "EmbeddingEncoder",
    "EmbeddingCache",
]
