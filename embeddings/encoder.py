"""
Embedding encoder for the AI Reliability Engine.

This module provides a production-ready embedding backend with model loading,
batching, and performance optimization for real-time requirements.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np
import structlog
import torch
from sentence_transformers import SentenceTransformer

from .cache import EmbeddingCache

logger = structlog.get_logger(__name__)


class EmbeddingEncoder:
    """
    Production-ready embedding encoder with caching and performance optimization.
    
    Designed for real-time reliability evaluation with strict latency requirements.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_sequence_length: int = 512,
        cache_ttl_seconds: int = 3600,
        cache_max_size: int = 10000,
        redis_url: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize embedding encoder.
        
        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for embedding computation
            max_sequence_length: Maximum sequence length for tokenization
            cache_ttl_seconds: Cache TTL in seconds
            cache_max_size: Maximum cache size
            redis_url: Redis URL for distributed caching
            device: PyTorch device (auto-detected if None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize cache
        self.cache = EmbeddingCache(
            max_size=cache_max_size,
            ttl_seconds=cache_ttl_seconds,
            redis_url=redis_url
        )
        
        # Model will be loaded lazily
        self._model: Optional[SentenceTransformer] = None
        self._model_loaded = False
        
        logger.info(
            "embedding_encoder_initialized",
            model_name=model_name,
            device=self.device,
            batch_size=batch_size,
            cache_enabled=redis_url is not None
        )
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model.
        
        Returns:
            Loaded model instance
        """
        if self._model is None:
            logger.info("loading_embedding_model", model_name=self.model_name)
            
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device
                )
                
                # Set max sequence length if supported
                if hasattr(self._model, 'max_seq_length'):
                    self._model.max_seq_length = self.max_sequence_length
                
                self._model_loaded = True
                
                logger.info(
                    "embedding_model_loaded",
                    model_name=self.model_name,
                    device=self.device,
                    max_seq_length=getattr(self._model, 'max_seq_length', 'unknown')
                )
                
            except Exception as e:
                logger.error("model_loading_failed", error=str(e))
                raise RuntimeError(f"Failed to load embedding model: {e}")
        
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,
        show_progress: bool = False,
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            normalize_embeddings: Whether to normalize embeddings
            show_progress: Whether to show progress bar (for debugging)
            
        Returns:
            Embedding(s) as list(s) of floats
        """
        # Normalize input to list
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Filter out cached embeddings
        uncached_texts = []
        uncached_indices = []
        embeddings = [None] * len(texts)
        
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                embeddings[i] = cached_embedding
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached_texts:
            model = self._load_model()
            
            # Batch encode
            new_embeddings = model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Cache and place new embeddings
            for text, embedding, idx in zip(uncached_texts, new_embeddings, uncached_indices):
                embedding_list = embedding.tolist()
                self.cache.set(text, self.model_name, embedding_list)
                embeddings[idx] = embedding_list
        
        # Return single embedding if input was single text
        if single_input:
            return embeddings[0] if embeddings[0] else []
        
        return embeddings
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text string.
        
        Optimized for single text encoding with caching.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding as list of floats
        """
        # Check cache first
        cached = self.cache.get(text, self.model_name)
        if cached is not None:
            return cached
        
        # Encode and cache
        embedding = self.encode(text, normalize_embeddings=True)
        return embedding  # type: ignore
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
        normalize: bool = True,
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            normalize: Whether to normalize embeddings
            
        Returns:
            Cosine similarity score [-1, 1]
        """
        # Get embeddings
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        
        if not emb1 or not emb2:
            return 0.0
        
        # Convert to numpy arrays
        arr1 = np.array(emb1)
        arr2 = np.array(emb2)
        
        # Compute cosine similarity
        if normalize:
            # Embeddings are already normalized if using sentence-transformers
            similarity = np.dot(arr1, arr2)
        else:
            # Manual normalization
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(arr1, arr2) / (norm1 * norm2)
        
        return float(similarity)
    
    def compute_batch_similarities(
        self,
        query_text: str,
        corpus_texts: List[str],
        normalize: bool = True,
    ) -> List[float]:
        """
        Compute similarities between query text and corpus texts.
        
        Optimized for batch similarity computation.
        
        Args:
            query_text: Query text
            corpus_texts: List of corpus texts
            normalize: Whether to normalize embeddings
            
        Returns:
            List of similarity scores
        """
        if not corpus_texts:
            return []
        
        # Get query embedding
        query_embedding = self.encode_single(query_text)
        if not query_embedding:
            return [0.0] * len(corpus_texts)
        
        # Get corpus embeddings
        corpus_embeddings = self.encode(corpus_texts, normalize_embeddings=normalize)
        
        # Compute similarities
        query_arr = np.array(query_embedding)
        similarities = []
        
        for corpus_emb in corpus_embeddings:
            if not corpus_emb:
                similarities.append(0.0)
                continue
            
            corpus_arr = np.array(corpus_emb)
            similarity = np.dot(query_arr, corpus_arr) if normalize else np.dot(query_arr, corpus_arr) / (
                np.linalg.norm(query_arr) * np.linalg.norm(corpus_arr)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.
        
        Returns:
            Embedding dimension
        """
        if not self._model_loaded:
            # Load model to get dimension
            self._load_model()
        
        if self._model and hasattr(self._model, 'get_sentence_embedding_dimension'):
            return self._model.get_sentence_embedding_dimension()
        
        # Default dimension for common models
        return 384  # all-MiniLM-L6-v2 dimension
    
    def warm_up_cache(self, common_texts: List[str]) -> None:
        """
        Warm up cache with commonly used texts.
        
        Args:
            common_texts: List of texts to pre-embed
        """
        logger.info("warming_up_cache", text_count=len(common_texts))
        self.cache.preload_embeddings(common_texts, self._load_model())
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
    
    def __del__(self):
        """Cleanup when encoder is destroyed."""
        if self._model is not None:
            # Clear model from GPU memory if applicable
            if hasattr(self._model, 'to'):
                try:
                    self._model.to('cpu')
                except Exception:
                    pass
