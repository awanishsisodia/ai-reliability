"""
Embedding cache for the AI Reliability Engine.

This module provides high-performance caching for embeddings to minimize
redundant computations and meet strict latency requirements.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """
    High-performance cache for text embeddings.
    
    Supports both in-memory and Redis-based caching with TTL and size limits.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: int = 3600,
        redis_url: Optional[str] = None,
        redis_prefix: str = "ai_reliability:embeddings:"
    ):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings (in-memory only)
            ttl_seconds: Time-to-live for cached embeddings
            redis_url: Redis URL for distributed caching (None = in-memory only)
            redis_prefix: Redis key prefix
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.redis_url = redis_url
        self.redis_prefix = redis_prefix
        
        # In-memory cache
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        
        # Redis client (if configured)
        self._redis_client = None
        if redis_url:
            try:
                import redis
                self._redis_client = redis.from_url(redis_url, decode_responses=False)
                logger.info("redis_cache_connected", url=redis_url)
            except ImportError:
                logger.warning("redis_not_available", message="Redis not installed, using in-memory cache only")
            except Exception as e:
                logger.error("redis_connection_failed", error=str(e))
    
    def _generate_key(self, text: str, model_name: str) -> str:
        """
        Generate cache key for text and model combination.
        
        Args:
            text: Input text
            model_name: Name of the embedding model
            
        Returns:
            Cache key
        """
        # Create deterministic key from text and model
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """
        Serialize embedding for storage.
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Serialized bytes
        """
        return json.dumps(embedding).encode()
    
    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """
        Deserialize embedding from storage.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Embedding vector
        """
        return json.loads(data.decode())
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: Input text
            model_name: Name of the embedding model
            
        Returns:
            Cached embedding or None if not found/expired
        """
        key = self._generate_key(text, model_name)
        
        # Try Redis first (if configured)
        if self._redis_client:
            try:
                cached_data = self._redis_client.get(f"{self.redis_prefix}{key}")
                if cached_data:
                    embedding = self._deserialize_embedding(cached_data)
                    logger.debug("cache_hit_redis", key=key[:8])
                    return embedding
            except Exception as e:
                logger.warning("redis_get_failed", error=str(e))
        
        # Try in-memory cache
        if key in self._cache:
            cache_entry = self._cache[key]
            current_time = time.time()
            
            # Check TTL
            if current_time - cache_entry["timestamp"] < self.ttl_seconds:
                self._access_times[key] = current_time
                logger.debug("cache_hit_memory", key=key[:8])
                return cache_entry["embedding"]
            else:
                # Expired, remove from cache
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
        
        logger.debug("cache_miss", key=key[:8])
        return None
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Cache embedding for text.
        
        Args:
            text: Input text
            model_name: Name of the embedding model
            embedding: Embedding vector to cache
        """
        key = self._generate_key(text, model_name)
        current_time = time.time()
        
        # Store in Redis (if configured)
        if self._redis_client:
            try:
                serialized = self._serialize_embedding(embedding)
                self._redis_client.setex(
                    f"{self.redis_prefix}{key}",
                    self.ttl_seconds,
                    serialized
                )
            except Exception as e:
                logger.warning("redis_set_failed", error=str(e))
        
        # Store in memory cache
        self._cache[key] = {
            "embedding": embedding,
            "timestamp": current_time
        }
        self._access_times[key] = current_time
        
        # Enforce size limit (LRU eviction)
        if len(self._cache) > self.max_size:
            self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries from in-memory cache."""
        if not self._access_times:
            return
        
        # Find least recently used key
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from cache
        del self._cache[lru_key]
        del self._access_times[lru_key]
        
        logger.debug("cache_evicted_lru", key=lru_key[:8])
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_times.clear()
        
        if self._redis_client:
            try:
                # Clear Redis cache entries with our prefix
                pattern = f"{self.redis_prefix}*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
                    logger.info("redis_cache_cleared", keys_deleted=len(keys))
            except Exception as e:
                logger.warning("redis_clear_failed", error=str(e))
        
        logger.info("cache_cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "memory_cache_size": len(self._cache),
            "memory_cache_limit": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "redis_enabled": self._redis_client is not None,
        }
        
        if self._redis_client:
            try:
                info = self._redis_client.info()
                stats["redis_memory_usage"] = info.get("used_memory_human", "unknown")
                stats["redis_connected"] = True
            except Exception as e:
                stats["redis_connected"] = False
                stats["redis_error"] = str(e)
        
        return stats
    
    def preload_embeddings(self, texts: List[str], model: SentenceTransformer) -> None:
        """
        Preload embeddings for a list of texts.
        
        Useful for warming up cache with frequently used texts.
        
        Args:
            texts: List of texts to embed and cache
            model: Sentence transformer model
        """
        logger.info("preloading_embeddings", text_count=len(texts))
        
        # Filter out texts that are already cached
        uncached_texts = []
        for text in texts:
            if not self.get(text, model._target_device):
                uncached_texts.append(text)
        
        if uncached_texts:
            # Batch embed uncached texts
            embeddings = model.encode(
                uncached_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Cache the new embeddings
            for text, embedding in zip(uncached_texts, embeddings):
                self.set(text, model._target_device, embedding.tolist())
            
            logger.info("preloaded_embeddings", count=len(uncached_texts))
        else:
            logger.info("all_texts_already_cached")
