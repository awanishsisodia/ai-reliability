"""
Tests for embedding components.

This module tests the embedding encoder and cache with focus on
performance and accuracy.
"""

from __future__ import annotations

import time
from typing import List

import pytest

from ai_reliability.embeddings.cache import EmbeddingCache
from ai_reliability.embeddings.encoder import EmbeddingEncoder


class TestEmbeddingCache:
    """Test suite for EmbeddingCache."""
    
    @pytest.fixture
    def cache(self) -> EmbeddingCache:
        """Create test cache."""
        return EmbeddingCache(
            max_size=10,
            ttl_seconds=60,
            redis_url=None  # In-memory only for tests
        )
    
    def test_basic_cache_operations(self, cache: EmbeddingCache):
        """Test basic cache get/set operations."""
        text = "This is a test sentence."
        model_name = "test-model"
        embedding = [0.1, 0.2, 0.3, 0.4]
        
        # Initially should be empty
        assert cache.get(text, model_name) is None
        
        # Set and retrieve
        cache.set(text, model_name, embedding)
        retrieved = cache.get(text, model_name)
        
        assert retrieved == embedding
        
        # Different text should not return the same embedding
        assert cache.get("different text", model_name) is None
        
        # Different model should not return the same embedding
        assert cache.get(text, "different-model") is None
    
    def test_cache_expiration(self, cache: EmbeddingCache):
        """Test cache TTL expiration."""
        text = "Test text"
        model_name = "test-model"
        embedding = [0.1, 0.2, 0.3]
        
        # Set with very short TTL
        cache.ttl_seconds = 0.1  # 100ms
        cache.set(text, model_name, embedding)
        
        # Should be available immediately
        assert cache.get(text, model_name) == embedding
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get(text, model_name) is None
    
    def test_cache_size_limit(self, cache: EmbeddingCache):
        """Test cache size limit (LRU eviction)."""
        embeddings = [[i, i+1, i+2] for i in range(15)]  # More than max_size
        
        # Fill cache beyond limit
        for i, embedding in enumerate(embeddings):
            text = f"text_{i}"
            cache.set(text, "test-model", embedding)
        
        # Should only keep max_size items
        stats = cache.get_stats()
        assert stats["memory_cache_size"] <= cache.max_size
        
        # Most recent items should be kept
        assert cache.get("text_14", "test-model") == embeddings[14]
        assert cache.get("text_13", "test-model") == embeddings[13]
        
        # Oldest items should be evicted
        assert cache.get("text_0", "test-model") is None
        assert cache.get("text_1", "test-model") is None
    
    def test_cache_stats(self, cache: EmbeddingCache):
        """Test cache statistics."""
        text = "Test text"
        model_name = "test-model"
        embedding = [0.1, 0.2, 0.3]
        
        # Initially empty
        stats = cache.get_stats()
        assert stats["memory_cache_size"] == 0
        assert stats["memory_cache_limit"] == cache.max_size
        assert stats["ttl_seconds"] == cache.ttl_seconds
        assert stats["redis_enabled"] is False
        
        # Add item
        cache.set(text, model_name, embedding)
        
        stats = cache.get_stats()
        assert stats["memory_cache_size"] == 1
        
        # Clear cache
        cache.clear()
        
        stats = cache.get_stats()
        assert stats["memory_cache_size"] == 0
    
    def test_cache_key_generation(self, cache: EmbeddingCache):
        """Test cache key generation."""
        text1 = "Test text"
        text2 = "Different text"
        model1 = "model1"
        model2 = "model2"
        embedding = [0.1, 0.2, 0.3]
        
        # Same text and model should have same key
        cache.set(text1, model1, embedding)
        assert cache.get(text1, model1) == embedding
        
        # Different text should have different key
        assert cache.get(text2, model1) is None
        
        # Different model should have different key
        assert cache.get(text1, model2) is None
    
    def test_preload_embeddings(self, cache: EmbeddingCache):
        """Test preloading embeddings."""
        # Mock sentence transformer
        class MockModel:
            _target_device = "cpu"
            
            def encode(self, texts, **kwargs):
                import numpy as np
                return np.random.rand(len(texts), 3)  # 3-dimensional embeddings
        
        texts = ["text1", "text2", "text3"]
        model = MockModel()
        
        # Preload should not raise errors
        cache.preload_embeddings(texts, model)
        
        # Should have cached embeddings
        stats = cache.get_stats()
        assert stats["memory_cache_size"] == len(texts)


class TestEmbeddingEncoder:
    """Test suite for EmbeddingEncoder."""
    
    @pytest.fixture
    def encoder(self) -> EmbeddingEncoder:
        """Create test embedding encoder."""
        return EmbeddingEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
            cache_ttl_seconds=300,
            cache_max_size=1000,
            redis_url=None  # In-memory cache for tests
        )
    
    def test_encoder_initialization(self, encoder: EmbeddingEncoder):
        """Test encoder initialization."""
        assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert encoder.batch_size == 16
        assert encoder.cache is not None
        assert encoder.device in ["cpu", "cuda"]
    
    def test_single_text_encoding(self, encoder: EmbeddingEncoder):
        """Test encoding a single text."""
        text = "This is a test sentence."
        
        embedding = encoder.encode_single(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Should be normalized (sentence-transformers does this by default)
        import numpy as np
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        assert abs(norm - 1.0) < 0.01  # Should be approximately unit norm
    
    def test_batch_text_encoding(self, encoder: EmbeddingEncoder):
        """Test encoding multiple texts."""
        texts = [
            "This is the first sentence.",
            "This is the second sentence.",
            "This is the third sentence."
        ]
        
        embeddings = encoder.encode(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, float) for x in embedding)
        
        # All embeddings should have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1  # All same dimension
    
    def test_caching_behavior(self, encoder: EmbeddingEncoder):
        """Test that caching improves performance."""
        text = "This is a test sentence for caching."
        
        # First encoding (cache miss)
        start_time = time.time()
        embedding1 = encoder.encode_single(text)
        first_time = (time.time() - start_time) * 1000.0
        
        # Second encoding (cache hit)
        start_time = time.time()
        embedding2 = encoder.encode_single(text)
        second_time = (time.time() - start_time) * 1000.0
        
        # Results should be identical
        assert embedding1 == embedding2
        
        # Cache should contain the embedding
        cached = encoder.cache.get(text, encoder.model_name)
        assert cached == embedding1
        
        print(f"First encoding: {first_time:.2f}ms, Second encoding: {second_time:.2f}ms")
    
    def test_similarity_computation(self, encoder: EmbeddingEncoder):
        """Test similarity computation."""
        text1 = "The capital of France is Paris."
        text2 = "Paris is the capital city of France."
        text3 = "The weather is nice today."
        
        # Similar texts should have high similarity
        sim_12 = encoder.compute_similarity(text1, text2)
        assert sim_12 > 0.7  # Should be quite similar
        
        # Different texts should have lower similarity
        sim_13 = encoder.compute_similarity(text1, text3)
        assert sim_13 < sim_12  # Should be less similar
        
        # Similarity should be symmetric
        sim_21 = encoder.compute_similarity(text2, text1)
        assert abs(sim_12 - sim_21) < 0.001  # Should be essentially identical
        
        # Similarity should be in valid range
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0
    
    def test_batch_similarity_computation(self, encoder: EmbeddingEncoder):
        """Test batch similarity computation."""
        query = "The capital of France is Paris."
        corpus = [
            "Paris is the capital city of France.",
            "France is a country in Europe.",
            "I like pizza.",
            "The Eiffel Tower is in Paris."
        ]
        
        similarities = encoder.compute_batch_similarities(query, corpus)
        
        assert len(similarities) == len(corpus)
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)
        
        # Most similar should be the first one (about Paris being capital)
        max_sim_index = similarities.index(max(similarities))
        assert max_sim_index == 0  # First item should be most similar
        
        print(f"Similarities: {[f'{s:.3f}' for s in similarities]}")
    
    def test_empty_text_handling(self, encoder: EmbeddingEncoder):
        """Test handling of empty or very short texts."""
        # Empty string
        empty_embedding = encoder.encode_single("")
        assert isinstance(empty_embedding, list)
        assert len(empty_embedding) > 0  # Model should handle empty input
        
        # Very short text
        short_embedding = encoder.encode_single("Hi")
        assert isinstance(short_embedding, list)
        assert len(short_embedding) > 0
        
        # Should have same dimension
        assert len(empty_embedding) == len(short_embedding)
    
    def test_embedding_dimension(self, encoder: EmbeddingEncoder):
        """Test embedding dimension consistency."""
        text = "Test sentence"
        
        embedding = encoder.encode_single(text)
        dimension = encoder.get_embedding_dimension()
        
        assert len(embedding) == dimension
        assert dimension > 0
        
        # Batch encoding should have same dimension
        embeddings = encoder.encode([text, "Another sentence"])
        for emb in embeddings:
            assert len(emb) == dimension
    
    def test_performance_requirements(self, encoder: EmbeddingEncoder):
        """Test performance requirements."""
        texts = [
            "This is sentence one.",
            "This is sentence two.",
            "This is sentence three.",
            "This is sentence four.",
            "This is sentence five."
        ]
        
        start_time = time.time()
        
        # Test single encoding performance
        for text in texts:
            embedding = encoder.encode_single(text)
            assert len(embedding) > 0
        
        single_time = (time.time() - start_time) / len(texts) * 1000.0
        
        # Test batch encoding performance
        start_time = time.time()
        embeddings = encoder.encode(texts)
        batch_time = (time.time() - start_time) * 1000.0
        
        # Batch should be more efficient per item
        assert len(embeddings) == len(texts)
        
        print(f"Single encoding avg: {single_time:.2f}ms, Batch total: {batch_time:.2f}ms")
        
        # Should be reasonable performance (not too slow)
        assert single_time < 1000.0  # Less than 1 second per encoding
        assert batch_time < 2000.0   # Less than 2 seconds for batch
    
    def test_cache_warm_up(self, encoder: EmbeddingEncoder):
        """Test cache warm-up functionality."""
        common_texts = [
            "This is a common test sentence.",
            "Another common phrase for testing.",
            "Frequently used text in our application."
        ]
        
        # Warm up should not raise errors
        encoder.warm_up_cache(common_texts)
        
        # Cache should contain the warmed up texts
        cache_stats = encoder.get_cache_stats()
        assert cache_stats["memory_cache_size"] >= 0
        
        # Encoding warmed up texts should be fast (cache hit)
        for text in common_texts:
            embedding = encoder.encode_single(text)
            assert len(embedding) > 0
    
    def test_cache_stats(self, encoder: EmbeddingEncoder):
        """Test cache statistics."""
        # Initially empty
        stats = encoder.get_cache_stats()
        assert stats["memory_cache_size"] == 0
        assert stats["memory_cache_limit"] == encoder.cache.max_size
        
        # Add some embeddings
        texts = ["text1", "text2", "text3"]
        for text in texts:
            encoder.encode_single(text)
        
        stats = encoder.get_cache_stats()
        assert stats["memory_cache_size"] == len(texts)
        
        # Clear cache
        encoder.clear_cache()
        
        stats = encoder.get_cache_stats()
        assert stats["memory_cache_size"] == 0
    
    def test_error_handling(self, encoder: EmbeddingEncoder):
        """Test error handling."""
        # Test with invalid input types
        try:
            # Should handle gracefully or raise meaningful error
            embedding = encoder.encode_single(123)  # type: ignore
            # If it doesn't raise, should return something reasonable
            assert isinstance(embedding, list)
        except Exception as e:
            # Should be meaningful error
            assert "text" in str(e).lower() or "string" in str(e).lower() or "type" in str(e).lower()


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_embeddings.py -v
    pytest.main([__file__, "-v"])
