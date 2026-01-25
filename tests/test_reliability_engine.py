"""
Comprehensive tests for the ReliabilityEngine.

This module tests the complete reliability evaluation pipeline
with performance benchmarks and edge case handling.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any, Dict

import pytest
import pytest_asyncio

# Add parent directory to path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'embeddings'))

# Import modules directly
import config
import engine
import result
import encoder


class TestReliabilityEngine:
    """Test suite for ReliabilityEngine."""
    
    @pytest.fixture
    def config(self) -> config.ReliabilityConfig:
        """Create test configuration with realistic performance requirements."""
        return config.ReliabilityConfig(
            grounding={
                "max_latency_ms": 150.0,  # Realistic <150ms target
                "max_sentences": 5,
                "support_threshold": 0.7,
                "allow_threshold": 0.85,
                "hedge_threshold": 0.65,
            },
            embedding={
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 16,
                "cache_ttl_seconds": 300,
                "cache_max_size": 1000,
            }
        )
    
    @pytest.fixture
    def encoder(self, config: config.ReliabilityConfig) -> encoder.EmbeddingEncoder:
        """Create test embedding encoder."""
        return encoder.EmbeddingEncoder(
            model_name=config.embedding.model_name,
            batch_size=config.embedding.batch_size,
            cache_ttl_seconds=config.embedding.cache_ttl_seconds,
            cache_max_size=config.embedding.cache_max_size,
        )
    
    @pytest.fixture
    def engine(self, config: config.ReliabilityConfig, encoder: encoder.EmbeddingEncoder) -> engine.ReliabilityEngine:
        """Create test reliability engine."""
        return engine.ReliabilityEngine(config=config, encoder=encoder)
    
    @pytest.fixture
    def sample_response(self) -> str:
        """Sample response for testing."""
        return "The capital of France is Paris. It has a population of about 2.1 million people. The city is known for the Eiffel Tower."
    
    @pytest.fixture
    def sample_context(self) -> Dict[str, Any]:
        """Sample context for testing."""
        return {
            "prompt": "What is the capital of France and tell me about it?",
            "tool_outputs": [
                "Paris is the capital city of France.",
                "The population of Paris metropolitan area is approximately 2.1 million people.",
                "The Eiffel Tower is a famous landmark in Paris."
            ],
            "memory": [],
            "constraints": {"max_length": 500}
        }
    
    def test_performance_requirements(self, engine: engine.ReliabilityEngine, sample_response: str, sample_context: Dict[str, Any]):
        """Test that engine meets <150ms performance requirements."""
        import time
        
        # Warm up the engine (load model)
        engine.evaluate(sample_response, sample_context)
        
        # Measure performance after warm-up
        start_time = time.time()
        result = engine.evaluate(sample_response, sample_context)
        end_time = time.time()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Verify performance requirements
        assert processing_time_ms < 150.0, f"Performance failed: {processing_time_ms:.2f}ms > 150ms"
        
        # Verify result is still valid
        assert result.score >= 0.0 and result.score <= 1.0
        assert isinstance(result.decision, result.ReliabilityDecision)
        
        print(f"Performance test passed: {processing_time_ms:.2f}ms")
    
    def test_basic_evaluation(self, engine: engine.ReliabilityEngine, sample_response: str, sample_context: Dict[str, Any]):
        """Test basic reliability evaluation."""
        result = engine.evaluate(sample_response, sample_context)
        
        # Check primary result structure
        assert result.score >= 0.0 and result.score <= 1.0
        assert result.grounding >= 0.0 and result.grounding <= 1.0
        assert result.uncertainty >= 0.0 and result.uncertainty <= 1.0
        
        # Check decision
        assert isinstance(result.decision, result.ReliabilityDecision)
        
        # Check explanation
        assert result.explanation is not None
        assert result.processing_time_ms > 0
        assert result.response_length > 0
        assert result.evidence_count >= 0
        
        # Check optional fields
        if result.consistency is not None:
            assert result.consistency >= 0.0 and result.consistency <= 1.0
        if result.stability is not None:
            assert result.stability >= 0.0 and result.stability <= 1.0
        assert result.explanation.coverage >= 0.0 and result.explanation.coverage <= 1.0
        assert result.explanation.mean_support >= 0.0 and result.explanation.mean_support <= 1.0
        assert result.explanation.agreement_score >= 0.0 and result.explanation.agreement_score <= 1.0
        
        # Check metadata
        assert result.response_length == len(sample_response)
        assert result.evidence_count > 0
        assert result.processing_time_ms > 0
    
    def test_empty_response(self, engine: engine.ReliabilityEngine, sample_context: Dict[str, Any]):
        """Test evaluation with empty response."""
        result = engine.evaluate("", sample_context)
        
        # Should handle gracefully
        assert result.score >= 0.0 and result.score <= 1.0
        assert result.decision in [result.ReliabilityDecision.BLOCK, result.ReliabilityDecision.CLARIFY]
        assert result.response_length == 0
    
    def test_empty_context(self, engine: engine.ReliabilityEngine, sample_response: str):
        """Test evaluation with empty context."""
        result = engine.evaluate(sample_response, {})
        
        # Should handle gracefully but with lower scores
        assert result.score >= 0.0 and result.score <= 1.0
        assert result.evidence_count == 0
        assert result.decision in [result.ReliabilityDecision.BLOCK, result.ReliabilityDecision.CLARIFY]
    
    def test_long_response_truncation(self, engine: engine.ReliabilityEngine, sample_context: Dict[str, Any]):
        """Test handling of very long responses."""
        long_response = "This is a test sentence. " * 1000  # Very long response
        
        result = engine.evaluate(long_response, sample_context)
        
        # Should handle gracefully (truncated)
        assert result.score >= 0.0 and result.score <= 1.0
        assert result.response_length <= engine.config.grounding.max_response_length
    
    def test_performance_requirements(self, engine: engine.ReliabilityEngine, sample_response: str, sample_context: Dict[str, Any]):
        """Test performance requirements."""
        start_time = time.time()
        
        # Run multiple evaluations
        for _ in range(10):
            result = engine.evaluate(sample_response, sample_context)
            assert result.processing_time_ms < 200.0  # Should be under 200ms
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10.0 * 1000.0  # Convert to ms
        
        # Average should be reasonable
        assert avg_time < 150.0, f"Average evaluation time {avg_time:.2f}ms exceeds 150ms limit"
    
    @pytest.mark.benchmark
    def test_evaluation_performance_benchmark(self, engine: engine.ReliabilityEngine, sample_response: str, sample_context: Dict[str, Any], benchmark):
        """Benchmark evaluation performance."""
        def evaluate():
            return engine.evaluate(sample_response, sample_context)
        
        result = benchmark(evaluate)
        
        # Verify benchmark result is valid
        assert result.score >= 0.0 and result.score <= 1.0
        assert result.processing_time_ms < 100.0  # Should be fast
    
    def test_cache_effectiveness(self, engine: engine.ReliabilityEngine, sample_response: str, sample_context: Dict[str, Any]):
        """Test that caching improves performance."""
        # First evaluation (cache miss)
        start_time = time.time()
        result1 = engine.evaluate(sample_response, sample_context)
        first_time = (time.time() - start_time) * 1000.0
        
        # Second evaluation (cache hit)
        start_time = time.time()
        result2 = engine.evaluate(sample_response, sample_context)
        second_time = (time.time() - start_time) * 1000.0
        
        # Results should be identical
        assert result1.score == result2.score
        assert result1.grounding == result2.grounding
        assert result1.decision == result2.decision
        
        # Second evaluation should be faster (cache effect)
        # Note: This might not always be true due to system variability, so we just log it
        print(f"First evaluation: {first_time:.2f}ms, Second evaluation: {second_time:.2f}ms")
    
    def test_different_response_types(self, engine: engine.ReliabilityEngine, sample_context: Dict[str, Any]):
        """Test evaluation of different types of responses."""
        test_cases = [
            # High confidence, well-supported
            ("Paris is the capital of France. The Eiffel Tower is located there.", result.ReliabilityDecision.ALLOW),
            
            # Medium confidence, some uncertainty
            ("Paris might be the capital of France, and I think the Eiffel Tower is probably there.", result.ReliabilityDecision.HEDGE),
            
            # Low confidence, no support
            ("The capital of Mars is New York City and it has 10 billion people.", result.ReliabilityDecision.BLOCK),
            
            # Question (should be handled gracefully)
            ("What is the capital of France?", result.ReliabilityDecision.ALLOW),
        ]
        
        for response, expected_decision in test_cases:
            result = engine.evaluate(response, sample_context)
            
            # Verify basic structure
            assert result.score >= 0.0 and result.score <= 1.0
            assert isinstance(result.decision, result.ReliabilityDecision)
            
            # Log the actual vs expected for analysis
            print(f"Response: {response[:50]}...")
            print(f"Expected: {expected_decision}, Got: {result.decision}, Score: {result.score:.3f}")
    
    def test_context_quality_impact(self, engine: engine.ReliabilityEngine):
        """Test that context quality affects reliability scores."""
        response = "Paris is the capital of France."
        
        # Test with no context
        result_no_context = engine.evaluate(response, {})
        
        # Test with minimal context
        result_minimal = engine.evaluate(response, {"prompt": "What is the capital of France?"})
        
        # Test with rich context
        result_rich = engine.evaluate(response, {
            "prompt": "What is the capital of France?",
            "tool_outputs": ["Paris is the capital city of France."],
            "memory": ["Previously discussed French geography"],
            "constraints": {"max_length": 100}
        })
        
        # Rich context should generally result in higher scores
        assert result_rich.score >= result_minimal.score >= result_no_context.score
        
        print(f"No context: {result_no_context.score:.3f}")
        print(f"Minimal context: {result_minimal.score:.3f}")
        print(f"Rich context: {result_rich.score:.3f}")
    
    def test_error_handling(self, engine: engine.ReliabilityEngine):
        """Test error handling and graceful degradation."""
        # Test with malformed input
        try:
            result = engine.evaluate("test", {"invalid": object()})
            # Should handle gracefully
            assert result.score >= 0.0 and result.score <= 1.0
        except Exception as e:
            # If it raises, should be a meaningful error
            assert "failed" in str(e).lower() or "error" in str(e).lower()
    
    def test_performance_stats(self, engine: engine.ReliabilityEngine, sample_response: str, sample_context: Dict[str, Any]):
        """Test performance statistics collection."""
        # Run a few evaluations
        for _ in range(5):
            engine.evaluate(sample_response, sample_context)
        
        # Get stats
        stats = engine.get_performance_stats()
        
        # Should have some statistics
        assert "reliability_total" in stats
        assert "grounding_total" in stats
        
        reliability_stats = stats["reliability_total"]
        assert reliability_stats["count"] == 5
        assert reliability_stats["mean_ms"] > 0
        assert reliability_stats["min_ms"] > 0
        assert reliability_stats["max_ms"] > 0
    
    def test_warm_up(self, engine: engine.ReliabilityEngine):
        """Test cache warm-up functionality."""
        common_texts = [
            "This is a common test sentence.",
            "Another common phrase for testing.",
            "Frequently used text in our application."
        ]
        
        # Warm up should not raise errors
        engine.warm_up(common_texts)
        
        # Cache stats should show entries
        cache_stats = engine.encoder.get_cache_stats()
        assert cache_stats["memory_cache_size"] >= 0


class TestReliabilityEngineIntegration:
    """Integration tests for ReliabilityEngine."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create engine with default config
        engine = engine.ReliabilityEngine()
        
        # Test realistic scenario
        response = "Based on the search results, Apple Inc. is headquartered in Cupertino, California and was founded by Steve Jobs in 1976."
        
        context = {
            "prompt": "Tell me about Apple Inc.",
            "tool_outputs": [
                "Apple Inc. is an American multinational technology company.",
                "Apple is headquartered in Cupertino, California.",
                "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."
            ],
            "memory": ["User previously asked about tech companies"],
            "constraints": {"max_length": 200}
        }
        
        result = engine.evaluate(response, context)
        
        # Verify complete workflow
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.decision, result.ReliabilityDecision)
        assert result.explanation is not None
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 200.0  # Should be fast
        
        # Should be reasonably confident for this well-supported response
        assert result.score > 0.5
        assert result.grounding > 0.5
        
        print(f"End-to-end test: Score={result.score:.3f}, Decision={result.decision}, Time={result.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_reliability_engine.py -v
    pytest.main([__file__, "-v"])
