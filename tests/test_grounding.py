"""
Tests for real-time grounding components.

This module tests the grounding pipeline with focus on
performance requirements and accuracy.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from ai_reliability.core.config import GroundingConfig
from ai_reliability.core.result import ReliabilityDecision
from ai_reliability.embeddings.encoder import EmbeddingEncoder
from ai_reliability.grounding.decomposition import SentenceDecomposer
from ai_reliability.grounding.realtime import RealTimeGrounding


class TestSentenceDecomposer:
    """Test suite for SentenceDecomposer."""
    
    @pytest.fixture
    def decomposer(self) -> SentenceDecomposer:
        """Create test sentence decomposer."""
        return SentenceDecomposer(max_sentences=10, max_response_length=1000)
    
    def test_basic_decomposition(self, decomposer: SentenceDecomposer):
        """Test basic sentence decomposition."""
        response = "Paris is the capital of France. It has many landmarks. The Eiffel Tower is famous."
        
        sentences, metadata = decomposer.decompose_response(response)
        
        assert len(sentences) == 3
        assert len(metadata) == 3
        
        # Check sentence content
        assert "Paris is the capital of France" in sentences[0]
        assert "It has many landmarks" in sentences[1]
        assert "The Eiffel Tower is famous" in sentences[2]
        
        # Check metadata
        for i, meta in enumerate(metadata):
            assert meta["index"] == i
            assert "text" in meta
            assert "is_claim" in meta
            assert "is_question" in meta
            assert "complexity" in meta
            assert "length" in meta
            assert "word_count" in meta
    
    def test_long_response_truncation(self, decomposer: SentenceDecomposer):
        """Test truncation of long responses."""
        long_response = "This is sentence one. " * 100
        
        sentences, metadata = decomposer.decompose_response(long_response)
        
        # Should respect max_sentences limit
        assert len(sentences) <= decomposer.max_sentences
        assert len(metadata) <= decomposer.max_sentences
    
    def test_empty_response(self, decomposer: SentenceDecomposer):
        """Test handling of empty response."""
        sentences, metadata = decomposer.decompose_response("")
        
        assert len(sentences) == 0
        assert len(metadata) == 0
    
    def test_question_detection(self, decomposer: SentenceDecomposer):
        """Test question detection."""
        response = "What is the capital of France? Paris is the capital. How many people live there?"
        
        sentences, metadata = decomposer.decompose_response(response)
        
        # Should detect questions
        questions = [meta for meta in metadata if meta["is_question"]]
        assert len(questions) == 2
        
        # Check specific questions
        question_texts = [meta["text"] for meta in questions]
        assert any("What is the capital" in text for text in question_texts)
        assert any("How many people" in text for text in question_texts)
    
    def test_claim_detection(self, decomposer: SentenceDecomposer):
        """Test claim detection."""
        response = "Paris is the capital of France. The population is 2.1 million. What do you think?"
        
        sentences, metadata = decomposer.decompose_response(response)
        
        # Should detect claims
        claims = [meta for meta in metadata if meta["is_claim"]]
        assert len(claims) >= 1  # At least one claim
        
        # Check specific claims (sentences with entities/numbers)
        claim_texts = [meta["text"] for meta in claims]
        assert any("capital of France" in text or "2.1 million" in text for text in claim_texts)
    
    def test_complexity_scoring(self, decomposer: SentenceDecomposer):
        """Test complexity scoring."""
        # Simple sentence
        simple = "Paris is a city."
        # Complex sentence with entities and numbers
        complex_sent = "Apple Inc., founded in 1976 by Steve Jobs, achieved a market capitalization of $2.5 trillion in 2023."
        
        sentences, metadata = decomposer.decompose_response(f"{simple} {complex_sent}")
        
        # Find complexity scores
        simple_meta = next(meta for meta in metadata if simple in meta["text"])
        complex_meta = next(meta for meta in metadata if complex_sent in meta["text"])
        
        # Complex sentence should have higher complexity
        assert complex_meta["complexity"] > simple_meta["complexity"]
    
    def test_prioritization(self, decomposer: SentenceDecomposer):
        """Test sentence prioritization."""
        response = "Paris is the capital. What is the weather? Apple Inc. is worth $2 trillion."
        
        sentences, metadata = decomposer.decompose_response(response)
        
        # Prioritize sentences
        prioritized = decomposer.prioritize_sentences(metadata)
        
        # Should maintain all sentences
        assert len(prioritized) == len(metadata)
        
        # Should have priority scores
        for meta in prioritized:
            assert "priority" in meta
            assert 0.0 <= meta["priority"] <= 1.0
        
        # Claims should generally have higher priority
        claims = [meta for meta in prioritized if meta["is_claim"]]
        non_claims = [meta for meta in prioritized if not meta["is_claim"]]
        
        if claims and non_claims:
            avg_claim_priority = sum(meta["priority"] for meta in claims) / len(claims)
            avg_non_claim_priority = sum(meta["priority"] for meta in non_claims) / len(non_claims)
            assert avg_claim_priority >= avg_non_claim_priority
    
    def test_performance_requirements(self, decomposer: SentenceDecomposer):
        """Test performance requirements."""
        response = "This is sentence one. This is sentence two. This is sentence three."
        
        start_time = time.time()
        
        # Run multiple decompositions
        for _ in range(100):
            sentences, metadata = decomposer.decompose_response(response)
            assert len(sentences) == 3
            assert len(metadata) == 3
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100.0 * 1000.0  # Convert to ms
        
        # Should be very fast (text processing only)
        assert avg_time < 5.0, f"Average decomposition time {avg_time:.2f}ms exceeds 5ms limit"


class TestRealTimeGrounding:
    """Test suite for RealTimeGrounding."""
    
    @pytest.fixture
    def config(self) -> GroundingConfig:
        """Create test grounding configuration."""
        return GroundingConfig(
            max_latency_ms=5000.0,  # Increased for tests (model loading takes time)
            max_sentences=5,
            support_threshold=0.7,
            allow_threshold=0.85,
            hedge_threshold=0.65,
        )
    
    @pytest.fixture
    def encoder(self) -> EmbeddingEncoder:
        """Create test embedding encoder."""
        return EmbeddingEncoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,
            cache_ttl_seconds=300,
            cache_max_size=1000,
        )
    
    @pytest.fixture
    def grounding(self, config: GroundingConfig, encoder: EmbeddingEncoder) -> RealTimeGrounding:
        """Create test real-time grounding."""
        return RealTimeGrounding(encoder=encoder, config=config)
    
    @pytest.fixture
    def well_supported_case(self) -> tuple[str, Dict[str, Any]]:
        """Well-supported response and context."""
        response = "Paris is the capital of France. The Eiffel Tower is located there."
        context = {
            "prompt": "Tell me about Paris.",
            "tool_outputs": [
                "Paris is the capital city of France.",
                "The Eiffel Tower is a famous landmark located in Paris."
            ],
            "memory": [],
            "constraints": {}
        }
        return response, context
    
    @pytest.fixture
    def unsupported_case(self) -> tuple[str, Dict[str, Any]]:
        """Unsupported response and context."""
        response = "The capital of Mars is New York City with 50 billion people."
        context = {
            "prompt": "Tell me about Mars.",
            "tool_outputs": ["Mars is the fourth planet from the Sun."],
            "memory": [],
            "constraints": {}
        }
        return response, context
    
    def test_well_supported_evaluation(self, grounding: RealTimeGrounding, well_supported_case: tuple[str, Dict[str, Any]]):
        """Test evaluation of well-supported response."""
        response, context = well_supported_case
        
        grounding_score, explanation, decision = grounding.evaluate(response, context)
        
        # Should have high grounding score
        assert grounding_score >= 0.7
        assert decision in [ReliabilityDecision.ALLOW, ReliabilityDecision.HEDGE]
        
        # Should have good coverage
        assert explanation.coverage >= 0.5
        assert explanation.mean_support >= 0.7
        
        # Should have few unsupported sentences
        assert len(explanation.unsupported_sentences) <= 1
    
    def test_unsupported_evaluation(self, grounding: RealTimeGrounding, unsupported_case: tuple[str, Dict[str, Any]]):
        """Test evaluation of unsupported response."""
        response, context = unsupported_case
        
        grounding_score, explanation, decision = grounding.evaluate(response, context)
        
        # Should have low grounding score
        assert grounding_score < 0.65
        assert decision == ReliabilityDecision.BLOCK
        
        # Should have poor coverage
        assert explanation.coverage < 0.5
        assert explanation.mean_support < 0.7
        
        # Should have unsupported sentences
        assert len(explanation.unsupported_sentences) >= 1
    
    def test_empty_evidence(self, grounding: RealTimeGrounding):
        """Test evaluation with no evidence."""
        response = "Paris is the capital of France."
        context = {}  # No evidence
        
        grounding_score, explanation, decision = grounding.evaluate(response, context)
        
        # Should handle gracefully but with low scores
        assert grounding_score < 0.5
        assert decision == ReliabilityDecision.BLOCK
        assert explanation.coverage == 0.0
        assert explanation.mean_support == 0.0
    
    def test_evidence_agreement(self, grounding: RealTimeGrounding):
        """Test evidence agreement calculation."""
        # Conflicting evidence
        response = "The population is 1 million."
        context = {
            "prompt": "What is the population?",
            "tool_outputs": [
                "The population is 1 million people.",
                "The population is 5 million people."  # Conflict
            ],
            "memory": [],
            "constraints": {}
        }
        
        grounding_score, explanation, decision = grounding.evaluate(response, context)
        
        # Should detect disagreement
        assert explanation.agreement_score < 0.8  # Some disagreement detected
        assert explanation.low_agreement == explanation.agreement_score < 0.5
    
    def test_performance_requirements(self, grounding: RealTimeGrounding, well_supported_case: tuple[str, Dict[str, Any]]):
        """Test performance requirements."""
        response, context = well_supported_case
        
        start_time = time.time()
        
        # Run multiple evaluations
        for _ in range(10):
            grounding_score, explanation, decision = grounding.evaluate(response, context)
            assert 0.0 <= grounding_score <= 1.0
            assert explanation.processing_time_ms < grounding.config.max_latency_ms
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10.0 * 1000.0  # Convert to ms
        
        # Should be under latency budget
        assert avg_time < grounding.config.max_latency_ms, f"Average time {avg_time:.2f}ms exceeds budget {grounding.config.max_latency_ms}ms"
    
    def test_latency_budget_enforcement(self, grounding: RealTimeGrounding, well_supported_case: tuple[str, Dict[str, Any]]):
        """Test that latency budget is enforced."""
        response, context = well_supported_case
        
        # Set very tight budget
        tight_budget = 10.0  # 10ms
        
        try:
            grounding_score, explanation, decision = grounding.evaluate(response, context, budget_ms=tight_budget)
            # If it succeeds, should be very fast
            assert explanation.processing_time_ms < tight_budget
        except Exception as e:
            # If it fails, should be due to latency budget
            assert "latency" in str(e).lower() or "budget" in str(e).lower() or "time" in str(e).lower()
    
    def test_explanation_completeness(self, grounding: RealTimeGrounding, well_supported_case: tuple[str, Dict[str, Any]]):
        """Test that explanations are complete and useful."""
        response, context = well_supported_case
        
        grounding_score, explanation, decision = grounding.evaluate(response, context)
        
        # Check explanation structure
        assert explanation.coverage >= 0.0 and explanation.coverage <= 1.0
        assert explanation.mean_support >= 0.0 and explanation.mean_support <= 1.0
        assert explanation.agreement_score >= 0.0 and explanation.agreement_score <= 1.0
        assert explanation.processing_time_ms > 0
        
        # Check sentence scores
        assert len(explanation.sentence_scores) > 0
        for sentence_score in explanation.sentence_scores:
            assert "sentence" in sentence_score
            assert "support" in sentence_score
            assert "is_claim" in sentence_score
            assert "complexity" in sentence_score
            assert "is_supported" in sentence_score
            assert 0.0 <= sentence_score["support"] <= 1.0
    
    def test_different_response_lengths(self, grounding: RealTimeGrounding):
        """Test handling of different response lengths."""
        context = {
            "prompt": "Test prompt",
            "tool_outputs": ["Test evidence"],
            "memory": [],
            "constraints": {}
        }
        
        test_cases = [
            "Short.",  # Very short
            "This is a medium length response with multiple sentences. It should be handled normally.",
            "This is a very long response. " * 20,  # Very long (will be truncated)
        ]
        
        for response in test_cases:
            grounding_score, explanation, decision = grounding.evaluate(response, context)
            
            # Should handle all cases gracefully
            assert 0.0 <= grounding_score <= 1.0
            assert isinstance(decision, ReliabilityDecision)
            assert explanation.processing_time_ms < grounding.config.max_latency_ms
            
            print(f"Response length: {len(response)}, Score: {grounding_score:.3f}, Time: {explanation.processing_time_ms:.2f}ms")


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_grounding.py -v
    pytest.main([__file__, "-v"])
