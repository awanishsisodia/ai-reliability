"""
Response decomposition for the AI Reliability Engine.

This module provides sentence segmentation and analysis for grounding evaluation.
"""

from __future__ import annotations

from typing import List, Tuple

import structlog

from ..utils.text import (
    extract_claim_like_sentences,
    get_sentence_complexity,
    is_question,
    split_into_sentences,
)
from ..utils.timing import Timer, latency_budget

logger = structlog.get_logger(__name__)


class SentenceDecomposer:
    """
    Optimized sentence decomposition for real-time grounding.
    
    Breaks down responses into evaluable sentences with claim detection
    and complexity analysis, all within strict latency budgets.
    """
    
    def __init__(self, max_sentences: int = 10, max_response_length: int = 5000):
        """
        Initialize sentence decomposer.
        
        Args:
            max_sentences: Maximum number of sentences to process
            max_response_length: Maximum response length to process
        """
        self.max_sentences = max_sentences
        self.max_response_length = max_response_length
        
        logger.info(
            "sentence_decomposer_initialized",
            max_sentences=max_sentences,
            max_response_length=max_response_length
        )
    
    def decompose_response(
        self,
        response: str,
        budget_ms: float = 10.0
    ) -> Tuple[List[str], List[dict]]:
        """
        Decompose response into sentences with metadata.
        
        Args:
            response: Response text to decompose
            budget_ms: Latency budget in milliseconds
            
        Returns:
            Tuple of (sentences, sentence_metadata_list)
        """
        with latency_budget(budget_ms, "sentence_decomposition"):
            return self._decompose_with_timing(response)
    
    def _decompose_with_timing(self, response: str) -> Tuple[List[str], List[dict]]:
        """
        Internal decomposition method with timing.
        
        Args:
            response: Response text to decompose
            
        Returns:
            Tuple of (sentences, sentence_metadata_list)
        """
        timer = Timer("decomposition")
        timer.start()
        
        # Step 1: Basic sentence splitting
        sentences = split_into_sentences(response, self.max_sentences)
        
        # Step 2: Extract claim-like sentences
        claim_sentences = extract_claim_like_sentences(sentences)
        
        # Step 3: Generate metadata for each sentence
        sentence_metadata = []
        for i, sentence in enumerate(sentences):
            metadata = {
                "index": i,
                "text": sentence,
                "is_claim": sentence in claim_sentences,
                "is_question": is_question(sentence),
                "complexity": get_sentence_complexity(sentence),
                "length": len(sentence),
                "word_count": len(sentence.split()),
            }
            sentence_metadata.append(metadata)
        
        timer.stop()
        
        logger.debug(
            "response_decomposed",
            sentence_count=len(sentences),
            claim_count=len(claim_sentences),
            processing_time_ms=timer.duration_ms,
            response_length=len(response)
        )
        
        return sentences, sentence_metadata
    
    def get_high_risk_sentences(
        self,
        sentence_metadata: List[dict],
        risk_threshold: float = 0.7
    ) -> List[dict]:
        """
        Identify high-risk sentences based on complexity and claim indicators.
        
        Args:
            sentence_metadata: List of sentence metadata
            risk_threshold: Minimum complexity score for high-risk classification
            
        Returns:
            List of high-risk sentence metadata
        """
        high_risk = []
        
        for metadata in sentence_metadata:
            # Skip questions (lower risk)
            if metadata["is_question"]:
                continue
            
            # High complexity or claim-like indicates higher risk
            if (metadata["complexity"] >= risk_threshold or 
                metadata["is_claim"]):
                high_risk.append(metadata)
        
        return high_risk
    
    def prioritize_sentences(
        self,
        sentence_metadata: List[dict]
    ) -> List[dict]:
        """
        Prioritize sentences for grounding evaluation.
        
        Higher priority sentences should be evaluated first for early
        detection of grounding issues.
        
        Args:
            sentence_metadata: List of sentence metadata
            
        Returns:
            Prioritized list of sentence metadata
        """
        # Calculate priority score for each sentence
        for metadata in sentence_metadata:
            priority = 0.0
            
            # Claims get higher priority
            if metadata["is_claim"]:
                priority += 0.4
            
            # Higher complexity gets higher priority
            priority += metadata["complexity"] * 0.3
            
            # Longer sentences get slightly higher priority
            length_factor = min(metadata["length"] / 200.0, 1.0)
            priority += length_factor * 0.2
            
            # Non-questions get higher priority
            if not metadata["is_question"]:
                priority += 0.1
            
            metadata["priority"] = priority
        
        # Sort by priority (descending)
        prioritized = sorted(sentence_metadata, key=lambda x: x["priority"], reverse=True)
        
        return prioritized
    
    def get_decomposition_stats(self, sentence_metadata: List[dict]) -> dict:
        """
        Get statistics about the decomposition.
        
        Args:
            sentence_metadata: List of sentence metadata
            
        Returns:
            Decomposition statistics
        """
        if not sentence_metadata:
            return {
                "total_sentences": 0,
                "claim_sentences": 0,
                "question_sentences": 0,
                "avg_complexity": 0.0,
                "avg_length": 0.0,
            }
        
        total_sentences = len(sentence_metadata)
        claim_sentences = sum(1 for m in sentence_metadata if m["is_claim"])
        question_sentences = sum(1 for m in sentence_metadata if m["is_question"])
        avg_complexity = sum(m["complexity"] for m in sentence_metadata) / total_sentences
        avg_length = sum(m["length"] for m in sentence_metadata) / total_sentences
        
        return {
            "total_sentences": total_sentences,
            "claim_sentences": claim_sentences,
            "question_sentences": question_sentences,
            "avg_complexity": avg_complexity,
            "avg_length": avg_length,
            "claim_ratio": claim_sentences / total_sentences,
            "question_ratio": question_sentences / total_sentences,
        }
