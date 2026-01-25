"""
Real-time grounding evaluation for the AI Reliability Engine.

This module provides fast, memory-efficient grounding analysis with
strict latency requirements suitable for production use.
"""

from __future__ import annotations

import numpy as np
import structlog

# Import using relative imports (correct approach for library code)
from ..core.config import GroundingConfig
from ..core.result import ReliabilityDecision, ReliabilityExplanation
from ..embeddings.encoder import EmbeddingEncoder
from ..utils.text import normalize_response
from ..utils.timing import Timer, latency_budget, performance_tracker

logger = structlog.get_logger(__name__)


class RealTimeGrounding:
    """
    Real-time grounding engine with 50ms latency budget.
    
    Implements the 6-step grounding pipeline:
    1. Response normalization
    2. Sentence segmentation  
    3. Lightweight claim proxy
    4. Semantic support check
    5. Coverage proxy
    6. Evidence agreement
    """
    
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        config: GroundingConfig,
    ):
        """
        Initialize real-time grounding engine.
        
        Args:
            encoder: Embedding encoder for semantic similarity
            config: Grounding configuration
        """
        self.encoder = encoder
        self.config = config
        
        logger.info(
            "realtime_grounding_initialized",
            max_latency_ms=config.max_latency_ms,
            max_sentences=config.max_sentences,
            support_threshold=config.support_threshold,
            allow_threshold=config.allow_threshold,
            hedge_threshold=config.hedge_threshold
        )
    
    def evaluate(
        self,
        response: str,
        context: Dict[str, Any],
        budget_ms: Optional[float] = None
    ) -> Tuple[float, ReliabilityExplanation, ReliabilityDecision]:
        """
        Evaluate response grounding in real-time.
        
        Args:
            response: Response text to evaluate
            context: Context containing evidence sources
            budget_ms: Latency budget (uses config default if None)
            
        Returns:
            Tuple of (grounding_score, explanation, decision)
        """
        if budget_ms is None:
            budget_ms = self.config.max_latency_ms
        
        with latency_budget(budget_ms, "realtime_grounding"):
            return self._evaluate_with_timing(response, context)
    
    def _evaluate_with_timing(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Tuple[float, ReliabilityExplanation, ReliabilityDecision]:
        """
        Internal evaluation method with detailed timing.
        
        Args:
            response: Response text to evaluate
            context: Context containing evidence sources
            
        Returns:
            Tuple of (grounding_score, explanation, decision)
        """
        total_timer = Timer("grounding_total")
        total_timer.start()
        
        try:
            # Step 1: Response normalization
            normalized_response, norm_time = self._normalize_response(response)
            
            # Step 2: Extract evidence from context
            evidence_sources, evidence_time = self._extract_evidence(context)
            
            # Step 3: Sentence segmentation
            sentences, sentence_metadata, seg_time = self._segment_sentences(normalized_response)
            
            # Step 4: Semantic support check
            support_scores, support_time = self._compute_support_scores(
                sentences, evidence_sources
            )
            
            # Step 5: Coverage calculation
            coverage_score, coverage_time = self._compute_coverage(support_scores)
            
            # Step 6: Evidence agreement
            agreement_score, agreement_time = self._compute_evidence_agreement(evidence_sources)
            
            # Step 7: Final grounding score
            grounding_score = self._compute_grounding_score(
                support_scores, coverage_score, agreement_score
            )
            
            # Step 8: Decision making
            decision = self._make_decision(grounding_score)
            
            # Step 9: Build explanation
            explanation = self._build_explanation(
                sentences,
                sentence_metadata,
                support_scores,
                coverage_score,
                agreement_score,
                {
                    "normalization_ms": norm_time,
                    "evidence_extraction_ms": evidence_time,
                    "segmentation_ms": seg_time,
                    "support_ms": support_time,
                    "coverage_ms": coverage_time,
                    "agreement_ms": agreement_time,
                }
            )
            
            total_timer.stop()
            
            # Record performance metrics
            performance_tracker.record_measurement("grounding_total", total_timer.duration_ms)
            
            logger.info(
                "grounding_evaluation_completed",
                grounding_score=grounding_score,
                decision=decision.value,
                processing_time_ms=total_timer.duration_ms,
                sentence_count=len(sentences),
                evidence_count=len(evidence_sources)
            )
            
            return grounding_score, explanation, decision
            
        except Exception as e:
            total_timer.stop()
            logger.error(
                "grounding_evaluation_failed",
                error=str(e),
                processing_time_ms=total_timer.duration_ms
            )
            # Return safe defaults on error
            return self._safe_fallback(response, context)
    
    def _normalize_response(self, response: str) -> Tuple[str, float]:
        """
        Normalize response text.
        
        Args:
            response: Raw response text
            
        Returns:
            Tuple of (normalized_response, processing_time_ms)
        """
        timer = Timer("normalization")
        with timer:
            normalized = normalize_response(response, self.config.max_response_length)
        return normalized, timer.duration_ms
    
    def _extract_evidence(self, context: Dict[str, Any]) -> List[str]:
        """
        Extract evidence sources from context.
        
        Args:
            context: Context dictionary
            
        Returns:
            List of evidence texts
        """
        timer = Timer("evidence_extraction")
        timer.start()
        
        evidence_sources = []
        
        # Extract from prompt
        if "prompt" in context and context["prompt"]:
            evidence_sources.append(str(context["prompt"]))
        
        # Extract from tool outputs
        if "tool_outputs" in context and context["tool_outputs"]:
            for output in context["tool_outputs"]:
                if output:
                    evidence_sources.append(str(output))
        
        # Extract from memory
        if "memory" in context and context["memory"]:
            for memory_item in context["memory"]:
                if memory_item:
                    evidence_sources.append(str(memory_item))
        
        # Extract from constraints
        if "constraints" in context and context["constraints"]:
            constraints_text = str(context["constraints"])
            if constraints_text and constraints_text != "{}":
                evidence_sources.append(constraints_text)
        
        timer.stop()
        return evidence_sources, timer.duration_ms
    
    def _segment_sentences(self, response: str) -> Tuple[List[str], List[dict], float]:
        """
        Segment response into sentences with metadata.
        
        Args:
            response: Normalized response text
            
        Returns:
            Tuple of (sentences, sentence_metadata, processing_time_ms)
        """
        from .decomposition import SentenceDecomposer
        
        timer = Timer("segmentation")
        timer.start()
        
        decomposer = SentenceDecomposer(
            max_sentences=self.config.max_sentences,
            max_response_length=self.config.max_response_length
        )
        
        sentences, sentence_metadata = decomposer.decompose_response(
            response, budget_ms=10.0
        )
        
        timer.stop()
        return sentences, sentence_metadata, timer.duration_ms
    
    def _compute_support_scores(
        self,
        sentences: List[str],
        evidence_sources: List[str]
    ) -> Tuple[List[float], float]:
        """
        Compute semantic support scores for each sentence.
        
        Args:
            sentences: List of sentences to evaluate
            evidence_sources: List of evidence texts
            
        Returns:
            Tuple of (support_scores, processing_time_ms)
        """
        timer = Timer("support_computation")
        timer.start()
        
        if not sentences or not evidence_sources:
            timer.stop()
            return [], timer.duration_ms
        
        # Compute similarities in batch for efficiency
        all_support_scores = []
        
        for sentence in sentences:
            # Compute similarity against all evidence sources
            similarities = self.encoder.compute_batch_similarities(
                sentence, evidence_sources, normalize=True
            )
            
            # Use maximum similarity as support score
            max_support = max(similarities) if similarities else 0.0
            all_support_scores.append(max_support)
        
        timer.stop()
        return all_support_scores, timer.duration_ms
    
    def _compute_coverage(self, support_scores: List[float]) -> Tuple[float, float]:
        """
        Compute coverage proxy from support scores.
        
        Args:
            support_scores: List of support scores for sentences
            
        Returns:
            Tuple of (coverage_score, processing_time_ms)
        """
        timer = Timer("coverage_computation")
        with timer:
            if not support_scores:
                coverage = 0.0
            else:
                supported_count = sum(
                    1 for score in support_scores
                    if score >= self.config.support_threshold
                )
                coverage = supported_count / len(support_scores)
        
        return coverage, timer.duration_ms
    
    def _compute_evidence_agreement(self, evidence_sources: List[str]) -> Tuple[float, float]:
        """
        Compute agreement score between evidence sources.
        
        Args:
            evidence_sources: List of evidence texts
            
        Returns:
            Tuple of (agreement_score, processing_time_ms)
        """
        timer = Timer("agreement_computation")
        timer.start()
        
        if len(evidence_sources) < 2:
            timer.stop()
            return 1.0, timer.duration_ms  # Perfect agreement with single source
        
        # Compute pairwise similarities between evidence sources
        similarities = []
        
        for i, evidence1 in enumerate(evidence_sources):
            for j, evidence2 in enumerate(evidence_sources[i+1:], i+1):
                similarity = self.encoder.compute_similarity(evidence1, evidence2)
                similarities.append(similarity)
        
        # Agreement is the average pairwise similarity
        agreement = np.mean(similarities) if similarities else 1.0
        
        timer.stop()
        return float(agreement), timer.duration_ms
    
    def _compute_grounding_score(
        self,
        support_scores: List[float],
        coverage: float,
        agreement: float
    ) -> float:
        """
        Compute final grounding score.
        
        Args:
            support_scores: List of support scores
            coverage: Coverage proxy score
            agreement: Evidence agreement score
            
        Returns:
            Final grounding score [0,1]
        """
        mean_support = np.mean(support_scores) if support_scores else 0.0
        
        grounding_score = (
            self.config.support_weight * mean_support +
            self.config.coverage_weight * coverage +
            self.config.agreement_weight * agreement
        )
        
        # Ensure score is in valid range
        grounding_score = max(0.0, min(1.0, grounding_score))
        
        return grounding_score
    
    def _make_decision(self, grounding_score: float) -> ReliabilityDecision:
        """
        Make reliability decision based on grounding score.
        
        Args:
            grounding_score: Computed grounding score
            
        Returns:
            Reliability decision
        """
        if grounding_score >= self.config.allow_threshold:
            return ReliabilityDecision.ALLOW
        elif grounding_score >= self.config.hedge_threshold:
            return ReliabilityDecision.HEDGE
        else:
            return ReliabilityDecision.BLOCK
    
    def _build_explanation(
        self,
        sentences: List[str],
        sentence_metadata: List[dict],
        support_scores: List[float],
        coverage: float,
        agreement: float,
        timing: Dict[str, float]
    ) -> ReliabilityExplanation:
        """
        Build detailed explanation for grounding evaluation.
        
        Args:
            sentences: List of evaluated sentences
            sentence_metadata: Sentence metadata
            support_scores: Support scores for each sentence
            coverage: Coverage score
            agreement: Agreement score
            timing: Processing timing information
            
        Returns:
            Detailed explanation
        """
        # Identify unsupported sentences
        unsupported_sentences = []
        sentence_scores = []
        
        for sentence, support_score, metadata in zip(sentences, support_scores, sentence_metadata):
            sentence_info = {
                "sentence": sentence,
                "support": support_score,
                "is_claim": metadata.get("is_claim", False),
                "complexity": metadata.get("complexity", 0.0),
                "is_supported": support_score >= self.config.support_threshold,
            }
            sentence_scores.append(sentence_info)
            
            if support_score < self.config.support_threshold:
                unsupported_sentences.append(sentence)
        
        # Check for low agreement
        low_agreement = agreement < 0.5
        
        # Calculate mean support
        mean_support = np.mean(support_scores) if support_scores else 0.0
        
        # Total processing time
        total_time = sum(timing.values())
        
        return ReliabilityExplanation(
            unsupported_sentences=unsupported_sentences,
            low_agreement=low_agreement,
            coverage=coverage,
            mean_support=mean_support,
            agreement_score=agreement,
            processing_time_ms=total_time,
            sentence_scores=sentence_scores,
            evidence_sources=[],  # Will be filled by caller
            warnings=[],  # Will be filled if needed
        )
    
    def _safe_fallback(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Tuple[float, ReliabilityExplanation, ReliabilityDecision]:
        """
        Safe fallback when evaluation fails.
        
        Args:
            response: Original response
            context: Original context
            
        Returns:
            Conservative fallback results
        """
        logger.warning("grounding_fallback_activated", response_length=len(response))
        
        explanation = ReliabilityExplanation(
            unsupported_sentences=[],
            low_agreement=False,
            coverage=0.0,
            mean_support=0.0,
            agreement_score=1.0,
            processing_time_ms=0.0,
            sentence_scores=[],
            evidence_sources=[],
            warnings=["Evaluation failed, using conservative fallback"]
        )
        
        return 0.0, explanation, ReliabilityDecision.BLOCK
