"""
Main reliability engine orchestrator.

This module implements the core ReliabilityEngine that coordinates
all reliability evaluation components and provides the main API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog

from .config import ReliabilityConfig
from .result import ReliabilityDecision, ReliabilityResult, ReliabilityExplanation
from ..embeddings.encoder import EmbeddingEncoder
from ..grounding.realtime import RealTimeGrounding
from ..utils.timing import Timer, latency_budget, performance_tracker

logger = structlog.get_logger(__name__)


class ReliabilityEngine:
    """
    Main reliability engine orchestrator.
    
    Coordinates grounding evaluation, signal computation, and decision making
    to produce comprehensive reliability assessments.
    """
    
    def __init__(
        self,
        config: Optional[ReliabilityConfig] = None,
        encoder: Optional[EmbeddingEncoder] = None,
    ):
        """
        Initialize reliability engine.
        
        Args:
            config: Reliability configuration (uses defaults if None)
            encoder: Embedding encoder (creates default if None)
        """
        self.config = config or ReliabilityConfig()
        
        # Initialize embedding encoder
        if encoder is None:
            self.encoder = EmbeddingEncoder(
                model_name=self.config.embedding.model_name,
                batch_size=self.config.embedding.batch_size,
                max_sequence_length=self.config.embedding.max_sequence_length,
                cache_ttl_seconds=self.config.embedding.cache_ttl_seconds,
                cache_max_size=self.config.embedding.cache_max_size,
                redis_url=self.config.embedding.redis_url,
            )
        else:
            self.encoder = encoder
        
        # Initialize real-time grounding
        self.grounding = RealTimeGrounding(
            encoder=self.encoder,
            config=self.config.grounding
        )
        
        logger.info(
            "reliability_engine_initialized",
            embedding_model=self.config.embedding.model_name,
            grounding_budget_ms=self.config.grounding.max_latency_ms,
            cache_enabled=self.config.embedding.redis_url is not None
        )
    
    def evaluate(
        self,
        response: str,
        context: Dict[str, Any],
        history: Optional[Dict[str, Any]] = None,
        budget_ms: Optional[float] = None
    ) -> ReliabilityResult:
        """
        Evaluate reliability of a response.
        
        Args:
            response: Response text to evaluate
            context: Context containing evidence sources
            history: Historical reliability data (optional)
            budget_ms: Total evaluation budget (uses 100ms default if None)
            
        Returns:
            Complete reliability evaluation result
        """
        if budget_ms is None:
            budget_ms = 100.0  # Default total budget
        
        with latency_budget(budget_ms, "reliability_evaluation"):
            return self._evaluate_with_timing(response, context, history)
    
    def _evaluate_with_timing(
        self,
        response: str,
        context: Dict[str, Any],
        history: Optional[Dict[str, Any]]
    ) -> ReliabilityResult:
        """
        Internal evaluation with detailed timing.
        
        Args:
            response: Response text to evaluate
            context: Context containing evidence sources
            history: Historical reliability data
            
        Returns:
            Complete reliability evaluation result
        """
        total_timer = Timer("reliability_total")
        total_timer.start()
        
        try:
            # Step 1: Real-time grounding evaluation
            grounding_score, grounding_explanation, grounding_decision = self.grounding.evaluate(
                response, context
            )
            
            # Step 2: Compute additional signals (placeholder implementations)
            consistency_score = self._compute_consistency(response, context, history)
            uncertainty_score = self._compute_uncertainty(response, context)
            stability_score = self._compute_stability(response, history)
            context_quality_score = self._compute_context_quality(context)
            
            # Step 3: Aggregate final reliability score
            reliability_score = self._aggregate_reliability_score(
                grounding_score,
                consistency_score,
                uncertainty_score,
                stability_score,
                context_quality_score
            )
            
            # Step 4: Make final decision (may override grounding decision)
            final_decision = self._make_final_decision(
                grounding_decision,
                reliability_score,
                grounding_score
            )
            
            # Step 5: Build complete result
            result = self._build_result(
                response=response,
                context=context,
                reliability_score=reliability_score,
                grounding_score=grounding_score,
                consistency_score=consistency_score,
                uncertainty_score=uncertainty_score,
                stability_score=stability_score,
                decision=final_decision,
                grounding_explanation=grounding_explanation,
                processing_time_ms=total_timer.duration_ms
            )
            
            total_timer.stop()
            
            # Record performance metrics
            performance_tracker.record_measurement("reliability_total", total_timer.duration_ms)
            
            logger.info(
                "reliability_evaluation_completed",
                score=reliability_score,
                decision=final_decision.value,
                grounding=grounding_score,
                processing_time_ms=total_timer.duration_ms
            )
            
            return result
            
        except Exception as e:
            total_timer.stop()
            logger.error(
                "reliability_evaluation_failed",
                error=str(e),
                processing_time_ms=total_timer.duration_ms
            )
            return self._safe_fallback(response, context, total_timer.duration_ms)
    
    def _compute_consistency(
        self,
        response: str,
        context: Dict[str, Any],
        history: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute consistency score.
        
        Placeholder implementation - will be enhanced in future phases.
        
        Args:
            response: Response text
            context: Context dictionary
            history: Historical data
            
        Returns:
            Consistency score [0,1]
        """
        timer = Timer("consistency_computation")
        with timer:
            # Simple heuristic: check for internal contradictions
            # This is a placeholder for more sophisticated consistency checking
            
            # For now, return a neutral score
            return 0.8  # Placeholder
    
    def _compute_uncertainty(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Compute uncertainty score.
        
        Placeholder implementation - will be enhanced in future phases.
        
        Args:
            response: Response text
            context: Context dictionary
            
        Returns:
            Uncertainty score [0,1], higher = more uncertain
        """
        timer = Timer("uncertainty_computation")
        with timer:
            # Simple heuristic: look for uncertainty indicators
            uncertainty_indicators = [
                "might", "could", "perhaps", "possibly", "likely", "probably",
                "uncertain", "unsure", "estimate", "approximate", "around"
            ]
            
            response_lower = response.lower()
            indicator_count = sum(1 for indicator in uncertainty_indicators if indicator in response_lower)
            
            # Normalize uncertainty based on response length
            uncertainty = min(indicator_count / 10.0, 1.0)
            
            return uncertainty
    
    def _compute_stability(
        self,
        response: str,
        history: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute stability score.
        
        Placeholder implementation - will be enhanced in future phases.
        
        Args:
            response: Response text
            history: Historical data
            
        Returns:
            Stability score [0,1]
        """
        timer = Timer("stability_computation")
        with timer:
            # For now, return a neutral score
            # In future phases, this will analyze response stability over time
            return 0.9  # Placeholder
    
    def _compute_context_quality(self, context: Dict[str, Any]) -> float:
        """
        Compute context quality score.
        
        Args:
            context: Context dictionary
            
        Returns:
            Context quality score [0,1]
        """
        timer = Timer("context_quality_computation")
        with timer:
            if not context:
                return 0.0
            
            quality_score = 0.0
            
            # Check for prompt
            if context.get("prompt"):
                quality_score += 0.3
            
            # Check for tool outputs
            tool_outputs = context.get("tool_outputs", [])
            if tool_outputs:
                quality_score += 0.3 * min(len(tool_outputs) / 3.0, 1.0)
            
            # Check for memory
            memory = context.get("memory", [])
            if memory:
                quality_score += 0.2 * min(len(memory) / 5.0, 1.0)
            
            # Check for constraints
            if context.get("constraints"):
                quality_score += 0.2
            
            return min(quality_score, 1.0)
    
    def _aggregate_reliability_score(
        self,
        grounding: float,
        consistency: float,
        uncertainty: float,
        stability: float,
        context_quality: float
    ) -> float:
        """
        Aggregate component scores into final reliability score.
        
        Args:
            grounding: Grounding score
            consistency: Consistency score
            uncertainty: Uncertainty score (higher = more uncertain)
            stability: Stability score
            context_quality: Context quality score
            
        Returns:
            Final reliability score [0,1]
        """
        weights = self.config.get_reliability_weights()
        
        # Note: uncertainty is inverted (1 - uncertainty) since higher uncertainty = lower reliability
        reliability_score = (
            weights["grounding"] * grounding +
            weights["consistency"] * consistency +
            weights["uncertainty"] * (1 - uncertainty) +
            weights["stability"] * stability +
            weights["context_quality"] * context_quality
        )
        
        return max(0.0, min(1.0, reliability_score))
    
    def _make_final_decision(
        self,
        grounding_decision: ReliabilityDecision,
        reliability_score: float,
        grounding_score: float
    ) -> ReliabilityDecision:
        """
        Make final reliability decision.
        
        Args:
            grounding_decision: Decision from grounding evaluation
            reliability_score: Overall reliability score
            grounding_score: Grounding component score
            
        Returns:
            Final reliability decision
        """
        # Safety first: if grounding says BLOCK, respect it
        if grounding_decision == ReliabilityDecision.BLOCK:
            return ReliabilityDecision.BLOCK
        
        # If grounding says ALLOW but overall reliability is low, hedge
        if grounding_decision == ReliabilityDecision.ALLOW and reliability_score < 0.7:
            return ReliabilityDecision.HEDGE
        
        # Otherwise, use grounding decision
        return grounding_decision
    
    def _build_result(
        self,
        response: str,
        context: Dict[str, Any],
        reliability_score: float,
        grounding_score: float,
        consistency_score: float,
        uncertainty_score: float,
        stability_score: float,
        decision: ReliabilityDecision,
        grounding_explanation: ReliabilityExplanation,
        processing_time_ms: float
    ) -> ReliabilityResult:
        """
        Build complete reliability result.
        
        Args:
            response: Original response text
            context: Original context
            reliability_score: Final reliability score
            grounding_score: Grounding score
            consistency_score: Consistency score
            uncertainty_score: Uncertainty score
            stability_score: Stability score
            decision: Final decision
            grounding_explanation: Grounding explanation
            processing_time_ms: Total processing time
            
        Returns:
            Complete reliability result
        """
        # Update explanation with evidence sources
        evidence_sources = []
        if "prompt" in context:
            evidence_sources.append("prompt")
        if "tool_outputs" in context:
            evidence_sources.extend([f"tool_output_{i}" for i in range(len(context["tool_outputs"]))])
        if "memory" in context:
            evidence_sources.extend([f"memory_{i}" for i in range(len(context["memory"]))])
        
        grounding_explanation.evidence_sources = evidence_sources
        
        return ReliabilityResult(
            score=reliability_score,
            grounding=grounding_score,
            consistency=consistency_score,
            uncertainty=uncertainty_score,
            stability=stability_score,
            decision=decision,
            explanation=grounding_explanation,
            response_length=len(response),
            sentence_count=len(grounding_explanation.sentence_scores),
            evidence_count=len(evidence_sources),
            processing_time_ms=processing_time_ms,
        )
    
    def _safe_fallback(
        self,
        response: str,
        context: Dict[str, Any],
        processing_time_ms: float
    ) -> ReliabilityResult:
        """
        Safe fallback when evaluation fails.
        
        Args:
            response: Original response
            context: Original context
            processing_time_ms: Processing time before failure
            
        Returns:
            Conservative fallback result
        """
        logger.warning("reliability_fallback_activated", response_length=len(response))
        
        explanation = ReliabilityExplanation(
            unsupported_sentences=[],
            low_agreement=False,
            coverage=0.0,
            mean_support=0.0,
            agreement_score=1.0,
            processing_time_ms=processing_time_ms,
            sentence_scores=[],
            evidence_sources=[],
            warnings=["Evaluation failed, using conservative fallback"]
        )
        
        return ReliabilityResult(
            score=0.0,
            grounding=0.0,
            consistency=0.0,
            uncertainty=1.0,
            stability=0.0,
            decision=ReliabilityDecision.BLOCK,
            explanation=explanation,
            response_length=len(response),
            sentence_count=0,
            evidence_count=0,
            processing_time_ms=processing_time_ms,
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        return performance_tracker.get_all_stats()
    
    def warm_up(self, common_texts: list[str]) -> None:
        """
        Warm up the engine with common texts.
        
        Args:
            common_texts: List of commonly used texts for cache warming
        """
        logger.info("warming_up_reliability_engine", text_count=len(common_texts))
        self.encoder.warm_up_cache(common_texts)
