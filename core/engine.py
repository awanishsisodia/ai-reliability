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
        Initialize the reliability engine.
        
        Args:
            config: Configuration for reliability evaluation
        """
        self.config = config
        
        # Initialize components
        self.encoder = EmbeddingEncoder(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size,
            cache_max_size=config.embedding.cache_max_size,
            cache_ttl_seconds=config.embedding.cache_ttl_seconds,
            redis_url=config.embedding.redis_url
        )
        
        self.grounding = RealTimeGrounding(
            encoder=self.encoder,
            config=config.grounding
        )
        
        # Pre-load the model to exclude it from evaluation timing
        logger.info("preloading_embedding_model")
        self.encoder._load_model()
        
        logger.info(
            "reliability_engine_initialized",
            embedding_model=config.embedding.model_name,
            grounding_budget_ms=config.grounding.max_latency_ms,
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
            budget_ms: Total evaluation budget (uses grounding.max_latency_ms from config)
            
        Returns:
            Complete reliability evaluation result
        """
        budget_ms = self.config.grounding.max_latency_ms
        
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
            
            # Step 5: Stop timer and build complete result
            total_timer.stop()
            
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
            # Ensure timer is stopped to get duration
            if total_timer.start_time is not None and total_timer.end_time is None:
                total_timer.stop()
            
            # Handle case where duration_ms might still be None
            duration_ms = total_timer.duration_ms if total_timer.duration_ms is not None else 0.0
            
            logger.error(
                "reliability_evaluation_failed",
                error=str(e),
                processing_time_ms=duration_ms
            )
            return self._safe_fallback(response, context, duration_ms)
    
    def _compute_consistency(
        self,
        response: str,
        context: Dict[str, Any],
        history: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute consistency score using lightweight analysis.
        
        Fast implementation optimized for <150ms total latency.
        
        Args:
            response: Response text
            context: Context dictionary
            history: Historical data
            
        Returns:
            Consistency score [0,1]
        """
        # Skip timer for performance - this is already fast enough
        # Need to implement more sophisticated consistency checking in future version
        
        # Simple contradiction patterns (fast regex would be better)
        contradiction_pairs = [
            ("always", "never"),
            ("all", "none"),
            ("every", "no"),
            ("definitely", "definitely not"),
        ]
        
        response_lower = response.lower()
        contradiction_count = 0
        
        for word1, word2 in contradiction_pairs:
            if word1 in response_lower and word2 in response_lower:
                contradiction_count += 1
        
        # Base score with penalty for contradictions
        base_score = 0.9
        consistency_penalty = min(contradiction_count * 0.2, 0.4)
        
        return max(0.5, base_score - consistency_penalty)
    
    def _compute_uncertainty(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Compute uncertainty score using optimized keyword detection.
        
        Fast implementation optimized for <150ms total latency. 
        will be enhanced in future phases.
        
        Args:
            response: Response text
            context: Context dictionary
            
        Returns:
            Uncertainty score [0,1], higher = more uncertain
        """
        # Optimized uncertainty detection - pre-compiled patterns for speed
        uncertainty_indicators = {
            "might", "could", "perhaps", "possibly", "likely", "probably",
            "uncertain", "unsure", "estimate", "approximate", "around",
            "maybe", "sometimes", "often", "usually", "generally"
        }
        
        # Fast word tokenization (split is faster than regex for simple cases)
        words = response.lower().split()
        
        # Count uncertainty indicators
        indicator_count = sum(1 for word in words if word in uncertainty_indicators)
        
        # Normalize based on response length (avoid division by zero)
        word_count = max(len(words), 1)
        uncertainty = min(indicator_count / max(word_count * 0.1, 1.0), 1.0)
        
        return uncertainty
    
    def _compute_stability(
        self,
        response: str,
        history: Optional[Dict[str, Any]]
    ) -> float:
        """
        Compute stability score using fast heuristics.
        
        Optimized for <150ms total latency.
        
        Args:
            response: Response text
            history: Historical data
            
        Returns:
            Stability score [0,1]
        """
        # Fast stability check based on response characteristics
        
        # Length-based stability (very short responses are less stable)
        length_score = min(len(response) / 100.0, 1.0)
        
        # If we have history, check for consistency (simplified)
        if history and "previous_responses" in history:
            prev_responses = history["previous_responses"]
            if prev_responses:
                # Simple similarity check (would use embeddings in production)
                # For now, check if length is similar
                avg_prev_length = sum(len(r) for r in prev_responses) / len(prev_responses)
                length_similarity = 1.0 - abs(len(response) - avg_prev_length) / max(avg_prev_length, 1.0)
                stability_score = (length_score + length_similarity) / 2.0
            else:
                stability_score = length_score
        else:
            stability_score = length_score
        
        return max(0.3, min(stability_score, 1.0))
    
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
        Make final reliability decision with enhanced safety checks.
        
        Implements guardrails-style safety measures:
        - Conservative blocking for uncertain content
        - Hedge for borderline cases
        - Allow only for high-confidence responses
        
        Args:
            grounding_decision: Decision from grounding evaluation
            reliability_score: Overall reliability score
            grounding_score: Grounding component score
            
        Returns:
            Final reliability decision
        """
        # SAFETY FIRST: Multiple layers of protection
        
        # Layer 1: If grounding says BLOCK, respect it immediately
        if grounding_decision == ReliabilityDecision.BLOCK:
            return ReliabilityDecision.BLOCK
        
        # Layer 2: Hard safety thresholds (never allow below these)
        if reliability_score < 0.3:
            logger.warning(
                "safety_block_low_reliability",
                score=reliability_score,
                grounding=grounding_score
            )
            return ReliabilityDecision.BLOCK
        
        # Layer 3: Conservative hedge for uncertain content
        if reliability_score < 0.6:
            logger.info(
                "safety_hedge_borderline",
                score=reliability_score,
                grounding=grounding_score
            )
            return ReliabilityDecision.HEDGE
        
        # Layer 4: Require high grounding for allow decisions
        if grounding_decision == ReliabilityDecision.ALLOW:
            if grounding_score < 0.8:
                # Even if grounding says allow, require high grounding score
                logger.info(
                    "safety_hedge_insufficient_grounding",
                    grounding_score=grounding_score,
                    overall_score=reliability_score
                )
                return ReliabilityDecision.HEDGE
            
            # Final check: overall reliability must be high
            if reliability_score < 0.75:
                logger.info(
                    "safety_hedge_low_overall",
                    reliability_score=reliability_score
                )
                return ReliabilityDecision.HEDGE
        
        # Layer 5: Default to cautious approach
        # If we get here, allow but with logging
        logger.info(
            "safety_allow_high_confidence",
            score=reliability_score,
            grounding=grounding_score
        )
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
        Build simplified reliability result.
        
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
            Simplified reliability result
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
            uncertainty=uncertainty_score,
            decision=decision,
            explanation=grounding_explanation,
            processing_time_ms=processing_time_ms,
            consistency=consistency_score,
            stability=stability_score,
            response_length=len(response),
            evidence_count=len(evidence_sources)
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
            uncertainty=1.0,
            decision=ReliabilityDecision.BLOCK,
            explanation=explanation,
            processing_time_ms=processing_time_ms,
            response_length=len(response),
            evidence_count=0
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
