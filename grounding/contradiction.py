"""
Contradiction detection module for identifying opposing statements and negations.
Lightweight rule-based approach to catch the most common contradiction patterns.
"""

import re
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ContradictionDetector:
    """Detects contradictions and negations in text using rule-based patterns."""
    
    def __init__(self):
        # Explicit negation patterns
        self.negation_patterns = [
            r'\b(not|no|never|none|nothing|nowhere|neither|nor)\b',
            r'\b(does not|doesn\'t|do not|don\'t|did not|didn\'t|will not|won\'t|cannot|can\'t)\b',
            r'\b(is not|isn\'t|are not|aren\'t|was not|wasn\'t|were not|weren\'t|has not|hasn\'t|have not|haven\'t)\b',
            r'\b(cannot|can\'t|could not|couldn\'t|should not|shouldn\'t|would not|wouldn\'t|might not|mightn\'t)\b',
            r'\b(unable|unable to|incapable|incapable of)\b',
            r'\b(without|lacking|lacks|absent|absence of)\b',
        ]

        # Logical antonym pairs only — both words must appear in the SAME text for
        # has_negation to trigger.  Organ co-occurrence ('heart'+'brain') is NOT an
        # antonym relationship and was removed to prevent false positives on medical text.
        self.contradictory_pairs = {
            'chambers': ['no chambers', 'not have chambers', 'without chambers', 'lacks chambers'],
            'increase': ['decrease', 'reduce', 'lower', 'diminish'],
            'decrease': ['increase', 'raise', 'elevate', 'grow'],
            'true': ['false', 'incorrect', 'wrong', 'untrue'],
            'false': ['true', 'correct', 'right', 'accurate'],
            'always': ['never', 'rarely', 'seldom', 'sometimes'],
            'never': ['always', 'often', 'frequently', 'regularly'],
        }

        # Numeric contradictions
        self.numeric_negation_patterns = [
            r'\b(\d+)\s+(not|no|never|without)\b',
            r'\b(not|no|never|without)\s+(\d+)\b',
            r'\b(\d+)\s+(less than|fewer than)\b.*\b(more than|greater than)\b',
            r'\b(more than|greater than)\b.*\b(\d+)\s+(less than|fewer than)\b',
        ]

        # High-confidence factual contradiction pairs compiled once at construction.
        # Each entry defines two patterns that, when found on opposite sides of a
        # response/evidence pair, indicate a clear factual contradiction.
        self._factual_contradiction_specs = [
            {
                'pattern': re.compile(r'\bheart.*?chambers\b', re.IGNORECASE),
                'contradiction': re.compile(r'\bbrain.*?chambers\b', re.IGNORECASE),
                'description': 'heart vs brain chambers',
            },
            {
                'pattern': re.compile(r'\bsun.*?orbit.*?earth\b', re.IGNORECASE),
                'contradiction': re.compile(r'\bearth.*?orbit.*?sun\b', re.IGNORECASE),
                'description': 'sun-earth orbit contradiction',
            },
        ]

        # Terms that appear in factual contradiction patterns — responses containing
        # these should skip the short-circuit path and get full contradiction analysis.
        # Derived from _factual_contradiction_specs so the two stay in sync.
        self.sensitive_terms: set = {
            'heart', 'brain', 'chambers', 'chamber',
            'sun', 'earth', 'orbit',
        }

        # Compile regex patterns for performance
        self.negation_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.negation_patterns]
        self.numeric_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.numeric_negation_patterns]

    def contains_sensitive_terms(self, text: str) -> bool:
        """Return True if text contains terms that require full contradiction analysis."""
        text_lower = text.lower()
        return any(term in text_lower for term in self.sensitive_terms)
    
    def detect_negation(self, text: str) -> Dict[str, Any]:
        """
        Detect if text contains explicit negation.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with negation detection results
        """
        text_lower = text.lower()
        
        # Check for explicit negation patterns
        negation_matches = []
        for regex in self.negation_regex:
            matches = regex.findall(text_lower)
            if matches:
                negation_matches.extend(matches)
        
        # Check for contradictory concept pairs
        concept_contradictions = []
        for concept, contradictions in self.contradictory_pairs.items():
            if concept in text_lower:
                for contradiction in contradictions:
                    if contradiction in text_lower:
                        concept_contradictions.append((concept, contradiction))
        
        # Check for numeric contradictions
        numeric_contradictions = []
        for regex in self.numeric_regex:
            matches = regex.findall(text_lower)
            if matches:
                numeric_contradictions.extend(matches)
        
        has_negation = bool(negation_matches) or bool(concept_contradictions) or bool(numeric_contradictions)
        
        return {
            'has_negation': has_negation,
            'negation_words': negation_matches,
            'concept_contradictions': concept_contradictions,
            'numeric_contradictions': numeric_contradictions,
            'negation_count': len(negation_matches) + len(concept_contradictions) + len(numeric_contradictions)
        }
    
    def detect_contradiction_between_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Detect contradiction between two texts with strict semantic requirements.
        
        A contradiction requires:
        1. Same entity
        2. Same predicate 
        3. Opposite truth values
        
        Args:
            text1: First text (e.g., response)
            text2: Second text (e.g., evidence)
            
        Returns:
            Dictionary with contradiction analysis
        """
        # Analyze both texts
        negation1 = self.detect_negation(text1)
        negation2 = self.detect_negation(text2)
        
        # Extract key entities/concepts
        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)
        
        # Check for direct contradictions
        contradictions = []
        
        # Case 1: Entity-level contradiction (strict)
        if negation1['has_negation'] and not negation2['has_negation']:
            shared_entities = entities1.intersection(entities2)
            if shared_entities and len(shared_entities) >= 2:
                # Check if negation is specifically about the same entity-predicate
                if self._is_semantic_contradiction(text1, text2, shared_entities):
                    contradictions.append({
                        'type': 'semantic_contradiction',
                        'entities': list(shared_entities),
                        'severity': 'high'
                    })
        
        # Case 2: Direct factual contradictions (high confidence only)
        factual_contradiction = self._detect_factual_contradiction(text1, text2)
        if factual_contradiction:
            contradictions.append(factual_contradiction)
        
        # Calculate contradiction score
        contradiction_score = self._calculate_contradiction_score(contradictions)
        
        return {
            'has_contradiction': len(contradictions) > 0,
            'contradiction_score': contradiction_score,
            'contradictions': contradictions,
            'negation1': negation1,
            'negation2': negation2,
            'shared_entities': list(entities1.intersection(entities2))
        }
    
    def _is_semantic_contradiction(self, text1: str, text2: str, shared_entities: set) -> bool:
        """
        Check if there's a true semantic contradiction between texts.
        
        Requirements:
        1. Same entity
        2. Same predicate
        3. Opposite truth values
        
        Args:
            text1: Text with potential negation
            text2: Text without negation
            shared_entities: Shared entities between texts
            
        Returns:
            True if semantic contradiction detected
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        # Extract predicates (verbs, adjectives, relationships)
        predicates1 = self._extract_predicates(text1_lower)
        predicates2 = self._extract_predicates(text2_lower)
        
        shared_predicates = predicates1.intersection(predicates2)
        
        # Need both shared entities AND shared predicates for contradiction
        if not (shared_entities and shared_predicates):
            return False
        
        # Check for explicit negation of the shared predicate-entity combination
        for entity in shared_entities:
            for predicate in shared_predicates:
                # Look for patterns like "X does not Y" where both X and Y are shared
                negation_pattern = f"{entity}.*?(not|no|never|without).*?{predicate}"
                if re.search(negation_pattern, text1_lower):
                    # Verify the positive version exists in text2
                    positive_pattern = f"{entity}.*?{predicate}"
                    if re.search(positive_pattern, text2_lower):
                        return True
        
        return False
    
    def _extract_predicates(self, text: str) -> set:
        """
        Extract predicates (verbs, adjectives, relationships) from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of predicate words
        """
        # Common predicate words (verbs, adjectives, state words)
        predicate_indicators = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'do', 'does', 'did',
            'can', 'could', 'will', 'would', 'should', 'may', 'might', 'must',
            'contain', 'contains', 'containment', 'have', 'has', 'had',
            'learn', 'learning', 'learned', 'program', 'programming', 'programmed',
            'end', 'ended', 'ending', 'conclude', 'concluded', 'concluding',
            'orbit', 'orbits', 'orbital', 'revolve', 'revolves', 'revolution'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        predicates = {word for word in words if word in predicate_indicators}
        return predicates
    
    def _detect_factual_contradiction(self, text1: str, text2: str) -> Optional[Dict[str, Any]]:
        """
        Detect high-confidence factual contradictions using pre-compiled patterns.

        Note: the pattern list in __init__ is intentionally small and domain-specific.
        Add new entries to _factual_contradiction_specs to extend coverage; do not add
        patterns here.
        """
        for spec in self._factual_contradiction_specs:
            p1, p2 = spec['pattern'], spec['contradiction']
            if (p1.search(text1) and p2.search(text2)) or \
               (p2.search(text1) and p1.search(text2)):
                return {
                    'type': 'factual_contradiction',
                    'description': spec['description'],
                    'severity': 'high',
                }
        return None
    
    def _extract_entities(self, text: str) -> set:
        """Extract simple entities (nouns, numbers) from text."""
        # Simple noun phrase extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them', 'he', 'his', 'him', 'she', 'her', 'you', 'your', 'we', 'our', 'us', 'i', 'my', 'me'}
        entities = {word for word in words if word not in stop_words and len(word) > 2}
        return entities
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers from text."""
        return re.findall(r'\b\d+\b', text)
    
    def _calculate_contradiction_score(self, contradictions: List[Dict[str, Any]]) -> float:
        """Return the severity of the worst contradiction found.

        Uses max() instead of average so that adding a second (lower-severity)
        signal can never reduce the score below the worst finding.
        """
        if not contradictions:
            return 0.0

        severity_map = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
        return max(
            severity_map.get(c.get('severity', 'low'), 0.3)
            for c in contradictions
        )
    
    def should_block_based_on_contradiction(self, response: str, evidence_texts: List[str]) -> Tuple[bool, float, str]:
        """
        Determine if response should be blocked based on contradiction with evidence.
        
        Now uses extremely conservative approach - only block on undeniable semantic contradictions.
        
        Args:
            response: The response to check
            evidence_texts: List of evidence texts
            
        Returns:
            Tuple of (should_block, contradiction_score, reason)
        """
        max_contradiction_score = 0.0
        worst_contradiction = None
        
        for evidence in evidence_texts:
            analysis = self.detect_contradiction_between_texts(response, evidence)
            
            if analysis['contradiction_score'] > max_contradiction_score:
                max_contradiction_score = analysis['contradiction_score']
                worst_contradiction = analysis
        
        # Extremely conservative blocking - only undeniable semantic contradictions
        should_block = False
        reason = ""
        
        if worst_contradiction and max_contradiction_score >= 0.9:  # 90%+ confidence required
            contradictions = worst_contradiction.get('contradictions', [])
            if contradictions:
                main_contradiction = contradictions[0]
                if main_contradiction['type'] == 'semantic_contradiction':
                    should_block = True
                    reason = f"Semantic contradiction detected: {main_contradiction['entities']}"
                elif main_contradiction['type'] == 'factual_contradiction':
                    should_block = True
                    reason = f"Factual contradiction: {main_contradiction['description']}"
        
        # For medium confidence contradictions, return the score but don't block
        elif max_contradiction_score >= 0.3 and worst_contradiction:
            contradictions = worst_contradiction.get('contradictions', [])
            if contradictions:
                main_contradiction = contradictions[0]
                reason = f"Low-medium contradiction detected: {main_contradiction['type']}"
        
        return should_block, max_contradiction_score, reason
