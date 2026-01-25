"""
Text processing utilities for the AI Reliability Engine.

This module provides optimized text processing functions designed for
real-time performance with strict latency requirements.
"""

from __future__ import annotations

import re
from typing import List

# Pre-compiled regex patterns for performance
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=[.!?])\s*$|(?<=[.!?])\s+(?=[a-z])',
    flags=re.MULTILINE
)

WHITESPACE_PATTERN = re.compile(r'\s+')
MARKDOWN_PATTERN = re.compile(
    r'[#*_`~\[\](){}]|```[\s\S]*?```|`[^`]*`|\*[^*]*\*|_[^_]*_'
)

CLAIM_INDICATORS = {
    # Entity mentions (capitalized words that aren't at start)
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
    # Numbers and quantities
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|million|billion|thousand|times|degrees|celsius|fahrenheit)\b',
    r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP|dollars|euros|pounds)\b',
    # Categorical assertions
    r'\b(is|are|was|were|will be|shall be|has been|have been)\s+(?:not\s+)?(?:the|a|an)\s+\w+',
    r'\b(?:always|never|only|exclusively|solely)\b',
    # Action verbs
    r'\b(?:achieved|completed|failed|succeeded|reached|exceeded|improved|worsened|increased|decreased)\b',
    # Superlatives and comparatives
    r'\b(?:best|worst|highest|lowest|fastest|slowest|most|least|better|worse|higher|lower)\b',
}

# Compile claim indicator patterns
CLAIM_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in CLAIM_INDICATORS]


def normalize_response(response: str, max_length: int = 5000) -> str:
    """
    Normalize response text for reliable processing.
    
    Performs:
    - Markdown removal
    - Whitespace normalization  
    - Length truncation
    - Basic sanitization
    
    Args:
        response: Raw response text
        max_length: Maximum allowed length after normalization
        
    Returns:
        Normalized text ready for processing
    """
    if not response:
        return ""
    
    # Remove markdown formatting
    text = MARKDOWN_PATTERN.sub(' ', response)
    
    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0]  # Don't cut mid-word
    
    return text


def split_into_sentences(text: str, max_sentences: int = 10) -> List[str]:
    """
    Split text into sentences with performance optimization.
    
    Uses regex-based splitting for speed over NLP libraries.
    Handles common edge cases and respects sentence limits.
    
    Args:
        text: Input text to split
        max_sentences: Maximum number of sentences to return
        
    Returns:
        List of sentences, ordered and cleaned
    """
    if not text:
        return []
    
    # Basic sentence splitting using regex
    raw_sentences = SENTENCE_SPLIT_PATTERN.split(text)
    
    # Clean and filter sentences
    sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filter very short fragments
            sentences.append(sentence)
    
    # Limit number of sentences
    return sentences[:max_sentences]


def extract_claim_like_sentences(sentences: List[str]) -> List[str]:
    """
    Extract sentences that contain claim-like indicators.
    
    This is a lightweight proxy for full claim extraction.
    Looks for entities, numbers, categorical assertions, and action verbs.
    
    Args:
        sentences: List of sentences to analyze
        
    Returns:
        Sentences that contain claim indicators
    """
    claim_sentences = []
    
    for sentence in sentences:
        # Check if sentence matches any claim indicator pattern
        is_claim = any(pattern.search(sentence) for pattern in CLAIM_PATTERNS)
        
        # Additional heuristic: sentences with specific punctuation or structure
        has_colon = ':' in sentence
        has_semicolon = ';' in sentence
        has_quotes = '"' in sentence or "'" in sentence
        
        if is_claim or has_colon or has_semicolon or has_quotes:
            claim_sentences.append(sentence)
    
    return claim_sentences


def count_entities(sentence: str) -> int:
    """
    Count entity mentions in a sentence.
    
    Simple heuristic based on capitalized words.
    
    Args:
        sentence: Sentence to analyze
        
    Returns:
        Number of entity mentions found
    """
    # Find capitalized words (excluding first word)
    capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
    # Adjust for first word being capitalized
    if sentence and sentence[0].isupper():
        capitalized_words = capitalized_words[1:]
    
    return len(capitalized_words)


def count_numbers(sentence: str) -> int:
    """
    Count numerical values in a sentence.
    
    Args:
        sentence: Sentence to analyze
        
    Returns:
        Number of numerical values found
    """
    # Find numbers with optional commas and decimals
    numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', sentence)
    return len(numbers)


def is_question(sentence: str) -> bool:
    """
    Check if sentence is a question.
    
    Questions are typically less risky for grounding.
    
    Args:
        sentence: Sentence to check
        
    Returns:
        True if sentence appears to be a question
    """
    return sentence.strip().endswith('?') or sentence.strip().startswith(('Who', 'What', 'When', 'Where', 'Why', 'How', 'Is', 'Are', 'Do', 'Does', 'Can', 'Could', 'Will', 'Would', 'Should'))


def get_sentence_complexity(sentence: str) -> float:
    """
    Estimate sentence complexity for risk assessment.
    
    Higher complexity may indicate higher risk.
    
    Args:
        sentence: Sentence to analyze
        
    Returns:
        Complexity score [0,1], higher = more complex
    """
    if not sentence:
        return 0.0
    
    # Base complexity factors
    word_count = len(sentence.split())
    entity_count = count_entities(sentence)
    number_count = count_numbers(sentence)
    
    # Normalize factors
    complexity = 0.0
    
    # Word count contribution (capped at 30 words)
    word_factor = min(word_count / 30.0, 1.0) * 0.3
    
    # Entity density contribution
    entity_factor = min(entity_count / 5.0, 1.0) * 0.4
    
    # Number density contribution  
    number_factor = min(number_count / 3.0, 1.0) * 0.3
    
    complexity = word_factor + entity_factor + number_factor
    
    return min(complexity, 1.0)
