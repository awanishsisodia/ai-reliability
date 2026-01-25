#!/usr/bin/env python3
"""
Basic usage example for the AI Reliability Engine.

This example demonstrates how to use the reliability engine to evaluate
AI responses with real-time grounding.
"""

import time
from typing import Dict, Any

from ai_reliability import ReliabilityEngine, ReliabilityConfig


def main():
    """Run basic usage example."""
    print("🔍 AI Reliability Engine - Basic Usage Example")
    print("=" * 50)
    
    # Initialize the reliability engine
    print("\n📦 Initializing reliability engine...")
    config = ReliabilityConfig.from_env()  # Can also customize manually
    engine = ReliabilityEngine(config=config)
    
    print(f"✅ Engine initialized with model: {config.embedding.model_name}")
    print(f"⏱️  Grounding budget: {config.grounding.max_latency_ms}ms")
    
    # Example 1: Well-supported response
    print("\n📝 Example 1: Well-supported response")
    print("-" * 40)
    
    response1 = "Paris is the capital of France. The Eiffel Tower is located there and it's a famous landmark."
    context1 = {
        "prompt": "Tell me about Paris, France.",
        "tool_outputs": [
            "Paris is the capital city of France.",
            "The Eiffel Tower is a famous landmark located in Paris.",
            "Paris is known for its art, fashion, and culture."
        ],
        "memory": [],
        "constraints": {"max_length": 200}
    }
    
    print(f"Response: {response1}")
    print(f"Evidence sources: {len(context1['tool_outputs'])} tool outputs")
    
    start_time = time.time()
    result1 = engine.evaluate(response1, context1)
    end_time = time.time()
    
    print(f"\n📊 Results:")
    print(f"   Overall Score: {result1.score:.3f}")
    print(f"   Grounding Score: {result1.grounding:.3f}")
    print(f"   Decision: {result1.decision.value}")
    print(f"   Processing Time: {result1.processing_time_ms:.2f}ms")
    print(f"   Coverage: {result1.explanation.coverage:.3f}")
    print(f"   Mean Support: {result1.explanation.mean_support:.3f}")
    print(f"   Unsupported Sentences: {len(result1.explanation.unsupported_sentences)}")
    
    # Example 2: Unsupported response
    print("\n📝 Example 2: Unsupported response")
    print("-" * 40)
    
    response2 = "The capital of Mars is New York City with a population of 50 billion people and they drive flying cars."
    context2 = {
        "prompt": "Tell me about Mars.",
        "tool_outputs": [
            "Mars is the fourth planet from the Sun.",
            "Mars has a thin atmosphere and is cold.",
            "Mars has two small moons: Phobos and Deimos."
        ],
        "memory": [],
        "constraints": {"max_length": 200}
    }
    
    print(f"Response: {response2}")
    print(f"Evidence sources: {len(context2['tool_outputs'])} tool outputs")
    
    start_time = time.time()
    result2 = engine.evaluate(response2, context2)
    end_time = time.time()
    
    print(f"\n📊 Results:")
    print(f"   Overall Score: {result2.score:.3f}")
    print(f"   Grounding Score: {result2.grounding:.3f}")
    print(f"   Decision: {result2.decision.value}")
    print(f"   Processing Time: {result2.processing_time_ms:.2f}ms")
    print(f"   Coverage: {result2.explanation.coverage:.3f}")
    print(f"   Mean Support: {result2.explanation.mean_support:.3f}")
    print(f"   Unsupported Sentences: {len(result2.explanation.unsupported_sentences)}")
    
    if result2.explanation.unsupported_sentences:
        print(f"   Unsupported: {result2.explanation.unsupported_sentences[0][:50]}...")
    
    # Example 3: Mixed response
    print("\n📝 Example 3: Mixed support response")
    print("-" * 40)
    
    response3 = "Paris is the capital of France. The population is 10 million people. The city was founded in 5000 BC by aliens."
    context3 = {
        "prompt": "Tell me about Paris.",
        "tool_outputs": [
            "Paris is the capital city of France.",
            "The Paris metropolitan area has about 2.1 million people.",
            "Paris was founded around the 3rd century BC by a Celtic people."
        ],
        "memory": [],
        "constraints": {"max_length": 200}
    }
    
    print(f"Response: {response3}")
    print(f"Evidence sources: {len(context3['tool_outputs'])} tool outputs")
    
    start_time = time.time()
    result3 = engine.evaluate(response3, context3)
    end_time = time.time()
    
    print(f"\n📊 Results:")
    print(f"   Overall Score: {result3.score:.3f}")
    print(f"   Grounding Score: {result3.grounding:.3f}")
    print(f"   Decision: {result3.decision.value}")
    print(f"   Processing Time: {result3.processing_time_ms:.2f}ms")
    print(f"   Coverage: {result3.explanation.coverage:.3f}")
    print(f"   Mean Support: {result3.explanation.mean_support:.3f}")
    print(f"   Unsupported Sentences: {len(result3.explanation.unsupported_sentences)}")
    
    # Show sentence-level details
    print(f"\n🔍 Sentence-level details:")
    for i, sentence_score in enumerate(result3.explanation.sentence_scores):
        print(f"   Sentence {i+1}: {sentence_score['support']:.3f} support "
              f"({'✓' if sentence_score['is_supported'] else '✗'}) "
              f"{'[claim]' if sentence_score['is_claim'] else '[info]'}")
        print(f"      {sentence_score['sentence'][:60]}...")
    
    # Performance statistics
    print("\n📈 Performance Statistics")
    print("-" * 40)
    
    stats = engine.get_performance_stats()
    if "reliability_total" in stats:
        rel_stats = stats["reliability_total"]
        print(f"   Total Evaluations: {rel_stats['count']}")
        print(f"   Average Time: {rel_stats['mean_ms']:.2f}ms")
        print(f"   Min Time: {rel_stats['min_ms']:.2f}ms")
        print(f"   Max Time: {rel_stats['max_ms']:.2f}ms")
        print(f"   P95 Time: {rel_stats['p95_ms']:.2f}ms")
    
    # Cache statistics
    cache_stats = engine.encoder.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"   Cached Embeddings: {cache_stats['memory_cache_size']}")
    print(f"   Cache Limit: {cache_stats['memory_cache_limit']}")
    print(f"   TTL: {cache_stats['ttl_seconds']}s")
    
    print("\n✅ Example completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   • Well-supported responses get high scores and ALLOW decisions")
    print("   • Unsupported responses get low scores and BLOCK decisions")
    print("   • Mixed support results in HEDGE decisions")
    print("   • Processing stays within 50ms latency budget")
    print("   • Caching improves performance for repeated evaluations")


if __name__ == "__main__":
    main()
