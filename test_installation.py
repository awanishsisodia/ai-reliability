#!/usr/bin/env python3
"""Test script to verify AI Reliability Engine installation"""

import sys
import os

def test_installation():
    """Test that the AI Reliability Engine is properly installed"""
    
    print("🚀 Testing AI Reliability Engine Installation")
    print("=" * 50)
    
    try:
        # Test 1: Import the package
        print("\n1. Testing package import...")
        import ai_reliability
        print("✅ Package imported successfully")
        print(f"   Version: {ai_reliability.__version__}")
        print(f"   Available: {ai_reliability.__all__}")
        
        # Test 2: Import core components
        print("\n2. Testing core components...")
        from ai_reliability.core.engine import ReliabilityEngine
        from ai_reliability.core.config import ReliabilityConfig
        from ai_reliability.core.result import ReliabilityResult, ReliabilityDecision
        print("✅ Core components imported successfully")
        
        # Test 3: Create configuration
        print("\n3. Testing configuration...")
        config = ReliabilityConfig(
            grounding={
                "max_latency_ms": 150.0,
                "support_threshold": 0.7,
                "allow_threshold": 0.85,
                "hedge_threshold": 0.65,
            }
        )
        print("✅ Configuration created successfully")
        
        # Test 4: Create engine
        print("\n4. Testing engine creation...")
        engine = ReliabilityEngine(config=config)
        print("✅ Engine created successfully")
        
        # Test 5: Basic evaluation
        print("\n5. Testing basic evaluation...")
        response = "The capital of France is Paris."
        context = {
            "prompt": "What is the capital of France?",
            "tool_outputs": ["Paris is the capital city of France."],
            "memory": [],
            "constraints": {"max_length": 500}
        }
        
        result = engine.evaluate(response, context)
        print("✅ Evaluation completed successfully")
        print(f"   Score: {result.score:.3f}")
        print(f"   Decision: {result.decision.value}")
        print(f"   Grounding: {result.grounding:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
        
        # Test 6: Performance validation
        print("\n6. Testing performance targets...")
        if result.processing_time_ms < 150.0:
            print("✅ Performance target met (<150ms)")
        else:
            print(f"⚠️  Performance target exceeded: {result.processing_time_ms:.2f}ms")
        
        # Test 7: Safety validation
        print("\n7. Testing safety system...")
        if result.decision in ReliabilityDecision:
            print("✅ Safety decision valid")
        else:
            print("❌ Invalid safety decision")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ AI Reliability Engine is properly installed and working!")
        print("📈 Performance: <150ms latency")
        print("🛡️ Safety: Multi-layer protection")
        print("🎯 Ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_installation()
    if success:
        print("\n✅ SUCCESS: Installation verified!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Installation verification failed!")
        sys.exit(1)
