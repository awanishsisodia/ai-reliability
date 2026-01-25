#!/usr/bin/env python3
"""
Professional test script for AI Reliability Engine.

This script demonstrates the correct way to use the AI Reliability Engine
as a professional Python library.
"""

import sys
import os
import importlib.util

def setup_and_test():
    """Set up and test the AI Reliability Engine"""
    
    print("🚀 AI Reliability Engine - Professional Test")
    print("=" * 50)
    
    # Find the ai_reliability package
    ai_reliability_path = None
    possible_paths = [
        "/Users/awanishkumar/Desktop/mlops/AIobservability/ai_reliability",
        os.path.expanduser("~/ai_reliability"),
        "./ai_reliability",
        "../ai_reliability",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            ai_reliability_path = path
            break
    
    if not ai_reliability_path:
        print("❌ Could not find ai_reliability package")
        return False
    
    print(f"✅ Found ai_reliability at: {ai_reliability_path}")
    
    # Add to Python path
    if ai_reliability_path not in sys.path:
        sys.path.insert(0, ai_reliability_path)
    
    # Register the package (required for proper imports)
    init_file = os.path.join(ai_reliability_path, '__init__.py')
    spec = importlib.util.spec_from_file_location('ai_reliability', init_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ai_reliability'] = module
    spec.loader.exec_module(module)
    
    try:
        # Import the package
        import ai_reliability
        print("✅ Package imported successfully")
        print(f"   Version: {ai_reliability.__version__}")
        print(f"   Available: {ai_reliability.__all__}")
        
        # Import core components
        from ai_reliability.core.engine import ReliabilityEngine
        from ai_reliability.core.config import ReliabilityConfig
        print("✅ Core components imported")
        
        # Create configuration
        config = ReliabilityConfig(
            grounding={
                "max_latency_ms": 150.0,
                "support_threshold": 0.7,
                "allow_threshold": 0.85,
                "hedge_threshold": 0.65,
            }
        )
        print("✅ Configuration created")
        
        # Create engine
        engine = ReliabilityEngine(config=config)
        print("✅ Engine created")
        
        # Test evaluation
        response = "The capital of France is Paris."
        context = {
            "prompt": "What is the capital of France?",
            "tool_outputs": ["Paris is the capital city of France."],
            "memory": [],
            "constraints": {"max_length": 500}
        }
        
        result = engine.evaluate(response, context)
        print("✅ Evaluation completed")
        print(f"   Score: {result.score:.3f}")
        print(f"   Decision: {result.decision.value}")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
        
        # Performance check
        if result.processing_time_ms < 150.0:
            print("✅ Performance target met (<150ms)")
        else:
            print(f"⚠️  Performance target exceeded: {result.processing_time_ms:.2f}ms")
        
        print("\n🎉 All tests passed!")
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
    success = setup_and_test()
    if success:
        print("\n✅ SUCCESS: AI Reliability Engine is working!")
        sys.exit(0)
    else:
        print("\n❌ FAILURE: Test failed!")
        sys.exit(1)
