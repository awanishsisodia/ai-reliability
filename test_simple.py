#!/usr/bin/env python3
"""
Simple test to verify ai_reliability package functionality.
"""

import sys
import os
import importlib.util

def test_ai_reliability():
    """Test the ai_reliability package"""
    
    print("🚀 Testing AI Reliability Engine")
    print("=" * 40)
    
    # Find the package
    ai_reliability_path = "/Users/awanishkumar/Desktop/mlops/AIobservability/ai_reliability"
    if not os.path.exists(ai_reliability_path):
        print("❌ Package not found")
        return False
    
    # Add to Python path
    sys.path.insert(0, ai_reliability_path)
    
    # Register the package
    init_file = os.path.join(ai_reliability_path, '__init__.py')
    spec = importlib.util.spec_from_file_location('ai_reliability', init_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules['ai_reliability'] = module
    spec.loader.exec_module(module)
    
    try:
        # Test imports
        import ai_reliability
        print(f"✅ Import successful - Version: {ai_reliability.__version__}")
        
        from ai_reliability.core.engine import ReliabilityEngine
        from ai_reliability.core.config import ReliabilityConfig
        print("✅ Core imports successful")
        
        # Test basic functionality
        config = ReliabilityConfig(
            grounding={
                "max_latency_ms": 150.0,
                "support_threshold": 0.7,
                "allow_threshold": 0.85,
                "hedge_threshold": 0.65,
            }
        )
        print("✅ Configuration created")
        
        engine = ReliabilityEngine(config=config)
        print("✅ Engine created")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ai_reliability()
    sys.exit(0 if success else 1)
