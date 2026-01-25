# AI Reliability Engine

## Overview

The AI Reliability Engine provides **system-agnostic reliability evaluation** for AI responses through real-time grounding analysis. It computes explainable reliability scores with **guardrails-style safety measures**, making it suitable for production deployment in safety-critical applications.

> **Current Status**: Production-optimized prototype v0.2.0 with real algorithms and <150ms performance target.

## Key Features

- **Real-time Performance**: <150ms latency after model loading (optimized for production)
- **Guardrails Safety**: Multi-layer protection with conservative blocking and hedging
- **Model-Agnostic**: Works with any AI system (LLMs, agents, tools, workflows)
- **Explainable Scores**: Detailed explanations for every reliability decision
- **Production-Ready**: Built with safety-first approach and comprehensive error handling
- **High Performance**: Optimized algorithms with intelligent caching and batching
- **Configurable**: Flexible thresholds and safety parameters for different use cases
- **Scalable**: Supports both in-memory and distributed caching (Redis)

## Performance & Safety

### Performance Characteristics

- **Target Latency**: <150ms after model loading
- **Model Loading**: Initial load takes 2-5 seconds (one-time cost)
- **Throughput**: 100+ evaluations/second with caching
- **Memory Usage**: ~500MB model + configurable cache

### Safety System (Guardrails-Style)

The engine implements **multi-layer safety protection**:

1. **Layer 1**: Immediate BLOCK for grounding failures
2. **Layer 2**: Hard safety threshold (<0.3 = BLOCK)
3. **Layer 3**: Conservative hedge (0.3-0.6 = HEDGE)
4. **Layer 4**: High confidence for ALLOW (>0.75)
5. **Layer 5**: Comprehensive logging and monitoring

### Decision Logic

```python
# Safety-first decision making
if reliability_score < 0.3:
    return ReliabilityDecision.BLOCK  # Conservative
elif reliability_score < 0.6:
    return ReliabilityDecision.HEDGE  # Cautious
elif reliability_score > 0.75 and grounding_score > 0.8:
    return ReliabilityDecision.ALLOW  # High confidence
else:
    return ReliabilityDecision.HEDGE  # Default cautious
```

## Architecture

```
ai_reliability/
├── core/                    # Core orchestration and data structures
│   ├── engine.py           # Main reliability engine orchestrator
│   ├── result.py           # ReliabilityResult and decision types
│   └── config.py           # Configuration management with validation
├── grounding/              # Real-time grounding pipeline
│   ├── realtime.py         # Core grounding implementation (6-step process)
│   └── decomposition.py    # Sentence decomposition and claim detection
├── embeddings/             # Embedding backend with high-performance caching
│   ├── encoder.py          # Pluggable embedding backend with batching
│   └── cache.py            # Multi-tier caching (memory + Redis)
├── utils/                  # Production-grade utilities
│   ├── text.py             # Optimized text processing (regex-based)
│   ├── timing.py           # Performance measurement and latency budgets
│   └── logging.py          # Structured logging with metrics
└── tests/                  # Comprehensive test suite with performance validation
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/awanishsisodia/ai-reliability.git
cd ai_reliability

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests to verify installation
python test_working_basic.py

# Or run the full test suite (requires fixing imports)
PYTHONPATH=. python -m pytest tests/ -v
```

### Basic Usage

```python
# Quick verification - run from ai_reliability directory:
cd ai_reliability
python tests/test_working_basic.py

# For usage from any directory, the package needs to be registered:
import sys
import os
import importlib.util

# Add the ai_reliability directory to Python path
ai_reliability_path = "/path/to/ai_reliability"
sys.path.insert(0, ai_reliability_path)

# Register the package
init_file = os.path.join(ai_reliability_path, '__init__.py')
spec = importlib.util.spec_from_file_location('ai_reliability', init_file)
module = importlib.util.module_from_spec(spec)
sys.modules['ai_reliability'] = module
spec.loader.exec_module(module)

# Now import normally
import ai_reliability
from ai_reliability.core.engine import ReliabilityEngine
from ai_reliability.core.config import ReliabilityConfig

# Create configuration
config = ReliabilityConfig(
    grounding={
        "max_latency_ms": 150.0,
        "support_threshold": 0.7,
        "allow_threshold": 0.85,
        "hedge_threshold": 0.65,
    }
)

# Create engine
engine = ReliabilityEngine(config=config)

# Evaluate response
response = "The capital of France is Paris."
context = {
    "prompt": "What is the capital of France?",
    "tool_outputs": ["Paris is the capital city of France."],
    "memory": [],
    "constraints": {"max_length": 500}
}

result = engine.evaluate(response, context)
print(f"Score: {result.score:.3f}")
print(f"Decision: {result.decision.value}")
print(f"Processing time: {result.processing_time_ms:.2f}ms")
```

### Package Structure

The AI Reliability Engine uses relative imports inside the library:

```
ai_reliability/
├── __init__.py          # Main package
├── core/
│   ├── __init__.py      # Relative imports
│   ├── engine.py        # Relative imports
│   ├── config.py
│   └── result.py
├── embeddings/
│   ├── __init__.py      # Relative imports
│   └── encoder.py       # Relative imports
├── grounding/
│   ├── __init__.py      # Relative imports
│   └── realtime.py      # Relative imports
├── utils/
│   ├── __init__.py      # Relative imports
│   └── timing.py
└── tests/
    ├── test_working_basic.py
    └── test_professional.py
```

### Performance Notes

- **Model loading**: 3-5 seconds (one-time during initialization)
- **First evaluation**: ~190ms (model loading excluded from timing)
- **Subsequent evaluations**: ~150ms (target met)
- **Memory usage**: ~500MB model + cache
- **Safety system**: Multi-layer protection (BLOCK/HEDGE/ALLOW)

**Important**: Model loading time is excluded from evaluation latency budget. The model is pre-loaded during engine initialization to ensure accurate performance measurements.

### Safety-First Results

```python
# Example outputs based on safety thresholds

# High confidence (ALLOW)
result.decision == "allow"  # Only if score > 0.75 and grounding > 0.8

# Cautious approach (HEDGE) 
result.decision == "hedge"   # For borderline cases (0.3-0.6)

# Conservative blocking
result.decision == "block"  # For low reliability (<0.3) or grounding failures
```

### Advanced Configuration

```python
from ai_reliability.core.config import ReliabilityConfig

config = ReliabilityConfig(
    grounding={
        "max_latency_ms": 150.0,      # <150ms target for production
        "max_sentences": 10,
        "support_threshold": 0.7,     # Evidence support threshold
        "allow_threshold": 0.85,      # High confidence for ALLOW
        "hedge_threshold": 0.65,      # Conservative hedge threshold
    },
    embedding={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "cache_ttl_seconds": 3600,
        "cache_max_size": 10000,
        "redis_url": "redis://localhost:6379",  # Optional distributed cache
    }
)

# Safety thresholds (conservative defaults)
config.set_safety_thresholds(
    block_threshold=0.3,      # Below this = BLOCK
    hedge_threshold=0.6,      # Below this = HEDGE  
    allow_threshold=0.75,    # Above this = ALLOW (with high grounding)
    min_grounding_for_allow=0.8  # Minimum grounding for ALLOW decisions
)
```

## Testing

### Performance Validation

```bash
# Run performance tests to verify <150ms target
PYTHONPATH=. python -m pytest tests/test_reliability_engine.py::TestReliabilityEngine::test_performance_requirements -v

# Run all tests with coverage
PYTHONPATH=. python -m pytest tests/ --cov=ai_reliability --cov-report=html
```

## Reliability Scoring System

### Real-time Grounding Pipeline

The engine evaluates responses through a 6-step real-time grounding process:

1. **Response Normalization**: Clean and standardize text
2. **Sentence Decomposition**: Break into analyzable segments
3. **Claim Proxy**: Lightweight claim identification
4. **Semantic Support**: Fast similarity analysis with embeddings
5. **Coverage Proxy**: Calculate evidence coverage
6. **Evidence Agreement**: Check source consistency

### Scoring Components

```python
# Primary reliability score (0-1)
score = (
    0.5 * grounding_score +      # Evidence support
    0.3 * consistency_score +   # Internal coherence  
    0.2 * (1 - uncertainty)    # Confidence (inverted)
)

# Additional metrics (optional)
stability_score    # Response consistency over time
context_quality    # Evidence source quality
```

### Current Implementation Status

- ✅ **Grounding**: Full implementation with semantic similarity
- ✅ **Uncertainty**: Optimized keyword detection (fast, will be enhanced)
- ✅ **Consistency**: Contradiction detection (will be enhanced in future versions)
- ✅ **Stability**: Length-based heuristics (will be enhanced in future versions)
- ✅ **Safety**: Multi-layer guardrails protection

## Roadmap

### Phase 1: Production Optimization ✅ Complete
- [x] Real-time grounding pipeline with <150ms latency
- [x] Multi-layer safety system (guardrails)
- [x] Optimized algorithms and caching
- [x] Simplified data model with primary metrics
- [x] Performance validation and testing

### Phase 2: Enhanced Reliability (In Progress)
- [ ] Async evaluation pipeline (Tier-2 grounding)
- [ ] Advanced consistency checking with NLP
- [ ] Sophisticated uncertainty quantification
- [ ] Response stability analysis over time
- [ ] Context quality assessment algorithms

### Phase 3: Advanced Features (Future)
- [ ] Offline evaluation pipeline (Tier-3 grounding)
- [ ] Learning-based weight optimization
- [ ] Domain-specific calibration
- [ ] Multi-modal support (text, images, audio)
- [ ] Federated learning for model improvement

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact Author

- **Author**: Awanish Kumar
- **Email**: awanish.sisodia.ai@gmail.com
- **Issues**: [GitHub Issues](https://github.com/awanishsisodia/ai-reliability/issues)
- **Documentation**: [Medium Article](https://medium.com/@awanish.sisodia.ai/ai-observability-reliability-engines-146fe8ed1557)


