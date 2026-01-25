# AI Reliability Engine

## Overview

The AI Reliability Engine provides **system-agnostic reliability evaluation** for AI responses through real-time grounding analysis. It computes explainable reliability scores with strict performance requirements, making it suitable for production deployment in safety-critical applications.

## Key Features

- **Real-time Performance**: ≤50ms latency for grounding evaluation
- **Model-Agnostic**: Works with any AI system (LLMs, agents, tools, workflows)
- **Explainable Scores**: Detailed explanations for every reliability decision
- **Production-Ready**: Built with safety, monitoring, and comprehensive error handling
- **High Performance**: Intelligent caching, batching, and optimized algorithms
- **Configurable**: Flexible thresholds and weights for different use cases
- **Scalable**: Supports both in-memory and distributed caching (Redis)

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
└── tests/                  # Comprehensive test suite with benchmarks
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
PYTHONPATH=. python -m pytest tests/ -v
```

### Basic Usage

```python
from ai_reliability.core.engine import ReliabilityEngine
from ai_reliability.core.config import ReliabilityConfig

# Initialize the engine with default configuration
engine = ReliabilityEngine()

# Evaluate a response with context
response = "Paris is the capital of France. The Eiffel Tower is located there."
context = {
    "prompt": "Tell me about Paris.",
    "tool_outputs": [
        "Paris is the capital city of France.",
        "The Eiffel Tower is a famous landmark in Paris."
    ],
    "memory": [],
    "constraints": {"max_length": 200}
}

result = engine.evaluate(response, context)

print(f"Reliability Score: {result.score:.3f}")
print(f"Decision: {result.decision.value}")
print(f"Grounding Score: {result.grounding:.3f}")
print(f"Processing Time: {result.processing_time_ms:.2f}ms")

# Access detailed explanations
for explanation in result.explanations:
    print(f"- {explanation.component}: {explanation.description}")
```

### Advanced Configuration

```python
from ai_reliability.core.config import ReliabilityConfig

config = ReliabilityConfig(
    grounding={
        "max_latency_ms": 50.0,        # Latency budget for real-time grounding
        "max_sentences": 10,            # Maximum sentences to evaluate
        "support_threshold": 0.7,       # Minimum support for sentences
        "allow_threshold": 0.85,        # Score threshold for ALLOW decision
        "hedge_threshold": 0.65,        # Score threshold for HEDGE decision
        "support_weight": 0.5,          # Weight for semantic support
        "coverage_weight": 0.3,         # Weight for coverage calculation
        "agreement_weight": 0.2,        # Weight for evidence agreement
    },
    embedding={
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "max_sequence_length": 512,
        "cache_ttl_seconds": 3600,
        "cache_max_size": 10000,
        "redis_url": "redis://localhost:6379",  # Optional distributed cache
    },
    log_level="INFO",
    enable_metrics=True,
    metrics_port=9090
)

engine = ReliabilityEngine(config=config)
```

## Reliability Scoring System

### Real-time Grounding Pipeline

The engine evaluates responses through a 6-step real-time grounding process:

1. **Response Normalization**: Clean and standardize text
2. **Sentence Decomposition**: Break into analyzable segments
3. **Semantic Support Check**: Compare sentences with evidence using embeddings
4. **Coverage Calculation**: Measure fraction of supported sentences
5. **Evidence Agreement**: Check consistency between evidence sources
6. **Decision Logic**: Apply thresholds to determine reliability

### Scoring Components

#### Grounding Score (Real-time, 30% weight)
- **Semantic Support**: Maximum cosine similarity between sentences and evidence
- **Coverage**: Fraction of sentences with adequate support (>0.7 similarity)
- **Evidence Agreement**: Consistency score between multiple evidence sources

#### Future Components (Planned)
- **Consistency** (25%): Internal consistency analysis
- **Uncertainty** (20%): Uncertainty quantification (inverted for reliability)
- **Stability** (15%): Response stability over time
- **Context Quality** (10%): Quality assessment of available context

### Decision Logic

| Score Range | Decision | Action |
|-------------|----------|--------|
| `score ≥ 0.85` | **ALLOW** | Response is safe to show to users |
| `0.65 ≤ score < 0.85` | **HEDGE** | Response needs qualification or context |
| `score < 0.65` | **BLOCK** | Response should not be shown |

## Performance Optimization

### Latency Targets

| Component | Target Latency | Typical Performance |
|-----------|----------------|-------------------|
| Text Normalization | 1ms | 0.5ms |
| Sentence Segmentation | 5ms | 2ms |
| Embedding Computation | 20ms | 15ms |
| Support Scoring | 15ms | 10ms |
| Coverage & Agreement | 4ms | 2ms |
| **Total** | **≤50ms** | **~30ms** |

### Caching Strategy

The engine implements multi-tier caching for optimal performance:

```python
# In-memory cache (default)
cache_stats = engine.encoder.get_cache_stats()
print(f"Memory cache hits: {cache_stats['memory_cache_hits']}")

# Distributed cache with Redis (optional)
config = ReliabilityConfig(
    embedding={"redis_url": "redis://localhost:6379"}
)
```

### Performance Monitoring

```python
# Get detailed performance statistics
stats = engine.get_performance_stats()

for operation, metrics in stats.items():
    print(f"{operation}:")
    print(f"  Count: {metrics['count']}")
    print(f"  Mean: {metrics['mean_ms']:.2f}ms")
    print(f"  Min: {metrics['min_ms']:.2f}ms")
    print(f"  Max: {metrics['max_ms']:.2f}ms")
```

## Testing and Quality Assurance

### Running Tests

```bash
# Run all tests with coverage
PYTHONPATH=. python -m pytest tests/ -v --cov=ai_reliability

# Run specific test categories
PYTHONPATH=. python -m pytest tests/test_reliability_engine.py -v
PYTHONPATH=. python -m pytest tests/test_grounding.py -v
PYTHONPATH=. python -m pytest tests/test_embeddings.py -v

# Run performance benchmarks
PYTHONPATH=. python -m pytest tests/ --benchmark-only

# Run with detailed output
PYTHONPATH=. python -m pytest tests/ -v -s
```

### Test Coverage

The test suite includes:
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Latency and throughput validation
- **Edge Case Tests**: Error handling and boundary conditions
- **Benchmark Tests**: Performance regression detection

### Quality Metrics

- **Code Coverage**: >90% target
- **Performance Benchmarks**: Automated regression detection
- **Type Safety**: Full mypy compliance
- **Code Quality**: Black formatting + Ruff linting


### Environment Configuration

Copy the provided `.env.example` file to `.env` and adjust the values:

```bash
cp .env.example .env
```

Key environment variables:

```bash
# Core configuration
AI_RELIABILITY_MAX_LATENCY_MS=50
AI_RELIABILITY_SUPPORT_THRESHOLD=0.7
AI_RELIABILITY_ALLOW_THRESHOLD=0.85

# Embedding settings
AI_RELIABILITY_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AI_RELIABILITY_REDIS_URL=redis://redis:6379

# Monitoring
AI_RELIABILITY_LOG_LEVEL=INFO
AI_RELIABILITY_ENABLE_METRICS=true
AI_RELIABILITY_METRICS_PORT=9090
```

See `.env.example` for a complete list of all available configuration options.

### ReliabilityEngine

```python
class ReliabilityEngine:
    """Main reliability evaluation engine."""
    
    def __init__(self, config: ReliabilityConfig = None, encoder: EmbeddingEncoder = None):
        """Initialize the reliability engine."""
    
    def evaluate(self, response: str, context: Dict[str, Any]) -> ReliabilityResult:
        """Evaluate a response with given context."""
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
    
    def warm_up_cache(self, texts: List[str]) -> None:
        """Warm up cache with common texts."""
```

### ReliabilityResult

```python
class ReliabilityResult(BaseModel):
    """Complete reliability evaluation result."""
    
    score: float                    # Overall reliability score (0-1)
    decision: ReliabilityDecision    # ALLOW/HEDGE/BLOCK decision
    grounding: float               # Grounding score
    consistency: float             # Consistency score (placeholder)
    uncertainty: float             # Uncertainty score (placeholder)
    stability: float               # Stability score (placeholder)
    context_quality: float         # Context quality score
    explanations: List[ReliabilityExplanation]  # Detailed explanations
    processing_time_ms: float      # Total processing time
    sentence_scores: List[Dict]    # Per-sentence analysis
    evidence_sources: List[str]    # Evidence sources used
    warnings: List[str]            # Any warnings generated
```

### Configuration Classes

```python
class ReliabilityConfig(BaseModel):
    """Main configuration for the reliability engine."""
    
    grounding: GroundingConfig      # Grounding pipeline configuration
    embedding: EmbeddingConfig      # Embedding backend configuration
    log_level: str                  # Logging level
    enable_metrics: bool           # Enable Prometheus metrics
    metrics_port: int              # Metrics port
```

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/awanishsisodia/ai-reliability.git
cd ai_reliability

# Set up development environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run all checks
uv run ruff check
uv run black --check .
uv run mypy ai_reliability
PYTHONPATH=. python -m pytest tests/
```

### Code Quality Standards

- **Formatting**: Black (88 character line length)
- **Linting**: Ruff with strict rules
- **Type Checking**: MyPy with strict mode
- **Testing**: pytest with >90% coverage
- **Documentation**: Full docstring coverage

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Run the full test suite (`PYTHONPATH=. python -m pytest tests/`)
5. Ensure code quality checks pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Roadmap

### Phase 1: Real-time Grounding (✅ Complete)
- [x] Real-time grounding pipeline with ≤50ms latency
- [x] Production-grade embedding backend with caching
- [x] Comprehensive test suite with benchmarks
- [x] Performance optimization and monitoring

### Phase 2: Enhanced Reliability (In Progress)
- [ ] Async evaluation pipeline (Tier-2 grounding)
- [ ] Internal consistency checking
- [ ] Uncertainty quantification methods
- [ ] Response stability analysis
- [ ] Context quality assessment

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


