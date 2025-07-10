# HomeLLM-Bench

A comprehensive benchmark suite for evaluating small quantized LLMs in home environments.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/HomeLLM-Bench.git
cd HomeLLM-Bench

# Start vLLM server
./homellm-server --model microsoft/Phi-3.5-mini-instruct

# Run benchmarks (in another terminal)
./homellm-benchmark --model microsoft/Phi-3.5-mini-instruct
```

## Project Structure

```
HomeLLM-Bench/
├── homellm-benchmark      # CLI: Run benchmarks
├── homellm-server        # CLI: Start vLLM server
├── homellm_bench/        # Main package
│   ├── benchmark/        # Benchmark runner
│   ├── server/          # Server management
│   ├── config/          # Configuration
│   ├── utils/           # Utilities
│   ├── data/            # Test conversations
│   ├── engines/         # LLM engines
│   ├── metrics/         # Performance metrics
│   ├── output/          # Result formatters
│   ├── schemas/         # Data schemas
│   └── templates/       # Chat templates
├── results/             # Benchmark results
└── tests/              # Test files
```

## Features

- **Smart Server Detection** - Automatically detects running vLLM servers
- **Context-Aware Testing** - Selects appropriate conversations based on model context size
- **GPU Memory Management** - Dynamic memory allocation with safety margins
- **RAG Simulation** - Tests prefix caching with context injection/removal
- **Multi-turn Conversations** - Back-and-forth dialogue simulation
- **Comprehensive Error Handling** - Actionable error messages with specific solutions
- **Multiple Output Formats** - CSV, JSON, and Markdown reports
- **Resource Monitoring** - GPU, CPU, and memory usage tracking

## CLI Usage

### Server Management

```bash
# Start vLLM server with auto-detection
./homellm-server --model microsoft/Phi-3.5-mini-instruct

# Specify GPU memory usage
./homellm-server --model phi-3.5-mini-4bit --target-memory 6.0

# Use different port
./homellm-server --model phi-3.5-mini-4bit --port 8002

# Disable torch compilation for debugging
./homellm-server --model phi-3.5-mini-4bit --disable-torch-compilation
```

### Benchmark Execution

```bash
# Basic benchmark run
./homellm-benchmark --model microsoft/Phi-3.5-mini-instruct

# List available conversations
./homellm-benchmark --list-conversations --context-size 32000

# Run specific conversation types
./homellm-benchmark --include-tags rag,technical --context-size 32000

# Exclude certain conversations  
./homellm-benchmark --exclude-tags long --max-conversations 3

# Use different server port
./homellm-benchmark --model phi-3.5-mini-4bit --server-port 8002
```

## Available Conversation Types

The benchmark automatically selects appropriate conversations based on your model's context size:

### By Context Requirements
- **Simple Q&A** (~800 tokens) - Basic capabilities testing
- **Code Discussion** (~2.5K tokens) - Technical dialogue and code generation  
- **Deep Technical** (~7K tokens) - Long-form technical discussions
- **Ultra Long Context** (~15K tokens) - Maximum context utilization

### By Test Type
- **RAG Simulation** - Tests prefix caching with context changes
- **Multi-turn** - Extended conversations with context building
- **Technical** - Code and technical discussions
- **Creative** - Creative writing and storytelling

## Smart Server Detection

The benchmark runner automatically detects vLLM server status:

- ✅ **Server Ready** - Proceeds with benchmarks
- ⏳ **Server Starting** - Waits for server to become ready
- ❌ **No Server** - Prompts to start server with exact command
- 🚫 **Port Conflict** - Suggests alternative port

## Configuration

### System Requirements
- **GPU Memory**: 6GB+ for 4-bit models, 12GB+ for full precision
- **System RAM**: 8GB+ recommended
- **Python**: 3.8+
- **CUDA**: 11.8+ (for GPU acceleration)

### Dependencies
```bash
pip install vllm pydantic requests psutil pynvml
```

## Results Output

Results are saved in the `results/` directory with timestamps:

- **`*.csv`** - Machine-readable metrics for analysis
- **`*.json`** - Complete benchmark data with metadata
- **`*.md`** - Human-readable report with summaries
- **`*_system.json`** - System configuration and specs

## Error Handling

The framework provides actionable error messages:

```bash
❌ Error: Model 'invalid-model' not found
💡 Solution: Check model name or add to model registry

❌ Error: No vLLM server detected on port 8001  
💡 Solution: Start vLLM server first: ./homellm-server --model your-model

❌ Error: GPU out of memory during inference
💡 Solution: Reduce context size or use smaller model
```

## Advanced Usage

### Custom Model Configuration

Add new models by extending the model registry in `homellm_bench/config/model_config.py`:

```python
@dataclass
class ModelConfig:
    model_type: ModelType
    context_size: int
    chat_template: str
    quantization: Optional[str] = None
    estimated_size_gb: float = 2.0
```

### Custom Chat Templates

Create new templates in `homellm_bench/templates/` following the base template pattern.

### Adding Test Conversations

Extend `homellm_bench/data/conversations.json` with new conversation templates.

## Troubleshooting

### Common Issues

**Server won't start:**
- Check GPU memory with `nvidia-smi`
- Verify model path exists
- Try smaller model or reduce target memory

**Benchmark fails:**
- Ensure vLLM server is running
- Check network connectivity
- Verify model compatibility

**Out of memory errors:**
- Reduce context size: `--context-size 8192`
- Use smaller model variant
- Lower target memory: `--target-memory 4.0`

## Development

### Package Structure

The project follows Python package conventions with relative imports:

```python
# Example import structure
from homellm_bench.benchmark.runner import BenchmarkRunner
from homellm_bench.server.manager import ServerManager
```

### Running Tests

```bash
python -m pytest tests/
```

### Contributing

1. Follow the existing code structure
2. Use relative imports within the package
3. Add comprehensive error handling
4. Include docstrings for new functions
5. Test with multiple model types