# LLM Benchmark Framework

Benchmark framework for testing LLM models with focus on single-user local inference.

## Features

- Generic prompts compatible with different chat templates
- Metrics collection (GPU, CPU, memory, generation performance)
- Process isolation for clean benchmarking between tests
- Multiple output formats (CSV, JSON, Markdown)
- Automatic model downloading
- Prefix caching effectiveness measurement
- vLLM optimization for single-user scenarios
- Context-aware test selection based on model capabilities
- Separated test data from code
- Multi-turn conversations with back-and-forth dialogue
- RAG simulation to test prefix caching strategies
- Long context tests up to 128K tokens

## Quick Start

### 1. Setup Environment

```bash
# Download and setup the recommended model
python download_models.py --setup
```

### 2. Run Benchmark

```bash
# Run benchmark with context awareness
python enhanced_benchmark_runner.py

# List available conversations for your model
python enhanced_benchmark_runner.py --list-conversations --context-size 128000

# Run specific test types
python enhanced_benchmark_runner.py --include-tags rag,long --context-size 32000

# Limit number of conversations
python enhanced_benchmark_runner.py --max-conversations 3
```

### 3. View Results

Results are saved in the `results/` directory:
- `*.csv` - Machine-readable metrics
- `*.json` - Detailed benchmark data
- `*.md` - Human-readable report
- `*_system.json` - System information

## Project Structure

```
bench/
├── schemas/                 # Data models
│   └── conversation.py     # Conversation schemas with RAG support
├── templates/              # Chat templates
│   └── phi3.py            # Phi-3.5 chat template
├── engines/               # Inference engines
│   └── vllm_engine.py     # vLLM engine wrapper
├── metrics/               # Metrics collection
│   ├── schemas.py         # Metrics data models
│   ├── vllm_collector.py  # vLLM metrics collector
│   └── system_collector.py # System metrics collector
├── config/                # Configuration
│   └── vllm_config.py     # vLLM server configurations
├── output/                # Output formatting
│   └── formatters.py      # Output formatting (CSV/JSON/Markdown)
├── data/                  # Test data
│   ├── conversations.json # Conversation templates
│   ├── rag_data.json     # RAG simulation data
│   └── conversation_loader.py # Conversation loader
├── tests/                 # Test suite
│   ├── test_phi3_template.py
│   └── test_rag_simulation.py
├── download_models.py     # Model download automation
├── run_benchmark.py       # Basic benchmark runner
├── enhanced_benchmark_runner.py # Main benchmark runner
└── start_*_vllm.py       # vLLM server starter scripts
```

## Test Conversations

Available conversation types:

### Context-Aware Selection
- **Simple Q&A** (~800 tokens) - Basic capabilities
- **Code Discussion** (~2.5K tokens) - Technical dialogue and code generation  
- **Deep Technical** (~7K tokens) - Long-form technical discussions
- **Ultra Long Context** (~15K tokens) - Maximum context utilization

### RAG Simulation
- **RAG Context Simulation** - Tests prefix caching with context changes:
  1. Conversation starts with RAG data
  2. Multiple turns using the RAG information
  3. RAG data removed mid-conversation
  4. Conversation continues without RAG context
  5. Measures cache hit rates and performance differences

### Multi-Turn Conversations
- Each user message generates an assistant response
- Context builds through the conversation
- RAG data can be injected/removed at specific turns
- Context usage is monitored

## Available Models

List available models for download:

```bash
python download_models.py --list
```

Download specific model:

```bash
python download_models.py --model phi-3.5-mini-4bit
```

## Manual vLLM Server Management

Start vLLM server:

```bash
python start_optimized_vllm.py  # optimized settings
python start_debug_vllm.py      # debug mode, no compilation
```

## Configuration

### vLLM Configuration

Default settings for single-user scenarios:

- `max_num_batched_tokens: 512` - Reduced memory usage
- `max_num_seqs: 2` - Minimal concurrent sequences
- `enforce_eager: True` - Avoid compilation cache issues
- `enable_prefix_caching: True` - Test caching effectiveness

### System Requirements

- GPU Memory: 8GB+ for 4-bit models
- System RAM: 8GB+ 
- Disk Space: 3GB+ for models
- Python: 3.8+

### Dependencies

```bash
pip install vllm pydantic requests psutil pynvml
```

## Testing

Run individual components:

```bash
# Test chat template
python -m pytest tests/test_phi3_template.py

# Test vLLM metrics collection (requires running server)
python test_vllm_metrics.py

# Test system metrics collection
python metrics/system_collector.py

# Test output formatting
python output/formatters.py
```

## Output Formats

### CSV Output
Machine-readable metrics:
- Conversation-level aggregates
- Average performance across turns
- System resource usage

### Markdown Report
Human-readable report:
- System specifications
- Configuration used
- Performance summary
- Per-conversation results
- Turn-by-turn metrics

### JSON Output
Complete benchmark data:
- System information
- Individual turn metrics
- Configuration details
- Timestamps and metadata

## Benchmark Methodology

1. **System Info Collection** - Gather system specifications
2. **Model Loading** - Start vLLM server with optimized settings
3. **Warmup** - Run warmup generation to stabilize performance
4. **Conversation Processing** - Process each test conversation:
   - Convert to model's chat format
   - Generate responses turn by turn
   - Collect metrics per turn
5. **Results Compilation** - Aggregate metrics and save in multiple formats

## Performance Notes

### GPU Memory Usage
- Monitor with `nvidia-smi` during benchmarks
- Adjust `gpu_memory_utilization` if needed
- Use smaller quantized models for limited VRAM

### Prefix Caching
- Enabled by default to test caching effectiveness
- Measures cache hit rates across conversation turns

### Batch Size
- Optimized for single-user scenarios
- Reduces memory overhead
- Maintains long context support

## Extending

### Adding New Models
1. Add model configuration to `download_models.py`
2. Create chat template in `templates/`
3. Update benchmark runner if needed

### Adding New Metrics
1. Extend schemas in `metrics/schemas.py`
2. Update collectors to gather new metrics
3. Modify formatters to include new data

### Custom Conversations
Add conversations to `data/conversations.json` or create new data files.

## Troubleshooting

### vLLM Server Issues
- Check port conflicts (default: 8001)
- Verify model path exists
- Monitor GPU memory usage
- Use debug mode: `start_debug_vllm.py`

### Memory Issues
- Reduce `max_model_len` for smaller contexts
- Lower `gpu_memory_utilization`
- Use smaller quantized models

### Performance Issues
- Enable `enforce_eager` to avoid compilation overhead
- Reduce batch sizes further
- Check system resource usage