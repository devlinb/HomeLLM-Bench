# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

HomeLLM-Bench is a benchmark framework for testing LLM models with focus on single-user local inference scenarios. The framework provides comprehensive metrics collection, context-aware conversation selection, and RAG simulation capabilities for evaluating model performance.

## Development Commands

### Two-Step Workflow (Required)

**Step 1: Start vLLM Server**
```bash
# Phi-3 Debug mode (no compilation, optimized for single-user testing)
python start_debug_vllm.py

# Phi-3 Production mode (with optimizations)
python start_optimized_vllm.py

# Qwen Optimized mode (16K context, single-user, GPTQ)
python start_qwen_vllm.py --port 8002
```

**Step 2: Run Benchmarks**
```bash
# Run Phi-3 benchmark (default port 8001)
python enhanced_benchmark_runner.py

# Run Qwen benchmark (port 8002)
python enhanced_benchmark_runner.py --model "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4" --server-port 8002

# List available conversations for specific context size
python enhanced_benchmark_runner.py --list-conversations --context-size 128000

# Run specific test types with tags
python enhanced_benchmark_runner.py --include-tags rag,long

# Limit number of conversations
python enhanced_benchmark_runner.py --max-conversations 3
```

### Model Management

```bash
# Setup recommended model
python download_models.py --setup

# List available models
python download_models.py --list

# Download specific model
python download_models.py --model phi-3.5-mini-4bit
```

### vLLM Server Configurations

**Phi-3 Debug Mode** (`start_debug_vllm.py`):
- 8K context window (8,192 tokens)
- 50% GPU memory utilization
- No CUDA compilation (`enforce_eager=True`)
- Single concurrent sequence
- Request logging enabled
- Optimized for development and testing

**Phi-3 Production Mode** (`start_optimized_vllm.py`):
- 16K context window (configurable)
- 60% GPU memory utilization
- CUDA optimizations enabled
- Prefix caching enabled
- Request logging disabled

**Qwen Optimized Mode** (`start_qwen_vllm.py`):
- 16K context window (16,384 tokens)
- 60% GPU memory utilization
- GPTQ quantization for efficiency
- Single-user optimized (max_num_seqs=1)
- Prefix caching enabled
- ~4.95 GB VRAM usage (fits in 8GB easily)

### Testing Individual Components

```bash
# Test all templates
python -m pytest tests/ -v

# Test specific templates
python -m pytest tests/test_phi3_template.py -v
python -m pytest tests/test_qwen_template.py -v

# Validate Qwen template against official format
python test_qwen_official_format.py

# Validate optimized Qwen configuration
python validate_qwen_config.py

# Test vLLM metrics (requires running server)
python test_vllm_metrics.py

# Test system metrics
python metrics/system_collector.py

# Test output formatting
python output/formatters.py

# Test RAG simulation
python test_rag_simulation.py
```

## Architecture

### Core Components

**Main Benchmark Runner** (`enhanced_benchmark_runner.py`):
- Context-aware conversation selection based on model capabilities
- RAG simulation for testing prefix caching effectiveness
- Multi-turn conversation processing with metrics collection
- Results output in multiple formats (CSV, JSON, Markdown)

**Conversation System** (`schemas/conversation.py`):
- Pydantic models for conversation templates and messages
- Support for RAG data injection/removal mid-conversation
- Context size estimation and model compatibility checking
- Message types: normal, rag_data, rag_query, rag_removal, continuation

**Inference Engine** (`engines/vllm_engine.py`):
- Client connector to external vLLM servers
- Integrated metrics collection during generation
- Health checking and connection validation
- Support for both chat completions and text completions

**Metrics Collection** (`metrics/`):
- `vllm_collector.py`: vLLM-specific metrics (prefill/decode time, cache hits)
- `system_collector.py`: System resource metrics (GPU, CPU, memory)
- `schemas.py`: Pydantic models for all metrics data

**Chat Templates** (`templates/`):
- `phi3.py`: Phi-3.5 chat format with proper token handling
- `base.py`: Abstract base class for implementing new model templates

**Configuration** (`config/vllm_config.py`):
- VLLMServerConfig: Server parameter configuration
- VLLMConfigs: Pre-configured settings for different scenarios
- Single-user optimized settings (max_num_seqs=1, batched_tokens=model_len)

### Data Flow

1. **Server Startup**: External vLLM server started with appropriate configuration
2. **Health Check**: Benchmark runner verifies server connectivity
3. **Conversation Selection**: Based on model context size and tags
4. **Template Conversion**: Messages converted to model-specific format
5. **Turn Processing**: Each conversation turn processed with metrics collection
6. **Results Aggregation**: Metrics compiled and output in multiple formats

### Key Technical Patterns

**Context-Aware Selection**:
```python
# Conversations filtered by model capabilities
conversations = runner.select_conversations(
    include_tags=["rag", "long"],
    max_conversations=5
)
```

**RAG Simulation Flow**:
```python
# RAG data injection -> conversation -> RAG removal -> continuation
# Measures prefix cache effectiveness across context changes
```

**Metrics Collection Pattern**:
```python
# System + vLLM metrics collected per turn
system_metrics = system_collector.collect_metrics()
vllm_metrics = vllm_collector.collect_generation_metrics(response)
```

## Configuration

### vLLM Server Settings

**Single-User Optimized Settings**:
- `max_num_seqs: 1` - Single concurrent request only
- `max_num_batched_tokens: <matches max_model_len>` - Efficient for single user
- `disable_sliding_window: True` - Better for single-user scenarios
- `enforce_eager: True` - No CUDA compilation (debug mode)

**Memory Management**:
- Debug: 50% GPU utilization, 8K context → ~8GB KV cache
- Production: 60% GPU utilization, 16K context → ~10GB KV cache
- Optimized based on actual memory requirements vs. batching assumptions

**Context Window Sizing**:
- Debug mode: 8,192 tokens (sufficient for most testing)
- Production mode: 16,384 tokens (good for longer conversations)
- Long context mode: Up to 32,768 tokens (near model limits)

### Model Compatibility

**Supported Models**:
- Phi-3.5 series (primary target)
- Quantized models (4-bit, 8-bit)
- Models with 128K+ context support

**Model Requirements**:
- Chat template compatibility
- OpenAI API compatibility via vLLM
- Sufficient GPU memory (8GB+ recommended)

## Conversation Types

### Context-Aware Categories

**Simple Q&A** (~800 tokens):
- Basic model capabilities testing
- Single-turn or minimal multi-turn
- Fits in small context windows

**Code Discussion** (~2.5K tokens):
- Technical dialogue and code generation
- Multi-turn with code context
- Medium context requirements

**Deep Technical** (~7K tokens):
- Long-form technical discussions
- Complex multi-turn conversations
- Substantial context usage

**Ultra Long Context** (~15K tokens):
- Maximum context utilization
- Stress testing for large context models
- Memory and performance intensive

### RAG Simulation

**Purpose**: Test prefix caching effectiveness with dynamic context changes

**Flow**:
1. Initial conversation with RAG data
2. Multiple turns using RAG information
3. RAG data removal mid-conversation
4. Continuation without RAG context
5. Performance comparison (cache hit rates)

## Testing Strategy

### Unit Tests
- Chat template formatting (`test_phi3_template.py`)
- Schema validation and conversion
- Metrics collection accuracy

### Integration Tests
- vLLM server integration (`test_vllm_metrics.py`)
- End-to-end conversation processing
- Multi-format output generation

### Performance Tests
- Memory usage monitoring
- Generation speed benchmarks
- Context length scaling tests

## Output Formats

### CSV Output
- Machine-readable metrics aggregation
- Conversation-level performance summaries
- System resource utilization data

### JSON Output
- Complete benchmark data with full detail
- Turn-by-turn metrics preservation
- Metadata and configuration information

### Markdown Reports
- Human-readable performance summaries
- System specifications and configuration
- Per-conversation detailed results

## System Requirements

**Hardware**:
- GPU Memory: 8GB+ (4-bit quantized models)
- System RAM: 8GB+
- Disk Space: 3GB+ for models

**Software**:
- Python 3.8+
- vLLM 0.6.0+
- CUDA-compatible GPU drivers

**Dependencies**:
```bash
pip install vllm pydantic requests psutil pynvml
```

## Troubleshooting

### vLLM Server Issues
- Check port conflicts (default: 8001)
- Verify model file exists and is accessible
- Monitor GPU memory usage during startup
- Use `start_debug_vllm.py` for compilation-free debugging

### Memory Issues
- Reduce `gpu_memory_utilization` if OOM errors occur
- Lower `max_model_len` for constrained memory
- Use smaller quantized models
- Monitor system memory usage during long conversations

### Performance Issues
- **Debug mode**: Use `start_debug_vllm.py` to avoid compilation overhead
- **Memory issues**: Reduce context size in server config
- **Slow startup**: Normal for first-time model loading
- **Connection failures**: Verify server is fully started before running benchmarks

### Server Management
- **Check server status**: `curl http://127.0.0.1:8001/health`
- **Stop server**: `pkill -f vllm` or Ctrl+C in server terminal
- **Port conflicts**: Change port in server config if needed
- **Model loading**: Wait for "Starting vLLM API server" message before running benchmarks

## Workflow Examples

### Quick Testing (Debug Mode)
```bash
# Terminal 1: Start debug server
python start_debug_vllm.py

# Terminal 2: Run quick test
python enhanced_benchmark_runner.py --max-conversations 1
```

### Production Benchmarking
```bash
# Terminal 1: Start production server
python start_optimized_vllm.py

# Terminal 2: Run full benchmark suite
python enhanced_benchmark_runner.py
```

### Testing Specific Scenarios
```bash
# Start appropriate server first, then:
python enhanced_benchmark_runner.py --include-tags rag,long --max-conversations 5
```

## Adding New Models

1. **Download Model**: Add to `download_models.py`
2. **Start Server**: Use existing server scripts with new model path
3. **Chat Template**: Create new template in `templates/` if needed
4. **Test**: Run benchmarks with `--model <new_model_path>`
5. **Documentation**: Update supported models list