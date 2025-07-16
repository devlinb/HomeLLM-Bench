# HomeLLM-Bench

A benchmark framework for testing LLM models with focus on single-user local inference scenarios. The framework provides comprehensive metrics collection, context-aware conversation selection, and RAG simulation capabilities.

## Quick Start

### 1. Start Your LLM Server

```bash
# Start vLLM server (default port 8000)
vllm serve Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --host 127.0.0.1 --port 8000

# Or any other OpenAI-compatible server
```

### 2. Run Benchmark

```bash
# Basic benchmark run
python -m homellm_bench.cli.benchmark

# With custom configuration
python -m homellm_bench.cli.benchmark --port 8000 --context-size 16000 --max-conversations 3
```

## Usage

### Basic Commands

```bash
# Run benchmark (connects to localhost:8000 by default)
python -m homellm_bench.cli.benchmark

# Specify server details
python -m homellm_bench.cli.benchmark --host 127.0.0.1 --port 8000

# Configure context size and engine
python -m homellm_bench.cli.benchmark --context-size 16000 --engine vllm

# Filter conversations
python -m homellm_bench.cli.benchmark --include-tags rag,coding --max-conversations 5

# List available conversations
python -m homellm_bench.cli.benchmark --list-conversations
```

### Command Line Options

- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)
- `--engine`: Engine type for metrics (vllm, ollama)
- `--context-size`: Context size in tokens (default: 32000)
- `--max-conversations`: Maximum number of conversations to run
- `--include-tags`: Include only conversations with these tags
- `--exclude-tags`: Exclude conversations with these tags
- `--list-conversations`: List available conversations and exit

### Supported Engines

- **vLLM**: Full support with detailed metrics
- **Ollama**: Planned support

### Conversation Types

- **Simple Q&A**: Basic capabilities testing
- **Code Discussion**: Technical dialogue and code generation
- **Deep Technical**: Long-form technical discussions
- **RAG Simulation**: Tests with dynamic context changes
- **Ultra Long Context**: Maximum context utilization tests

### RAG Simulation

The benchmark includes RAG (Retrieval-Augmented Generation) simulation that:
1. Loads initial context data
2. Runs conversations using the data
3. Removes data mid-conversation
4. Continues without RAG context
5. Measures prefix caching effectiveness

## Output

Results are saved in multiple formats:
- **CSV**: Machine-readable metrics
- **JSON**: Complete benchmark data
- **Markdown**: Human-readable reports

## Server Configuration

### vLLM Recommended Settings

```bash
vllm serve <model> \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 1 \
  --enable-prefix-caching \
  --disable-log-requests
```

### Key Parameters

- `--max-model-len`: Match your benchmark context size
- `--gpu-memory-utilization`: Adjust for available GPU memory
- `--max-num-seqs 1`: Single sequence for consistent benchmarking
- `--enable-prefix-caching`: Better multi-turn performance

## Installation

```bash
pip install -r requirements.txt
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_conversation_processing.py -v
```


