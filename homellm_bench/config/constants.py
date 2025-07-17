"""
Constants for HomeLLM-Bench - keeping it simple
"""

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
VLLM_DEFAULT_PORT = 8000
OLLAMA_DEFAULT_PORT = 11434
REQUEST_TIMEOUT = 60

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500
DEFAULT_CONTEXT_SIZE = 32000          # Default context size for benchmarks

# =============================================================================
# BENCHMARK SETTINGS
# =============================================================================
CONTEXT_WARNING_THRESHOLD = 0.8       # Warn at 80% of context limit
TOKEN_ESTIMATION_DIVISOR = 4          # Rough characters per token estimate