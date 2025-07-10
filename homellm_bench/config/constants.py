"""
Constants for HomeLLM-Bench - keeping it simple
"""

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
DEFAULT_HOST = "127.0.0.1"
SERVER_HEALTH_TIMEOUT = 5
REQUEST_TIMEOUT = 60
PORT_CHECK_TIMEOUT = 1

# =============================================================================
# SERVER LIFECYCLE MANAGEMENT
# =============================================================================
SERVER_STARTUP_TIMEOUT = 120          # 2 minutes for server startup
SERVER_SHUTDOWN_TIMEOUT = 30          # 30 seconds for graceful shutdown

# =============================================================================
# GPU MEMORY MANAGEMENT
# =============================================================================
# Single GPU utilization for all scenarios
GPU_MEMORY_UTILIZATION = 0.75        # 75% - safe for most GPUs
MEMORY_SAFETY_MARGIN = 0.80          # Use 80% of target to stay under limits
OVERHEAD_BUFFER_GB = 0.5             # 0.5GB for system overhead

# =============================================================================
# MODEL CONFIGURATION  
# =============================================================================
# Standard context size for home LLM use (fits our longest test case)
STANDARD_CONTEXT_SIZE = 16384         # 16K - handles "Ultra Long Context" test

# Model architecture defaults (for calculations)
DEFAULT_HIDDEN_SIZE = 2048
DEFAULT_NUM_LAYERS = 24

# =============================================================================
# GENERATION PARAMETERS
# =============================================================================
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500              # Reasonable response length
SINGLE_SEQUENCE = 1                   # One response at a time for home use

# =============================================================================
# BENCHMARK SETTINGS
# =============================================================================
CONTEXT_WARNING_THRESHOLD = 0.8       # Warn at 80% of context limit
TOKEN_ESTIMATION_DIVISOR = 4          # Rough tokens per word estimate

# =============================================================================
# CONVERSION FACTORS
# =============================================================================
MB_TO_GB = 1024
BYTES_TO_GB = 1024**3
BYTES_TO_MB = 1024**2