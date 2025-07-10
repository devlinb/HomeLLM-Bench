"""
HomeLLM-Bench: A benchmark suite for small quantized LLMs at home
"""

__version__ = "0.1.0"
__author__ = "HomeLLM-Bench Contributors"
__description__ = "Benchmark suite for evaluating small quantized LLMs in home environments"

# Core components
# Remove imports that cause circular issues
# from .benchmark.runner import BenchmarkRunner
# from .server.manager import ServerManager
from .config.constants import *

__all__ = [
    "__version__",
    "__author__",
    "__description__",
]