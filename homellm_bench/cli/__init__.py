"""
CLI module for HomeLLM-Bench
"""

from .benchmark import main as benchmark_main
from .server import main as server_main

__all__ = ["benchmark_main", "server_main"]