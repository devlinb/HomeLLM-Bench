"""
Simple factory pattern for benchmark dependencies
"""
from typing import Optional
from ..metrics.vllm_collector import VLLMMetricsCollector
from ..metrics.system_collector import SystemMetricsCollector
from ..engines.vllm_engine import VLLMEngine
from ..output.formatters import BenchmarkFormatter
from ..data.conversation_loader import ConversationLoader


class BenchmarkDependencies:
    """Simple factory for benchmark dependencies"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self._system_collector = None
        self._formatter = None
        self._conversation_loader = None
    
    @property
    def system_collector(self) -> SystemMetricsCollector:
        """Get system metrics collector"""
        if self._system_collector is None:
            self._system_collector = SystemMetricsCollector()
        return self._system_collector
    
    @property
    def formatter(self) -> BenchmarkFormatter:
        """Get benchmark formatter"""
        if self._formatter is None:
            self._formatter = BenchmarkFormatter(self.output_dir)
        return self._formatter
    
    @property
    def conversation_loader(self) -> ConversationLoader:
        """Get conversation loader"""
        if self._conversation_loader is None:
            self._conversation_loader = ConversationLoader()
        return self._conversation_loader
    
    
