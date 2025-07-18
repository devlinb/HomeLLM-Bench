import os
import sys
from typing import Optional, Dict, Any, List, Generator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..metrics.ollama_collector import OllamaMetricsCollector
from ..metrics.schemas import GenerationMetrics


class OllamaEngine:
    """Client for connecting to external Ollama server and collecting metrics"""
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b",
                 host: str = "127.0.0.1",
                 port: int = 11434):
        
        self.model_name = model_name
        self.host = host
        self.port = port
        
        self.metrics_collector = OllamaMetricsCollector(host, port)
        
    def is_running(self) -> bool:
        """Check if external Ollama server is running and healthy"""
        return self.metrics_collector.check_server_health()
    
    def generate_chat_with_metrics(self,
                                  messages: List[Dict[str, str]],
                                  max_tokens: int = 100,
                                  temperature: float = 0.7,
                                  **kwargs) -> tuple[str, GenerationMetrics]:
        """Generate chat completion with full metrics collection"""
        if not self.is_running():
            raise RuntimeError(f"Ollama server is not running on {self.host}:{self.port}")
        
        return self.metrics_collector.generate_chat_with_metrics(
            model_name=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def generate_chat_streaming(self,
                               messages: List[Dict[str, str]],
                               max_tokens: int = 100,
                               temperature: float = 0.7,
                               **kwargs) -> Generator[str, None, GenerationMetrics]:
        """Generate chat completion with streaming tokens and final metrics"""
        if not self.is_running():
            raise RuntimeError(f"Ollama server is not running on {self.host}:{self.port}")
        
        return self.metrics_collector.generate_chat_streaming(
            model_name=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def warmup(self) -> bool:
        """Warmup the model"""
        if not self.is_running():
            return False
        return self.metrics_collector.warmup_model(self.model_name)


if __name__ == "__main__":
    # Test the Ollama engine
    print("Testing Ollama Engine")
    print("=" * 30)
    
    # Test with external server (assumes Ollama server is running)
    engine = OllamaEngine(
        model_name="llama3.2:3b",
        host="127.0.0.1",
        port=11434
    )
    
    print(f"Engine created for model: {engine.model_name}")
    print(f"Server: {engine.host}:{engine.port}")
    
    if engine.is_running():
        print("Success: Server is healthy")
        
        # Test generation
        try:
            response, metrics = engine.generate_chat_with_metrics(
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=10
            )
            print(f"Success: Test response: {response}")
            print(f"Stats: Tokens/sec: {metrics.tokens_per_second:.2f}")
        except Exception as e:
            print(f"Error: Test failed: {e}")
    else:
        print("Error: Server is not responding")
        print("Start the Ollama server first:")
        print("  ollama serve")
    
    print("\nEngine ready!")