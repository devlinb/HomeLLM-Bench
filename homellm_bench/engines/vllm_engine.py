import os
from typing import Optional, Dict, Any, List

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..metrics.vllm_collector import VLLMMetricsCollector
from ..metrics.schemas import GenerationMetrics


class VLLMEngine:
    """Client for connecting to external vLLM server and collecting metrics"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
                 host: str = "127.0.0.1",
                 port: int = 8001):
        
        self.model_name = model_name
        self.host = host
        self.port = port
        
        self.metrics_collector = VLLMMetricsCollector(host, port)
        
    def is_running(self) -> bool:
        """Check if external server is running and healthy"""
        return self.metrics_collector.check_server_health()
    
    def generate_with_metrics(self, 
                            prompt: str,
                            max_tokens: int = 100,
                            temperature: float = 0.7,
                            **kwargs) -> tuple[str, GenerationMetrics]:
        """Generate text and collect metrics"""
        if not self.is_running():
            raise RuntimeError(f"vLLM server is not running on {self.host}:{self.port}")
        
        return self.metrics_collector.generate_with_metrics(
            prompt=prompt,
            model_name=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def generate_chat_with_metrics(self,
                                  messages: List[Dict[str, str]],
                                  max_tokens: int = 100,
                                  temperature: float = 0.7,
                                  **kwargs) -> tuple[str, GenerationMetrics]:
        """Generate chat completion with full metrics collection"""
        if not self.is_running():
            raise RuntimeError(f"vLLM server is not running on {self.host}:{self.port}")
        
        return self.metrics_collector.generate_chat_with_metrics(
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
    # Test the vLLM engine
    print("Testing vLLM Engine")
    print("=" * 30)
    
    # Test with external server (assumes debug server is running)
    engine = VLLMEngine(
        model_name="./phi-3.5-mini-Q4_K.gguf",
        host="127.0.0.1",
        port=8001
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
        print("Start the debug server first:")
        print("  python start_debug_vllm.py")
    
    print("\nEngine ready!")