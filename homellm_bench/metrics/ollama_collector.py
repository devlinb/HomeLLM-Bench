import requests
import time
import json
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False

from .schemas import GenerationMetrics


class OllamaMetricsCollector:
    """Collects metrics from Ollama server during inference"""
    
    def __init__(self, server_host: str = "127.0.0.1", server_port: int = 11434):
        self.server_host = server_host
        self.server_port = server_port
        self.base_url = f"http://{server_host}:{server_port}"
        self.session = requests.Session()
        
        # Initialize GPU monitoring if available
        self.gpu_available = PYNVML_AVAILABLE
        if self.gpu_available:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                self.gpu_available = False
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def check_server_health(self) -> bool:
        """Check if Ollama server is responding"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics (CPU, memory, GPU)"""
        metrics = {}
        
        # CPU and memory
        metrics['cpu_percent'] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics['memory_used_mb'] = memory.used / (1024 * 1024)
        metrics['memory_percent'] = memory.percent
        
        # GPU metrics
        if self.gpu_available:
            gpu_metrics = []
            for i in range(self.gpu_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    gpu_metrics.append({
                        'gpu_id': i,
                        'memory_used_mb': mem_info.used / (1024 * 1024),
                        'memory_total_mb': mem_info.total / (1024 * 1024),
                        'memory_percent': (mem_info.used / mem_info.total) * 100,
                        'utilization_percent': utilization.gpu
                    })
                except Exception as e:
                    print(f"Warning: Could not get GPU {i} metrics: {e}")
            
            metrics['gpus'] = gpu_metrics
            
            # Summary metrics
            if gpu_metrics:
                metrics['total_gpu_memory_used_mb'] = sum(gpu['memory_used_mb'] for gpu in gpu_metrics)
                metrics['avg_gpu_utilization'] = sum(gpu['utilization_percent'] for gpu in gpu_metrics) / len(gpu_metrics)
        
        return metrics
    
    def _convert_nanoseconds_to_seconds(self, nanoseconds: int) -> float:
        """Convert nanoseconds to seconds"""
        return nanoseconds / 1_000_000_000.0
    
    def generate_chat_with_metrics(self, 
                                  model_name: str,
                                  messages: List[Dict[str, str]],
                                  max_tokens: int = 100,
                                  temperature: float = 0.7,
                                  **kwargs) -> tuple[str, GenerationMetrics]:
        """Generate chat completion with metrics collection"""
        
        # Pre-generation metrics
        start_time = time.time()
        start_system_metrics = self.get_system_metrics()
        
        # Prepare Ollama chat request
        request_data = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        # Make generation request
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")
        
        # Post-generation metrics
        end_time = time.time()
        end_system_metrics = self.get_system_metrics()
        
        # Extract response data
        generated_text = result.get('message', {}).get('content', '')
        
        # Extract Ollama-specific metrics (all in nanoseconds)
        total_duration = result.get('total_duration', 0)
        load_duration = result.get('load_duration', 0)
        prompt_eval_duration = result.get('prompt_eval_duration', 0)
        eval_duration = result.get('eval_duration', 0)
        
        # Token counts
        prompt_tokens = result.get('prompt_eval_count', 0)
        completion_tokens = result.get('eval_count', 0)
        total_tokens = prompt_tokens + completion_tokens
        
        # Convert durations to seconds
        total_time_seconds = self._convert_nanoseconds_to_seconds(total_duration)
        load_time_seconds = self._convert_nanoseconds_to_seconds(load_duration)
        prompt_eval_time_seconds = self._convert_nanoseconds_to_seconds(prompt_eval_duration)
        eval_time_seconds = self._convert_nanoseconds_to_seconds(eval_duration)
        
        # Calculate performance metrics
        # Time to first token is approximately the prompt evaluation time
        ttft = prompt_eval_time_seconds
        
        # Tokens per second calculation
        tokens_per_second = completion_tokens / eval_time_seconds if eval_time_seconds > 0 else 0
        
        # Create metrics object
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_to_first_token=ttft,
            total_generation_time=total_time_seconds,
            tokens_per_second=tokens_per_second,
            engine_name="ollama",
            model_name=model_name,
            engine_metrics={
                'ollama_native_metrics': {
                    'total_duration_ns': total_duration,
                    'load_duration_ns': load_duration,
                    'prompt_eval_duration_ns': prompt_eval_duration,
                    'eval_duration_ns': eval_duration,
                    'load_time_seconds': load_time_seconds,
                    'prompt_eval_time_seconds': prompt_eval_time_seconds,
                    'eval_time_seconds': eval_time_seconds
                },
                'start_system_metrics': start_system_metrics,
                'end_system_metrics': end_system_metrics,
                'response_metadata': result
            }
        )
        
        return generated_text, metrics
    
    def warmup_model(self, model_name: str) -> bool:
        """Warmup the model with a simple request"""
        try:
            _, _ = self.generate_chat_with_metrics(
                model_name=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.0
            )
            return True
        except Exception as e:
            print(f"Warmup failed: {e}")
            return False


if __name__ == "__main__":
    # Test the metrics collector
    collector = OllamaMetricsCollector()
    
    print("Testing Ollama Metrics Collector")
    print("=" * 40)
    
    # Check server health
    print(f"Server health: {collector.check_server_health()}")
    
    # Get system metrics
    system_metrics = collector.get_system_metrics()
    print(f"System metrics: {system_metrics}")
    
    print("\nOllama metrics collector ready!")