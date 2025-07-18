import requests
import time
import json
import psutil
import re
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False

from .schemas import GenerationMetrics, VLLMMetrics


class VLLMMetricsCollector:
    """Collects metrics from vLLM server during inference"""
    
    def __init__(self, server_host: str = "127.0.0.1", server_port: int = 8001):
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
        """Check if vLLM server is responding"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_server_metrics(self) -> Dict[str, Any]:
        """Get metrics from vLLM server's metrics endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200:
                return self._parse_prometheus_metrics(response.text)
        except Exception as e:
            print(f"Warning: Could not get server metrics: {e}")
        return {}
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, Any]:
        """Parse Prometheus-format metrics from vLLM"""
        metrics = {}
        
        # Common vLLM metrics patterns
        patterns = {
            'running_requests': r'vllm:num_requests_running\s+(\d+\.?\d*)',
            'waiting_requests': r'vllm:num_requests_waiting\s+(\d+\.?\d*)', 
            'gpu_cache_usage': r'vllm:gpu_cache_usage_perc\s+(\d+\.?\d*)',
            'prefill_tokens_total': r'vllm:prefill_tokens_total\s+(\d+\.?\d*)',
            'generation_tokens_total': r'vllm:generation_tokens_total\s+(\d+\.?\d*)',
            'time_to_first_token': r'vllm:time_to_first_token_seconds\s+(\d+\.?\d*)',
            'time_per_output_token': r'vllm:time_per_output_token_seconds\s+(\d+\.?\d*)',
        }
        
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, metrics_text)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics
    
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
    
    def generate_with_metrics(self, 
                            prompt: str, 
                            model_name: str = "phi-3.5-mini",
                            max_tokens: int = 100,
                            temperature: float = 0.7,
                            **kwargs) -> tuple[str, GenerationMetrics]:
        """Generate text and collect comprehensive metrics"""
        
        # Pre-generation metrics
        start_time = time.time()
        start_system_metrics = self.get_system_metrics()
        start_server_metrics = self.get_server_metrics()
        
        # Prepare request
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,  # For simplicity, use non-streaming
            **kwargs
        }
        
        # Make generation request
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
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
        end_server_metrics = self.get_server_metrics()
        
        # Extract response data
        generated_text = result['choices'][0]['text']
        usage = result.get('usage', {})
        
        # Calculate timing metrics
        total_time = end_time - start_time
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Estimate time to first token (rough approximation)
        ttft = total_time * 0.1  # Assume 10% of time for first token
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
        
        # Cache hit rate calculation removed - now using timing delta instead
        
        # Create metrics object
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_to_first_token=ttft,
            total_generation_time=total_time,
            tokens_per_second=tokens_per_second,
            engine_metrics={
                'start_server_metrics': start_server_metrics,
                'end_server_metrics': end_server_metrics,
                'response_usage': usage
            },
            engine_name="vllm",
            model_name=model_name
        )
        
        return generated_text, metrics
    
    def generate_chat_with_metrics(self, 
                                  model_name: str,
                                  messages: List[Dict[str, str]],
                                  max_tokens: int = 100,
                                  temperature: float = 0.7,
                                  **kwargs) -> GenerationMetrics:
        """Generate chat completion with metrics collection"""
        
        # Pre-generation metrics
        start_time = time.time()
        start_system_metrics = self.get_system_metrics()
        start_server_metrics = self.get_server_metrics()
        
        # Prepare chat completion request
        request_data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **kwargs
        }
        
        # Make generation request
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
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
        end_server_metrics = self.get_server_metrics()
        
        # Extract response data
        generated_text = result['choices'][0]['message']['content']
        usage = result.get('usage', {})
        
        # Calculate timing metrics
        total_time = end_time - start_time
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        
        # Estimate time to first token (rough approximation)
        ttft = total_time * 0.1  # Assume 10% of time for first token
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
        
        # Cache hit rate calculation removed - now using timing delta instead
        
        # Build metrics object
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_to_first_token=ttft,
            total_generation_time=total_time,
            tokens_per_second=tokens_per_second,
            engine_name="vllm",
            model_name=model_name,
            engine_metrics={
                'start_server_metrics': start_server_metrics,
                'end_server_metrics': end_server_metrics,
                'response_metadata': result
            }
        )
        
        return generated_text, metrics
    
    def generate_chat_streaming(self, 
                               model_name: str,
                               messages: List[Dict[str, str]],
                               max_tokens: int = 100,
                               temperature: float = 0.7,
                               **kwargs) -> Generator[str, None, GenerationMetrics]:
        """Generate chat completion with streaming tokens and final metrics"""
        
        # Pre-generation metrics
        start_time = time.time()
        start_system_metrics = self.get_system_metrics()
        start_server_metrics = self.get_server_metrics()
        
        # Prepare streaming chat completion request
        request_data = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs
        }
        
        # Make streaming generation request
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
            # Stream processing
            full_text = ""
            completion_tokens = 0
            
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    
                    # Skip empty lines and non-data lines
                    if not line_text.startswith('data: '):
                        continue
                    
                    # Extract JSON data
                    json_str = line_text[6:]  # Remove 'data: ' prefix
                    
                    # Check for stream end
                    if json_str.strip() == '[DONE]':
                        break
                    
                    try:
                        chunk = json.loads(json_str)
                        delta = chunk.get('choices', [{}])[0].get('delta', {})
                        
                        # Extract token content
                        if 'content' in delta:
                            token = delta['content']
                            full_text += token
                            completion_tokens += 1
                            yield token
                            
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue
                        
        except Exception as e:
            raise RuntimeError(f"Streaming generation failed: {e}")
        
        # Post-generation metrics
        end_time = time.time()
        end_system_metrics = self.get_system_metrics()
        end_server_metrics = self.get_server_metrics()
        
        # Calculate timing metrics
        total_time = end_time - start_time
        # Note: streaming doesn't provide usage stats, so we estimate
        prompt_tokens = sum(len(msg['content'].split()) for msg in messages)  # Rough estimate
        total_tokens = prompt_tokens + completion_tokens
        
        # Create GenerationMetrics
        metrics = GenerationMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            time_to_first_token=0.1,  # Estimated - streaming doesn't provide this precisely
            total_generation_time=total_time,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            engine_name="vllm",
            model_name=model_name,
            engine_metrics={
                'start_server_metrics': start_server_metrics,
                'end_server_metrics': end_server_metrics,
                'streaming': True,
                'estimated_tokens': True
            }
        )
        
        return metrics
    
    def warmup_model(self, model_name: str = "phi-3.5-mini") -> bool:
        """Warmup the model with a simple request"""
        try:
            _, _ = self.generate_with_metrics(
                prompt="Hello",
                model_name=model_name,
                max_tokens=5,
                temperature=0.0
            )
            return True
        except Exception as e:
            print(f"Warmup failed: {e}")
            return False


if __name__ == "__main__":
    # Test the metrics collector - only for standalone testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from metrics.schemas import GenerationMetrics, VLLMMetrics
    
    collector = VLLMMetricsCollector()
    
    print("Testing vLLM Metrics Collector")
    print("=" * 40)
    
    # Check server health
    print(f"Server health: {collector.check_server_health()}")
    
    # Get system metrics
    system_metrics = collector.get_system_metrics()
    print(f"System metrics: {system_metrics}")
    
    # Get server metrics
    server_metrics = collector.get_server_metrics()
    print(f"Server metrics: {server_metrics}")
    
    print("\nMetrics collector ready!")