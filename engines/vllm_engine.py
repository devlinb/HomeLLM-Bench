import subprocess
import time
import signal
import psutil
import os
from typing import Optional, Dict, Any, List
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.vllm_collector import VLLMMetricsCollector
from metrics.schemas import GenerationMetrics
from config.vllm_config import VLLMServerConfig, VLLMConfigs


class VLLMEngine:
    """Manages vLLM server process and provides metrics collection"""
    
    def __init__(self, 
                 model_path: str,
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 max_model_len: Optional[int] = None,
                 enable_prefix_caching: bool = False,
                 gpu_memory_utilization: float = 0.8,
                 **kwargs):
        
        self.model_path = model_path
        self.host = host
        self.port = port
        self.max_model_len = max_model_len
        self.enable_prefix_caching = enable_prefix_caching
        self.gpu_memory_utilization = gpu_memory_utilization
        self.extra_args = kwargs
        
        self.process: Optional[subprocess.Popen] = None
        self.metrics_collector = VLLMMetricsCollector(host, port)
        
    def start_server(self, timeout: int = 60) -> bool:
        """Start vLLM server as subprocess"""
        if self.process and self.process.poll() is None:
            print("Server already running")
            return True
        
        # Build command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        
        # Add optional arguments
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        
        if self.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        
        # Add any extra arguments
        for key, value in self.extra_args.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key.replace('_', '-')}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        print(f"Starting vLLM server with command:")
        print(" ".join(cmd))
        
        # Start process
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group
            )
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
        
        # Wait for server to be ready
        print("Waiting for server to start...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.metrics_collector.check_server_health():
                print(f"Server ready on {self.host}:{self.port}")
                return True
            
            # Check if process died
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print(f"Server process died. STDERR: {stderr}")
                return False
            
            time.sleep(2)
        
        print(f"Server failed to start within {timeout} seconds")
        self.stop_server()
        return False
    
    def stop_server(self) -> None:
        """Stop vLLM server and clean up"""
        if self.process:
            try:
                # Try graceful shutdown first
                print("Stopping vLLM server...")
                
                # Kill the entire process group to get all child processes
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    print("Force killing server...")
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
                
                print("Server stopped")
                
            except Exception as e:
                print(f"Error stopping server: {e}")
            finally:
                self.process = None
    
    def restart_server(self) -> bool:
        """Restart the server (useful for cache clearing)"""
        self.stop_server()
        time.sleep(2)  # Allow cleanup
        return self.start_server()
    
    def generate_with_metrics(self, 
                            prompt: str,
                            max_tokens: int = 100,
                            temperature: float = 0.7,
                            **kwargs) -> tuple[str, GenerationMetrics]:
        """Generate text and collect metrics"""
        if not self.is_running():
            raise RuntimeError("Server is not running")
        
        return self.metrics_collector.generate_with_metrics(
            prompt=prompt,
            model_name=os.path.basename(self.model_path),
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def is_running(self) -> bool:
        """Check if server is running and healthy"""
        return (self.process is not None and 
                self.process.poll() is None and 
                self.metrics_collector.check_server_health())
    
    def get_server_logs(self) -> tuple[str, str]:
        """Get server stdout and stderr logs"""
        if self.process:
            try:
                # Non-blocking read
                stdout = self.process.stdout.read() if self.process.stdout else ""
                stderr = self.process.stderr.read() if self.process.stderr else ""
                return stdout, stderr
            except:
                return "", ""
        return "", ""
    
    def warmup(self) -> bool:
        """Warmup the model"""
        if not self.is_running():
            return False
        return self.metrics_collector.warmup_model(os.path.basename(self.model_path))
    
    def __enter__(self):
        """Context manager entry"""
        if not self.start_server():
            raise RuntimeError("Failed to start vLLM server")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_server()


if __name__ == "__main__":
    # Test the vLLM engine
    print("Testing vLLM Engine")
    print("=" * 30)
    
    # Test with the downloaded model
    model_path = "./phi-3.5-mini-Q4_K.gguf"
    
    engine = VLLMEngine(
        model_path=model_path,
        port=8001,  # Use different port to avoid conflicts
        max_model_len=1024,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        quantization="gguf"
    )
    
    print(f"Engine created for model: {model_path}")
    print(f"Server will run on: {engine.host}:{engine.port}")
    print(f"Prefix caching: {engine.enable_prefix_caching}")
    
    print("\nEngine ready for testing!")