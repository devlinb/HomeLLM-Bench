#!/usr/bin/env python3
"""
vLLM server startup and monitoring utilities.
"""

import subprocess
import time
import requests
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ServerStartupResult:
    """Result of server startup attempt."""
    success: bool
    process: Optional[subprocess.Popen] = None
    error_message: Optional[str] = None
    startup_time: float = 0.0


class VLLMServerManager:
    """Manages vLLM server startup, monitoring, and shutdown."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8001):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process: Optional[subprocess.Popen] = None
        self.startup_timeout = 120  # 2 minutes
        self.health_check_interval = 5  # 5 seconds
        self.process_error: Optional[str] = None
        self.monitor_thread: Optional[threading.Thread] = None
        self.should_monitor = False
    
    def start_server(self, model_path: str, **kwargs) -> ServerStartupResult:
        """Start vLLM server with the given configuration.
        
        Args:
            model_path: Path to the model
            **kwargs: Additional server configuration options
            
        Returns:
            ServerStartupResult with success status and details
        """
        start_time = time.time()
        
        # Build command
        cmd = self._build_server_command(model_path, **kwargs)
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start monitoring thread
            self.should_monitor = True
            self.process_error = None
            self.monitor_thread = threading.Thread(
                target=self._monitor_process,
                daemon=True
            )
            self.monitor_thread.start()
            
            # Wait for server to be ready
            ready_result = self._wait_for_ready()
            
            if ready_result.success:
                startup_time = time.time() - start_time
                return ServerStartupResult(
                    success=True,
                    process=self.process,
                    startup_time=startup_time
                )
            else:
                # Clean up failed process
                self.shutdown()
                return ready_result
                
        except Exception as e:
            return ServerStartupResult(
                success=False,
                error_message=f"Failed to start server process: {e}"
            )
    
    def _build_server_command(self, model_path: str, **kwargs) -> List[str]:
        """Build the vLLM server command with given configuration."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--trust-remote-code",
        ]
        
        # Add optional parameters
        if kwargs.get("gpu_memory_utilization"):
            cmd.extend(["--gpu-memory-utilization", str(kwargs["gpu_memory_utilization"])])
        
        if kwargs.get("max_model_len"):
            cmd.extend(["--max-model-len", str(kwargs["max_model_len"])])
        
        if kwargs.get("enable_prefix_caching"):
            cmd.append("--enable-prefix-caching")
        
        if kwargs.get("enforce_eager"):
            cmd.append("--enforce-eager")
        
        if kwargs.get("disable_log_stats"):
            cmd.append("--disable-log-stats")
        
        if kwargs.get("disable_log_requests"):
            cmd.append("--disable-log-requests")
        
        # Add any additional arguments
        extra_args = kwargs.get("extra_args", [])
        if extra_args:
            cmd.extend(extra_args)
        
        return cmd
    
    def _monitor_process(self):
        """Monitor the server process for errors."""
        if not self.process:
            return
            
        try:
            while self.should_monitor and self.process.poll() is None:
                time.sleep(0.5)
            
            # Process exited
            if self.should_monitor and self.process.poll() is not None:
                return_code = self.process.poll()
                if return_code != 0:
                    # Get error output
                    try:
                        _, stderr = self.process.communicate(timeout=5)
                        self.process_error = f"Server process exited with code {return_code}: {stderr}"
                    except:
                        self.process_error = f"Server process exited with code {return_code}"
                        
        except Exception as e:
            self.process_error = f"Error monitoring process: {e}"
    
    def _wait_for_ready(self) -> ServerStartupResult:
        """Wait for server to be ready by polling health endpoint."""
        start_time = time.time()
        
        while time.time() - start_time < self.startup_timeout:
            # Check if process crashed
            if self.process_error:
                return ServerStartupResult(
                    success=False,
                    error_message=self.process_error
                )
            
            # Check if process exited
            if self.process and self.process.poll() is not None:
                return ServerStartupResult(
                    success=False,
                    error_message=f"Server process exited unexpectedly with code {self.process.poll()}"
                )
            
            # Check health endpoint
            try:
                response = requests.get(
                    f"{self.base_url}/health",
                    timeout=2
                )
                if response.status_code == 200:
                    return ServerStartupResult(success=True)
                    
            except requests.exceptions.RequestException:
                # Expected during startup
                pass
            
            time.sleep(self.health_check_interval)
        
        # Timeout reached
        return ServerStartupResult(
            success=False,
            error_message=f"Server failed to become ready within {self.startup_timeout} seconds"
        )
    
    def is_healthy(self) -> bool:
        """Check if server is currently healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def shutdown(self):
        """Shutdown the server and clean up resources."""
        self.should_monitor = False
        
        if self.process:
            try:
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if necessary
                    self.process.kill()
                    self.process.wait()
                    
            except:
                pass
            
            self.process = None
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            self.monitor_thread = None
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get information about the running server."""
        info = {
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "process_id": self.process.pid if self.process else None,
            "is_healthy": self.is_healthy(),
        }
        
        # Try to get model info
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                if models.get("data"):
                    info["model"] = models["data"][0].get("id")
        except:
            pass
        
        return info
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.shutdown()


def start_vllm_server(model_path: str, 
                     host: str = "127.0.0.1", 
                     port: int = 8001,
                     **kwargs) -> VLLMServerManager:
    """Start a vLLM server with the given configuration.
    
    Args:
        model_path: Path to the model
        host: Server host
        port: Server port
        **kwargs: Additional server configuration
        
    Returns:
        VLLMServerManager instance
        
    Raises:
        RuntimeError: If server fails to start
    """
    manager = VLLMServerManager(host, port)
    result = manager.start_server(model_path, **kwargs)
    
    if not result.success:
        manager.shutdown()
        raise RuntimeError(f"Failed to start vLLM server: {result.error_message}")
    
    return manager