#!/usr/bin/env python3
"""
Generic vLLM server startup script
Supports any model with configuration-driven parameters
"""
import subprocess
import sys
import argparse
from typing import Optional

from ..config.vllm_config import VLLMServerConfig, DEFAULT_VLLM_PORT
from ..config.model_config import model_registry
from ..config.constants import STANDARD_CONTEXT_SIZE, SINGLE_SEQUENCE, DEFAULT_HOST
from ..utils.system_info import initialize_system, get_system_info, GPURuntime
from ..utils.gpu_memory import get_gpu_memory_manager
from ..utils.server_lifecycle import get_server
from ..utils.exceptions import safe_execute, handle_model_config_error, handle_gpu_memory_error


class ServerManager:
    """Server management for vLLM instances"""
    
    @staticmethod
    def start_server(
        model_name: str,
        port: int = DEFAULT_VLLM_PORT,
        target_memory_gb: Optional[float] = None,
        disable_torch_compilation: bool = False,
        gpu_runtime: GPURuntime = GPURuntime.CUDA
    ) -> VLLMServerConfig:
        """Start vLLM server with specified model and configuration"""
        
        # Initialize system info
        initialize_system(gpu_runtime)
        system = get_system_info()
        
        # Auto-detect target memory if not provided
        if target_memory_gb is None:
            if system.has_gpu:
                # Use 75% of GPU memory as target (with safety margins applied later)
                target_memory_gb = system.gpu_memory_gb * 0.75
            else:
                print("Error: No GPU detected and no target memory specified")
                sys.exit(1)
        
        # Get model configuration
        model_config = model_registry.get_model_config(model_name)
        if not model_config:
            handle_model_config_error(model_name)
        
        # Calculate optimal GPU utilization and context size
        gpu_manager = get_gpu_memory_manager()
        gpu_utilization, allocation_info = gpu_manager.calculate_gpu_utilization(target_memory_gb)
        recommended_context, context_analysis = gpu_manager.recommend_context_size(
            target_memory_gb, 
            model_size_gb=model_config.estimated_size_gb
        )
        
        print(f"Starting vLLM server...")
        print(f"   Model: {model_name}")
        print(f"   Type: {model_config.model_type.value}")
        print(f"   Estimated Size: {model_config.estimated_size_gb:.1f}GB")
        if system.has_gpu:
            print(f"   GPU: {system.gpu.name}")
            print(f"   GPU Memory: {system.gpu.memory_total_gb:.1f}GB total")
            print(f"   Target Usage: {target_memory_gb:.1f}GB ({allocation_info.get('utilization_percent', 0):.1f}%)")
        print(f"   Context: {recommended_context:,} tokens")
        print(f"   Quantization: {model_config.quantization or 'None'}")
        print(f"   Port: {port}")
        print(f"   Torch Compilation: {'Disabled' if disable_torch_compilation else 'Enabled'}")
        
        if allocation_info.get('warning'):
            print(f"Warning: {allocation_info['warning']}")
        
        # Create server configuration
        extra_args = model_config.extra_args.copy()
        if disable_torch_compilation:
            extra_args.update({
                "disable_log_stats": True,
                "disable_log_requests": True
            })
        
        config = VLLMServerConfig(
            model_path=model_name,
            host=DEFAULT_HOST,
            port=port,
            max_model_len=STANDARD_CONTEXT_SIZE,
            gpu_memory_utilization=gpu_utilization,
            max_num_batched_tokens=STANDARD_CONTEXT_SIZE,
            max_num_seqs=SINGLE_SEQUENCE,
            quantization=model_config.quantization,
            extra_args={
                "disable_sliding_window": True,
                "dtype": model_config.dtype,
                "trust_remote_code": True,
                "enable_prefix_caching": True,
                **extra_args
            }
        )
        
        # Build and display command
        cmd = config.to_command_args()
        print(f"\\nCommand: {' '.join(cmd)}")
        
        # Show detailed memory analysis
        print(f"\\nMemory Analysis:")
        print(f"   Model Size: {model_config.estimated_size_gb:.1f} GB")
        print(f"   KV Cache: {context_analysis.get('kv_cache_gb', 0):.2f} GB")
        print(f"   Total Est: {context_analysis.get('total_estimated_gb', 0):.2f} GB")
        print(f"   Efficiency: {context_analysis.get('memory_efficiency', 0):.1f}%")
        print(f"   vLLM Version: {system.vllm_version}")
        
        print(f"\\nStarting server...")
        print(f"   Use Ctrl+C to stop")
        print(f"   Server will be available at http://127.0.0.1:{port}")
        
        # Use simple server lifecycle management
        server = get_server()
        
        if server.start(cmd):
            # Wait for server to finish (or be interrupted)
            server.wait()
        else:
            print("Failed to start server")
            sys.exit(1)
        
        return config


def main():
    parser = argparse.ArgumentParser(description="Start vLLM server with any supported model")
    parser.add_argument("--model", required=True, 
                       help="Model name or path (e.g., 'Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4')")
    parser.add_argument("--port", type=int, default=DEFAULT_VLLM_PORT, 
                       help=f"Port for vLLM server (default: {DEFAULT_VLLM_PORT})")
    parser.add_argument("--target-memory", type=float, default=None,
                       help="Target GPU memory usage in GB (default: auto-detect from GPU)")
    parser.add_argument("--disable-torch-compilation", action="store_true",
                       help="Disable torch compilation for debugging")
    parser.add_argument("--gpu-runtime", 
                       choices=["cuda", "rocm", "xpu"],
                       default="cuda",
                       help="GPU runtime to use (only cuda supported currently)")
    
    args = parser.parse_args()
    
    safe_execute(
        ServerManager.start_server,
        "vLLM server startup",
        model_name=args.model,
        port=args.port,
        target_memory_gb=args.target_memory,
        disable_torch_compilation=args.disable_torch_compilation,
        gpu_runtime=GPURuntime(args.gpu_runtime)
    )


if __name__ == "__main__":
    main()