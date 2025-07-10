"""
GPU memory detection and allocation utilities
"""
import subprocess
import json
import re
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from ..config.constants import (
    MEMORY_SAFETY_MARGIN, OVERHEAD_BUFFER_GB, DEFAULT_HIDDEN_SIZE, 
    DEFAULT_NUM_LAYERS, STANDARD_CONTEXT_SIZE, MB_TO_GB, BYTES_TO_GB
)


@dataclass
class GPUInfo:
    """GPU information container"""
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    compute_capability: str
    driver_version: str
    cuda_version: str


class GPUMemoryManager:
    """Manages GPU memory allocation calculations"""
    
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
    
    def _get_gpu_info(self) -> Optional[GPUInfo]:
        """Get GPU information from nvidia-smi"""
        try:
            # Get GPU memory and basic info
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=name,memory.total,memory.free,memory.used,compute_cap,driver_version",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)
            
            line = result.stdout.strip().split('\n')[0]  # First GPU
            parts = [p.strip() for p in line.split(',')]
            
            # Get CUDA version (fallback if query fails)
            try:
                cuda_result = subprocess.run([
                    "nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, check=True)
                cuda_version = cuda_result.stdout.strip() or "Unknown"
            except:
                # Fallback: extract from nvidia-smi output
                try:
                    smi_result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
                    import re
                    match = re.search(r'CUDA Version: (\d+\.\d+)', smi_result.stdout)
                    cuda_version = match.group(1) if match else "Unknown"
                except:
                    cuda_version = "Unknown"
            
            return GPUInfo(
                name=parts[0],
                total_memory_mb=int(parts[1]),
                free_memory_mb=int(parts[2]),
                used_memory_mb=int(parts[3]),
                compute_capability=parts[4],
                driver_version=parts[5],
                cuda_version=cuda_version
            )
            
        except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
            return None
    
    def get_vllm_version(self) -> str:
        """Get vLLM version"""
        try:
            import vllm
            return vllm.__version__
        except ImportError:
            try:
                result = subprocess.run(
                    ["python", "-c", "import vllm; print(vllm.__version__)"],
                    capture_output=True, text=True, check=True
                )
                return result.stdout.strip()
            except:
                return "Unknown"
    
    def calculate_gpu_utilization(self, target_memory_gb: float) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal GPU memory utilization for target memory usage
        
        Args:
            target_memory_gb: Desired total GPU memory usage in GB
            
        Returns:
            Tuple of (utilization_ratio, allocation_info)
        """
        if not self.gpu_info:
            # Fallback for systems without nvidia-smi
            return 0.5, {
                "error": "Could not detect GPU",
                "fallback_utilization": 0.5,
                "target_memory_gb": target_memory_gb
            }
        
        total_memory_gb = self.gpu_info.total_memory_mb / MB_TO_GB
        free_memory_gb = self.gpu_info.free_memory_mb / MB_TO_GB
        used_memory_gb = self.gpu_info.used_memory_mb / MB_TO_GB
        
        # Apply safety margin to target - vLLM often uses more than requested
        safe_target = target_memory_gb * MEMORY_SAFETY_MARGIN
        
        # Ensure we don't exceed available memory
        available_memory_gb = min(free_memory_gb, total_memory_gb * 0.95)  # 5% safety margin
        
        if safe_target > available_memory_gb:
            # Scale down target to fit available memory
            actual_target = available_memory_gb * 0.9  # 10% buffer within available
            utilization = actual_target / total_memory_gb
            warning = f"Target {target_memory_gb:.1f}GB (safe: {safe_target:.1f}GB) exceeds available {available_memory_gb:.1f}GB, using {actual_target:.1f}GB"
        else:
            utilization = safe_target / total_memory_gb
            actual_target = safe_target
            warning = f"Using {safe_target:.1f}GB ({int(MEMORY_SAFETY_MARGIN*100)}% of {target_memory_gb:.1f}GB target) for safety margin"
        
        # Ensure utilization is within reasonable bounds
        utilization = max(0.1, min(0.85, utilization))
        
        allocation_info = {
            "gpu_name": self.gpu_info.name,
            "total_memory_gb": total_memory_gb,
            "free_memory_gb": free_memory_gb,
            "used_memory_gb": used_memory_gb,
            "target_memory_gb": target_memory_gb,
            "actual_memory_gb": actual_target,
            "utilization_ratio": utilization,
            "utilization_percent": utilization * 100,
            "compute_capability": self.gpu_info.compute_capability,
            "driver_version": self.gpu_info.driver_version,
            "cuda_version": self.gpu_info.cuda_version,
            "vllm_version": self.get_vllm_version(),
            "warning": warning
        }
        
        return utilization, allocation_info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information for benchmarks"""
        info = {
            "vllm_version": self.get_vllm_version(),
            "gpu_info": None
        }
        
        if self.gpu_info:
            info["gpu_info"] = {
                "name": self.gpu_info.name,
                "total_memory_mb": self.gpu_info.total_memory_mb,
                "total_memory_gb": self.gpu_info.total_memory_mb / MB_TO_GB,
                "compute_capability": self.gpu_info.compute_capability,
                "driver_version": self.gpu_info.driver_version,
                "cuda_version": self.gpu_info.cuda_version
            }
        
        return info
    
    def estimate_kv_cache_memory(self, context_length: int, hidden_size: int = DEFAULT_HIDDEN_SIZE, 
                                num_layers: int = DEFAULT_NUM_LAYERS, num_sequences: int = 1) -> float:
        """
        Estimate KV cache memory usage in GB
        
        Args:
            context_length: Maximum context length
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            num_sequences: Number of concurrent sequences
            
        Returns:
            Estimated KV cache memory in GB
        """
        # Formula: context_length × num_sequences × hidden_size × num_layers × 2 (K,V) × 2 (fp16 bytes)
        kv_cache_bytes = context_length * num_sequences * hidden_size * num_layers * 2 * 2
        return kv_cache_bytes / BYTES_TO_GB
    
    def recommend_context_size(self, target_memory_gb: float, model_size_gb: float = 2.0,
                             hidden_size: int = DEFAULT_HIDDEN_SIZE, num_layers: int = DEFAULT_NUM_LAYERS) -> Tuple[int, Dict[str, Any]]:
        """
        Recommend optimal context size for target memory usage
        
        Args:
            target_memory_gb: Target total memory usage
            model_size_gb: Estimated model size in GB
            hidden_size: Model hidden dimension  
            num_layers: Number of transformer layers
            
        Returns:
            Tuple of (recommended_context_size, analysis_info)
        """
        # Use standard context size for consistent benchmarking
        context_size = STANDARD_CONTEXT_SIZE
        
        # Use safety margin for target allocation
        safe_target = target_memory_gb * MEMORY_SAFETY_MARGIN
        available_for_kv = safe_target - model_size_gb - OVERHEAD_BUFFER_GB
        
        if available_for_kv <= 0:
            return context_size, {
                "error": "Insufficient memory",
                "target_memory_gb": target_memory_gb,
                "safe_target_gb": safe_target,
                "model_size_gb": model_size_gb,
                "available_for_kv_gb": available_for_kv
            }
        
        # Calculate KV cache memory for standard context size
        kv_memory = self.estimate_kv_cache_memory(context_size, hidden_size, num_layers)
        total_estimated = model_size_gb + kv_memory + OVERHEAD_BUFFER_GB
        
        analysis = {
            "recommended_context": context_size,
            "target_memory_gb": target_memory_gb,
            "safe_target_gb": safe_target,
            "model_size_gb": model_size_gb,
            "kv_cache_gb": kv_memory,
            "overhead_gb": OVERHEAD_BUFFER_GB,
            "total_estimated_gb": total_estimated,
            "available_for_kv_gb": available_for_kv,
            "memory_efficiency": (total_estimated / safe_target) * 100 if safe_target > 0 else 0
        }
        
        return context_size, analysis


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get global GPU memory manager instance"""
    return GPUMemoryManager()


if __name__ == "__main__":
    # Test the GPU memory manager
    manager = GPUMemoryManager()
    
    if manager.gpu_info:
        print(f"GPU: {manager.gpu_info.name}")
        print(f"Memory: {manager.gpu_info.total_memory_mb}MB total")
        print(f"CUDA: {manager.gpu_info.cuda_version}")
        print(f"Compute: {manager.gpu_info.compute_capability}")
        
        # Test 8GB target
        util, info = manager.calculate_gpu_utilization(8.0)
        print(f"\nFor 8GB target:")
        print(f"Utilization: {util:.2f} ({util*100:.1f}%)")
        print(f"Actual memory: {info['actual_memory_gb']:.2f}GB")
        
        # Test context recommendation
        context, analysis = manager.recommend_context_size(8.0)
        print(f"\nRecommended context: {context:,} tokens")
        print(f"Total estimated: {analysis['total_estimated_gb']:.2f}GB")
    else:
        print("No GPU detected")