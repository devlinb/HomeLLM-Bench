"""
System information singleton for HomeLLM-Bench
Collects hardware and software info once at startup
"""
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import platform
import psutil
import sys

from ..utils.gpu_memory import GPUMemoryManager


class GPUVendor(Enum):
    NVIDIA = "nvidia"
    AMD = "amd" 
    INTEL = "intel"


class GPURuntime(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"


@dataclass(frozen=True)
class GPUInfo:
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: str
    driver_version: str
    runtime_version: str  # CUDA 12.2, ROCm 5.4, etc.
    vendor: GPUVendor
    runtime: GPURuntime


@dataclass(frozen=True)
class SystemInfo:
    # GPU hardware info
    gpu: Optional[GPUInfo]
    
    # Software versions
    vllm_version: str
    
    # System specs
    platform: str
    cpu_cores: int
    system_memory_gb: float
    
    # Convenience properties
    @property
    def has_gpu(self) -> bool:
        return self.gpu is not None
    
    @property
    def gpu_memory_gb(self) -> float:
        return self.gpu.memory_total_gb if self.gpu else 0


def _initialize_system_info(gpu_runtime: GPURuntime = GPURuntime.CUDA) -> SystemInfo:
    """Initialize system info. Only CUDA is supported currently."""
    
    gpu_info = None
    vllm_version = "Unknown"
    
    if gpu_runtime == GPURuntime.CUDA:
        # Use existing CUDA-based GPUMemoryManager
        gpu_manager = GPUMemoryManager()
        vllm_version = gpu_manager.get_vllm_version()
        
        if gpu_manager.gpu_info:
            gpu_info = GPUInfo(
                name=gpu_manager.gpu_info.name,
                memory_total_gb=gpu_manager.gpu_info.total_memory_mb / 1024,
                memory_free_gb=gpu_manager.gpu_info.free_memory_mb / 1024,
                compute_capability=gpu_manager.gpu_info.compute_capability,
                driver_version=gpu_manager.gpu_info.driver_version,
                runtime_version=gpu_manager.gpu_info.cuda_version,
                vendor=GPUVendor.NVIDIA,  # CUDA implies NVIDIA for now
                runtime=GPURuntime.CUDA
            )
    elif gpu_runtime in [GPURuntime.ROCM, GPURuntime.XPU]:
        # Exit with error for unsupported runtimes
        print(f"Error: {gpu_runtime.value} support not implemented yet")
        print("Currently only CUDA is supported")
        sys.exit(1)
    
    return SystemInfo(
        gpu=gpu_info,
        vllm_version=vllm_version,
        platform=platform.system(),
        cpu_cores=psutil.cpu_count(logical=False) or 0,
        system_memory_gb=psutil.virtual_memory().total / (1024**3)
    )


# Global system info - must be initialized explicitly
SYSTEM: Optional[SystemInfo] = None


def initialize_system(gpu_runtime: GPURuntime = GPURuntime.CUDA) -> None:
    """Initialize system info singleton - call this first in main scripts"""
    global SYSTEM
    SYSTEM = _initialize_system_info(gpu_runtime)


def get_system_info() -> SystemInfo:
    """Get system info. Must call initialize_system() first."""
    global SYSTEM
    if SYSTEM is None:
        raise RuntimeError("SystemInfo not initialized. Call initialize_system() first.")
    return SYSTEM