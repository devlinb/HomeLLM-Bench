from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from ..config.constants import (
    DEFAULT_HOST, GPU_MEMORY_UTILIZATION, STANDARD_CONTEXT_SIZE, 
    SINGLE_SEQUENCE, DEFAULT_MAX_TOKENS
)

# Default port for vLLM server
DEFAULT_VLLM_PORT = 8001


class VLLMServerConfig(BaseModel):
    """Configuration for vLLM server parameters"""
    
    # Model configuration
    model_path: str = Field(..., description="Path to the model file")
    quantization: Optional[str] = Field(default=None, description="Quantization method (e.g., 'gguf')")
    
    # Server configuration
    host: str = Field(default=DEFAULT_HOST, description="Server host")
    port: int = Field(default=DEFAULT_VLLM_PORT, description="Server port")
    
    # Memory and performance settings
    gpu_memory_utilization: float = Field(default=GPU_MEMORY_UTILIZATION, description="GPU memory utilization ratio")
    max_model_len: Optional[int] = Field(default=STANDARD_CONTEXT_SIZE, description="Maximum model sequence length")
    
    # Batch size settings (optimized for single user)
    max_num_batched_tokens: int = Field(default=STANDARD_CONTEXT_SIZE, description="Maximum tokens in a batch")
    max_num_seqs: int = Field(default=SINGLE_SEQUENCE, description="Maximum concurrent sequences")
    
    # Caching settings
    enable_prefix_caching: bool = Field(default=True, description="Enable prefix caching for better performance")
    
    # Advanced settings
    enforce_eager: bool = Field(default=False, description="Disable CUDA graphs for debugging")
    disable_log_stats: bool = Field(default=True, description="Disable automatic log stats")
    disable_log_requests: bool = Field(default=True, description="Disable request logging")
    
    # Additional arguments
    extra_args: Dict[str, Any] = Field(default_factory=dict, description="Additional command line arguments")
    
    def to_command_args(self) -> List[str]:
        """Convert configuration to command line arguments"""
        args = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-num-batched-tokens", str(self.max_num_batched_tokens),
            "--max-num-seqs", str(self.max_num_seqs),
        ]
        
        # Optional arguments
        if self.max_model_len:
            args.extend(["--max-model-len", str(self.max_model_len)])
        
        if self.quantization:
            args.extend(["--quantization", self.quantization])
        
        if self.enable_prefix_caching:
            args.append("--enable-prefix-caching")
        
        if self.enforce_eager:
            args.append("--enforce-eager")
        
        if self.disable_log_stats:
            args.append("--disable-log-stats")
        
        if self.disable_log_requests:
            args.append("--disable-log-requests")
        
        # Add extra arguments
        for key, value in self.extra_args.items():
            if isinstance(value, bool) and value:
                args.append(f"--{key.replace('_', '-')}")
            elif not isinstance(value, bool):
                args.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return args
    
    def get_description(self) -> str:
        """Get human-readable description of the configuration"""
        desc = f"""vLLM Server Configuration:
  Model: {self.model_path}
  Server: {self.host}:{self.port}
  Memory: {self.gpu_memory_utilization*100:.0f}% GPU utilization
  Context: {self.max_model_len or 'default'} max tokens per conversation
  Batch: {self.max_num_batched_tokens} max batched tokens, {self.max_num_seqs} max concurrent sequences
  Caching: {'enabled' if self.enable_prefix_caching else 'disabled'}
  Quantization: {self.quantization or 'none'}"""
        return desc


# Standard configuration factory
class VLLMConfigs:
    """Standard vLLM configuration for home LLM benchmarking"""
    
    @staticmethod
    def standard_config(model_path: str, port: int = DEFAULT_VLLM_PORT) -> VLLMServerConfig:
        """Standard configuration for home LLM benchmarking"""
        return VLLMServerConfig(
            model_path=model_path,
            port=port,
            extra_args={
                "disable_sliding_window": True  # Better for single user scenarios
            }
        )