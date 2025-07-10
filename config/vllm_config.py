from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class VLLMServerConfig(BaseModel):
    """Configuration for vLLM server parameters"""
    
    # Model configuration
    model_path: str = Field(..., description="Path to the model file")
    quantization: Optional[str] = Field(default=None, description="Quantization method (e.g., 'gguf')")
    
    # Server configuration
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    
    # Memory and performance settings
    gpu_memory_utilization: float = Field(default=0.8, description="GPU memory utilization ratio")
    max_model_len: Optional[int] = Field(default=None, description="Maximum model sequence length")
    
    # Batch size settings (optimized for single user)
    max_num_batched_tokens: int = Field(default=512, description="Maximum tokens in a batch (single-user optimized)")
    max_num_seqs: int = Field(default=2, description="Maximum concurrent sequences (single-user optimized)")
    
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


# Predefined configurations for different use cases
class VLLMConfigs:
    """Predefined vLLM configurations"""
    
    @staticmethod
    def single_user_optimized(model_path: str, port: int = 8001) -> VLLMServerConfig:
        """Optimized configuration for single-user benchmarking"""
        return VLLMServerConfig(
            model_path=model_path,
            port=port,
            quantization="gguf",
            max_model_len=2048,  # Good for long conversations
            gpu_memory_utilization=0.6,  # Conservative to leave room for system
            max_num_batched_tokens=512,  # Reduced for single user
            max_num_seqs=2,  # Allow for conversation context
            enable_prefix_caching=True,
            disable_log_stats=True,
            disable_log_requests=True
        )
    
    @staticmethod
    def long_context_optimized(model_path: str, port: int = 8001) -> VLLMServerConfig:
        """Optimized for very long context conversations"""
        return VLLMServerConfig(
            model_path=model_path,
            port=port,
            quantization="gguf",
            max_model_len=4096,  # Longer context
            gpu_memory_utilization=0.7,
            max_num_batched_tokens=1024,  # Bit larger for long contexts
            max_num_seqs=1,  # Single conversation at a time
            enable_prefix_caching=True,
            disable_log_stats=True,
            disable_log_requests=True
        )
    
    @staticmethod
    def debug_mode(model_path: str, port: int = 8001) -> VLLMServerConfig:
        """Configuration for debugging and development"""
        return VLLMServerConfig(
            model_path=model_path,
            port=port,
            quantization="gguf",
            max_model_len=1024,
            gpu_memory_utilization=0.5,
            max_num_batched_tokens=256,
            max_num_seqs=1,
            enable_prefix_caching=False,  # Disable for consistent timing
            enforce_eager=True,  # Easier debugging
            disable_log_stats=False,  # Keep logs for debugging
            disable_log_requests=False
        )


if __name__ == "__main__":
    # Example usage
    config = VLLMConfigs.single_user_optimized("./phi-3.5-mini-Q4_K.gguf")
    
    print("Single User Optimized Configuration:")
    print("=" * 50)
    print(config.get_description())
    
    print("\nCommand Line:")
    print(" ".join(config.to_command_args()))