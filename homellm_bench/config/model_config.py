"""
Model configuration definitions for different LLM models
"""
from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum


class ModelType(Enum):
    """Model family types"""
    PHI3 = "phi3"
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    model_type: ModelType
    chat_template: str
    context_size: int
    estimated_size_gb: float  # Model size estimate for memory planning
    quantization: Optional[str] = None
    dtype: str = "auto"
    extra_args: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = {}


class ModelRegistry:
    """Registry of supported models and their configurations"""
    
    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model configurations"""
        
        # Phi-3.5 models
        self.register_model(ModelConfig(
            name="phi-3.5-mini-4bit",
            model_type=ModelType.PHI3,
            chat_template="phi3",
            context_size=128000,
            estimated_size_gb=2.5,  # 4-bit quantized Phi-3.5 Mini
            quantization="gguf",
            dtype="auto"
        ))
        
        self.register_model(ModelConfig(
            name="./phi-3.5-mini-Q4_K.gguf",
            model_type=ModelType.PHI3,
            chat_template="phi3",
            context_size=128000,
            estimated_size_gb=2.5,  # 4-bit quantized Phi-3.5 Mini
            quantization="gguf",
            dtype="auto"
        ))
        
        self.register_model(ModelConfig(
            name="microsoft/Phi-3.5-mini-instruct",
            model_type=ModelType.PHI3,
            chat_template="phi3",
            context_size=128000,
            estimated_size_gb=7.0,  # Full precision Phi-3.5 Mini
            dtype="bfloat16"
        ))
        
        # Qwen models
        self.register_model(ModelConfig(
            name="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            model_type=ModelType.QWEN,
            chat_template="qwen",
            context_size=32768,
            estimated_size_gb=2.0,  # GPTQ Int4 quantized
            quantization=None,  # Let vLLM auto-detect best quantization (gptq_marlin on NVIDIA)
            dtype="auto"
        ))
        
        self.register_model(ModelConfig(
            name="Qwen/Qwen2.5-3B-Instruct",
            model_type=ModelType.QWEN,
            chat_template="qwen",
            context_size=32768,
            estimated_size_gb=6.0,  # Full precision Qwen 3B
            dtype="bfloat16"
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a new model configuration"""
        self._models[config.name] = config
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a model"""
        return self._models.get(model_name)
    
    def get_model_type(self, model_name: str) -> Optional[ModelType]:
        """Get the model type for a given model name"""
        config = self.get_model_config(model_name)
        return config.model_type if config else None
    
    def get_chat_template(self, model_name: str) -> Optional[str]:
        """Get the chat template name for a model"""
        config = self.get_model_config(model_name)
        return config.chat_template if config else None
    
    def get_context_size(self, model_name: str) -> Optional[int]:
        """Get the context size for a model"""
        config = self.get_model_config(model_name)
        return config.context_size if config else None
    
    def list_models(self) -> Dict[str, ModelConfig]:
        """List all registered models"""
        return self._models.copy()
    
    def infer_model_config(self, model_name: str) -> ModelConfig:
        """Infer model configuration from model name patterns"""
        model_name_lower = model_name.lower()
        
        # Check if we have an exact match
        if model_name in self._models:
            return self._models[model_name]
        
        # Infer from common patterns
        if "phi" in model_name_lower:
            return ModelConfig(
                name=model_name,
                model_type=ModelType.PHI3,
                chat_template="phi3",
                context_size=128000,
                quantization="gptq" if "gptq" in model_name_lower else "gguf" if ".gguf" in model_name_lower else None,
                dtype="auto"
            )
        elif "qwen" in model_name_lower:
            return ModelConfig(
                name=model_name,
                model_type=ModelType.QWEN,
                chat_template="qwen",
                context_size=32768,
                quantization="gptq" if "gptq" in model_name_lower else "gguf" if ".gguf" in model_name_lower else None,
                dtype="auto"
            )
        elif "llama" in model_name_lower:
            return ModelConfig(
                name=model_name,
                model_type=ModelType.LLAMA,
                chat_template="llama",
                context_size=4096,  # Conservative default
                quantization="gptq" if "gptq" in model_name_lower else "gguf" if ".gguf" in model_name_lower else None,
                dtype="auto"
            )
        else:
            # Generic fallback
            return ModelConfig(
                name=model_name,
                model_type=ModelType.PHI3,  # Default to phi3 for now
                chat_template="phi3",
                context_size=8192,  # Conservative default
                dtype="auto"
            )


# Global model registry instance
model_registry = ModelRegistry()
