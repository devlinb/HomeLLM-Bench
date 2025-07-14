#!/usr/bin/env python3
"""
Compilation cache detection and management for vLLM models.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class CompilationCache:
    """Manages vLLM compilation cache detection and validation."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize compilation cache manager.
        
        Args:
            cache_dir: Optional custom cache directory. Defaults to vLLM's configured cache location
        """
        self.cache_dir = cache_dir or self._discover_vllm_cache_directory()
    
    def _discover_vllm_cache_directory(self) -> Path:
        """Discover vLLM's torch compile cache directory using proper environment variables."""
        import os
        
        # Check VLLM_CACHE_ROOT first
        vllm_cache_root = os.getenv("VLLM_CACHE_ROOT")
        if vllm_cache_root:
            return Path(vllm_cache_root) / "torch_compile_cache"
        
        # Check XDG_CACHE_HOME next
        xdg_cache_home = os.getenv("XDG_CACHE_HOME")
        if xdg_cache_home:
            return Path(xdg_cache_home) / "vllm" / "torch_compile_cache"
        
        # Default to ~/.cache/vllm/torch_compile_cache
        return Path.home() / ".cache" / "vllm" / "torch_compile_cache"
    
    def get_cache_hash(self, model_name: str, config_dict: Dict[str, Any]) -> str:
        """Generate cache hash that matches vLLM's internal logic.
        
        Args:
            model_name: Name or path of the model
            config_dict: Configuration dictionary with vLLM parameters
            
        Returns:
            10-character hash string matching vLLM's cache naming
        """
        # vLLM uses model + key config params for cache hash
        cache_key = {
            'model': model_name,
            'gpu_memory_utilization': config_dict.get('gpu_memory_utilization', 0.9),
            'max_model_len': config_dict.get('max_model_len', 32768),
            'quantization': config_dict.get('quantization'),
            'enable_prefix_caching': config_dict.get('enable_prefix_caching', True),
            'dtype': config_dict.get('dtype', 'auto'),
            'tensor_parallel_size': config_dict.get('tensor_parallel_size', 1),
            'enforce_eager': config_dict.get('enforce_eager', False),
        }
        
        # Create deterministic hash
        cache_str = json.dumps(cache_key, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()[:10]
    
    def is_compiled(self, model_name: str, config_dict: Dict[str, Any]) -> bool:
        """Check if model is already compiled for given configuration.
        
        Args:
            model_name: Name or path of the model
            config_dict: Configuration dictionary with vLLM parameters
            
        Returns:
            True if compilation cache exists and is complete
        """
        # First try exact hash match
        cache_hash = self.get_cache_hash(model_name, config_dict)
        cache_path = self.cache_dir / cache_hash / "rank_0_0"
        
        # Check for essential compiled files that indicate completion
        required_files = [
            "computation_graph.py",
            "transformed_code.py", 
            "vllm_compile_cache.py"
        ]
        
        if all((cache_path / f).exists() for f in required_files):
            return True
        
        # If exact match fails, check if ANY valid cache exists for this model
        # This handles cases where vLLM's internal hash algorithm differs
        return self.has_any_valid_cache_for_model(model_name)
    
    def has_any_valid_cache_for_model(self, model_name: str) -> bool:
        """Check if any valid compilation cache exists for the model (regardless of config)."""
        if not self.cache_dir.exists():
            return False
        
        required_files = [
            "computation_graph.py",
            "transformed_code.py", 
            "vllm_compile_cache.py"
        ]
        
        # Check all cache directories for valid caches
        for cache_dir in self.cache_dir.iterdir():
            if cache_dir.is_dir():
                rank_dir = cache_dir / "rank_0_0"
                if rank_dir.exists() and all((rank_dir / f).exists() for f in required_files):
                    return True
        
        return False
    
    def get_cache_path(self, model_name: str, config_dict: Dict[str, Any]) -> Path:
        """Get cache directory path for given model and configuration.
        
        Args:
            model_name: Name or path of the model
            config_dict: Configuration dictionary with vLLM parameters
            
        Returns:
            Path to the cache directory
        """
        cache_hash = self.get_cache_hash(model_name, config_dict)
        return self.cache_dir / cache_hash
    
    def get_cache_info(self, model_name: str, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed information about the compilation cache.
        
        Args:
            model_name: Name or path of the model
            config_dict: Configuration dictionary with vLLM parameters
            
        Returns:
            Dictionary with cache information
        """
        cache_path = self.get_cache_path(model_name, config_dict)
        rank_path = cache_path / "rank_0_0"
        
        info = {
            'cache_hash': self.get_cache_hash(model_name, config_dict),
            'cache_path': str(cache_path),
            'exists': cache_path.exists(),
            'is_compiled': self.is_compiled(model_name, config_dict),
            'files': []
        }
        
        if rank_path.exists():
            # Get file sizes and modification times
            for file_path in rank_path.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    info['files'].append({
                        'name': file_path.name,
                        'size_mb': stat.st_size / (1024 * 1024),
                        'modified': stat.st_mtime
                    })
        
        return info
    
    def clear_cache(self, model_name: str, config_dict: Dict[str, Any]) -> bool:
        """Clear compilation cache for given model and configuration.
        
        Args:
            model_name: Name or path of the model
            config_dict: Configuration dictionary with vLLM parameters
            
        Returns:
            True if cache was cleared successfully
        """
        import shutil
        
        cache_path = self.get_cache_path(model_name, config_dict)
        
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                return True
            except Exception:
                return False
        
        return True  # Nothing to clear
    
    def list_all_caches(self) -> list:
        """List all compilation caches in the cache directory.
        
        Returns:
            List of cache hash directories
        """
        if not self.cache_dir.exists():
            return []
        
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]


def get_compilation_cache() -> CompilationCache:
    """Get a default compilation cache instance."""
    return CompilationCache()