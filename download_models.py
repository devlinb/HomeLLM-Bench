#!/usr/bin/env python3
"""
Model download automation for benchmark suite
"""
import os
import sys
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import time


class ModelDownloader:
    """Handles automatic model downloading and verification"""
    
    def __init__(self, download_dir: str = "."):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Predefined model configurations
        self.models = {
            "phi-3.5-mini-4bit": {
                "name": "Phi-3.5 Mini 4-bit GGUF",
                "url": "https://huggingface.co/legraphista/unsloth-Phi-3.5-mini-instruct-IMat-GGUF/resolve/main/unsloth-Phi-3.5-mini-instruct.Q4_K.gguf?download=true",
                "filename": "phi-3.5-mini-Q4_K.gguf",
                "size_gb": 2.3,
                "sha256": None,  # Could add for verification
                "description": "4-bit quantized Phi-3.5 Mini model, good for testing and development"
            },
            "phi-3.5-mini-8bit": {
                "name": "Phi-3.5 Mini 8-bit GGUF",
                "url": "https://huggingface.co/legraphista/unsloth-Phi-3.5-mini-instruct-IMat-GGUF/resolve/main/unsloth-Phi-3.5-mini-instruct.Q8_0.gguf?download=true",
                "filename": "phi-3.5-mini-Q8_0.gguf",
                "size_gb": 4.1,
                "sha256": None,
                "description": "8-bit quantized Phi-3.5 Mini model, higher quality than 4-bit"
            },
            "phi-3.5-mini-fp16": {
                "name": "Phi-3.5 Mini FP16 GGUF",
                "url": "https://huggingface.co/legraphista/unsloth-Phi-3.5-mini-instruct-IMat-GGUF/resolve/main/unsloth-Phi-3.5-mini-instruct.f16.gguf?download=true",
                "filename": "phi-3.5-mini-f16.gguf",
                "size_gb": 7.6,
                "sha256": None,
                "description": "Full precision FP16 model, highest quality but largest size"
            }
        }
    
    def check_disk_space(self, required_gb: float) -> bool:
        """Check if enough disk space is available"""
        try:
            stat = os.statvfs(self.download_dir)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            return free_gb >= required_gb * 1.2  # 20% buffer
        except:
            return True  # Assume space is available if check fails
    
    def get_file_size(self, url: str) -> Optional[int]:
        """Get file size from HTTP headers"""
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            if response.status_code == 200:
                return int(response.headers.get('content-length', 0))
        except:
            pass
        return None
    
    def download_with_progress(self, url: str, filepath: Path) -> bool:
        """Download file with progress bar"""
        try:
            print(f"📥 Downloading from: {url}")
            print(f"📁 Saving to: {filepath}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            gb_downloaded = downloaded / (1024**3)
                            gb_total = total_size / (1024**3)
                            print(f"\r📊 Progress: {progress:.1f}% ({gb_downloaded:.2f}/{gb_total:.2f} GB)", end='', flush=True)
            
            print("\n✅ Download completed!")
            return True
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            # Clean up partial download
            if filepath.exists():
                filepath.unlink()
            return False
    
    def verify_file(self, filepath: Path, expected_sha256: Optional[str] = None) -> bool:
        """Verify downloaded file"""
        if not filepath.exists():
            return False
        
        # Basic size check
        size_gb = filepath.stat().st_size / (1024**3)
        print(f"📏 File size: {size_gb:.2f} GB")
        
        # SHA256 verification if provided
        if expected_sha256:
            print("🔍 Verifying file integrity...")
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_hash = sha256_hash.hexdigest()
            if actual_hash.lower() == expected_sha256.lower():
                print("✅ File integrity verified!")
                return True
            else:
                print(f"❌ File integrity check failed!")
                print(f"   Expected: {expected_sha256}")
                print(f"   Actual:   {actual_hash}")
                return False
        
        return True
    
    def download_model(self, model_key: str, force: bool = False) -> bool:
        """Download a specific model"""
        if model_key not in self.models:
            print(f"❌ Unknown model: {model_key}")
            print(f"Available models: {list(self.models.keys())}")
            return False
        
        model_info = self.models[model_key]
        filepath = self.download_dir / model_info["filename"]
        
        print(f"🤖 Model: {model_info['name']}")
        print(f"📦 Size: ~{model_info['size_gb']} GB")
        print(f"📝 Description: {model_info['description']}")
        
        # Check if file already exists
        if filepath.exists() and not force:
            if self.verify_file(filepath, model_info["sha256"]):
                print(f"✅ Model already exists and is valid: {filepath}")
                return True
            else:
                print(f"⚠️ Existing file appears corrupted, re-downloading...")
        
        # Check disk space
        if not self.check_disk_space(model_info["size_gb"]):
            print(f"❌ Insufficient disk space for {model_info['size_gb']} GB file")
            return False
        
        # Download the model
        print(f"\n🚀 Starting download...")
        return self.download_with_progress(model_info["url"], filepath)
    
    def list_models(self) -> None:
        """List available models"""
        print("📋 Available Models:")
        print("=" * 60)
        
        for key, info in self.models.items():
            print(f"\n🔑 Key: {key}")
            print(f"   Name: {info['name']}")
            print(f"   Size: ~{info['size_gb']} GB")
            print(f"   File: {info['filename']}")
            print(f"   Description: {info['description']}")
            
            # Check if already downloaded
            filepath = self.download_dir / info["filename"]
            if filepath.exists():
                size_gb = filepath.stat().st_size / (1024**3)
                print(f"   Status: ✅ Downloaded ({size_gb:.2f} GB)")
            else:
                print(f"   Status: ❌ Not downloaded")
    
    def download_recommended(self) -> bool:
        """Download the recommended model for benchmarking"""
        print("🎯 Downloading recommended model for benchmarking...")
        return self.download_model("phi-3.5-mini-4bit")
    
    def setup_benchmark_environment(self) -> bool:
        """Set up complete environment for benchmarking"""
        print("🔧 Setting up benchmark environment...")
        print("=" * 50)
        
        # Download recommended model
        if not self.download_recommended():
            return False
        
        # Verify vLLM can find the model
        model_path = self.download_dir / "phi-3.5-mini-Q4_K.gguf"
        if not model_path.exists():
            print("❌ Model file not found after download")
            return False
        
        print(f"\n✅ Environment setup complete!")
        print(f"📁 Model location: {model_path.absolute()}")
        print(f"🚀 Ready to run benchmarks!")
        
        return True


def main():
    """Main CLI interface for model downloading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download models for LLM benchmarking")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Download specific model by key")
    parser.add_argument("--setup", action="store_true", help="Setup complete environment")
    parser.add_argument("--force", action="store_true", help="Force re-download even if file exists")
    parser.add_argument("--dir", type=str, default=".", help="Download directory (default: current)")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.dir)
    
    if args.list:
        downloader.list_models()
    elif args.setup:
        success = downloader.setup_benchmark_environment()
        sys.exit(0 if success else 1)
    elif args.model:
        success = downloader.download_model(args.model, args.force)
        sys.exit(0 if success else 1)
    else:
        print("🤖 LLM Model Downloader")
        print("=" * 30)
        print("\nUsage examples:")
        print("  python download_models.py --list          # List available models")
        print("  python download_models.py --setup         # Setup complete environment")
        print("  python download_models.py --model phi-3.5-mini-4bit  # Download specific model")
        print("\nFor complete setup, run:")
        print("  python download_models.py --setup")


if __name__ == "__main__":
    main()