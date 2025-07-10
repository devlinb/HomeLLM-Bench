#!/usr/bin/env python3
"""
Start vLLM server with optimized single-user configuration
"""
import subprocess
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.vllm_config import VLLMConfigs


def main():
    print("Starting vLLM with Single-User Optimized Configuration")
    print("=" * 60)
    
    # Create optimized configuration
    config = VLLMConfigs.single_user_optimized("./phi-3.5-mini-Q4_K.gguf", port=8001)
    
    print(config.get_description())
    print("\nCommand:")
    cmd = config.to_command_args()
    print(" ".join(cmd))
    
    print("\nStarting server...")
    print("=" * 60)
    
    # Start the server
    process = subprocess.Popen(cmd)
    
    try:
        # Wait for the process
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    main()