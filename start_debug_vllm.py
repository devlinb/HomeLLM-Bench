#!/usr/bin/env python3
"""
Start vLLM server with debug configuration (no torch compilation)
"""
import subprocess
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.vllm_config import VLLMConfigs


def main():
    print("Starting vLLM with Debug Configuration (No Torch Compilation)")
    print("=" * 70)
    
    # Create debug configuration
    config = VLLMConfigs.debug_mode("./phi-3.5-mini-Q4_K.gguf", port=8001)
    
    print(config.get_description())
    print("\nCommand:")
    cmd = config.to_command_args()
    print(" ".join(cmd))
    
    print("\nStarting server...")
    print("=" * 70)
    
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