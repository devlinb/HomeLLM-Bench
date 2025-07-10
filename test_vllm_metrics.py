#!/usr/bin/env python3
"""
Test vLLM metrics collection
"""
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics.vllm_collector import VLLMMetricsCollector


def wait_for_server(collector, timeout=60):
    """Wait for vLLM server to be ready"""
    print("Waiting for vLLM server to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if collector.check_server_health():
            print("✅ Server is ready!")
            return True
        
        print("⏳ Server not ready yet, waiting...")
        time.sleep(5)
    
    print(f"❌ Server failed to start within {timeout} seconds")
    return False


def test_metrics_collection():
    """Test comprehensive metrics collection"""
    
    # Connect to vLLM server on port 8001
    collector = VLLMMetricsCollector(server_host="127.0.0.1", server_port=8001)
    
    # Wait for server to be ready
    if not wait_for_server(collector):
        return False
    
    print("\n" + "="*50)
    print("TESTING vLLM METRICS COLLECTION")
    print("="*50)
    
    # Test 1: Get system metrics
    print("\n1. System Metrics:")
    print("-" * 20)
    system_metrics = collector.get_system_metrics()
    for key, value in system_metrics.items():
        if key == 'gpus':
            print(f"  {key}: {len(value)} GPUs")
            for gpu in value:
                print(f"    GPU {gpu['gpu_id']}: {gpu['memory_used_mb']:.1f}MB used ({gpu['memory_percent']:.1f}%)")
        else:
            print(f"  {key}: {value}")
    
    # Test 2: Get server metrics
    print("\n2. Server Metrics:")
    print("-" * 20)
    server_metrics = collector.get_server_metrics()
    if server_metrics:
        for key, value in server_metrics.items():
            print(f"  {key}: {value}")
    else:
        print("  No server metrics available (may need to make a request first)")
    
    # Test 3: Generate text with metrics
    print("\n3. Generation with Metrics:")
    print("-" * 30)
    try:
        prompt = "What is 2+2? Answer briefly."
        print(f"📝 Prompt: {prompt}")
        
        generated_text, metrics = collector.generate_with_metrics(
            prompt=prompt,
            model_name="./phi-3.5-mini-Q4_K.gguf",
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"🤖 Generated: {generated_text.strip()}")
        print(f"\n📊 Generation Metrics:")
        print(f"  Prompt tokens: {metrics.prompt_tokens}")
        print(f"  Completion tokens: {metrics.completion_tokens}")
        print(f"  Total time: {metrics.total_generation_time:.3f}s")
        print(f"  Tokens/second: {metrics.tokens_per_second:.1f}")
        print(f"  Time to first token: {metrics.time_to_first_token:.3f}s")
        print(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
        if metrics.gpu_memory_used_mb:
            print(f"  GPU memory: {metrics.gpu_memory_used_mb:.1f}MB")
        if metrics.cache_hit_rate is not None:
            print(f"  Cache hit rate: {metrics.cache_hit_rate:.2f}")
        
        # Test 4: Second generation to test caching
        print("\n4. Second Generation (Test Caching):")
        print("-" * 40)
        prompt2 = "What is 3+3? Answer briefly."
        print(f"📝 Prompt: {prompt2}")
        
        generated_text2, metrics2 = collector.generate_with_metrics(
            prompt=prompt2,
            model_name="./phi-3.5-mini-Q4_K.gguf",
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"🤖 Generated: {generated_text2.strip()}")
        print(f"📊 Second Generation Metrics:")
        print(f"  Total time: {metrics2.total_generation_time:.3f}s")
        print(f"  Tokens/second: {metrics2.tokens_per_second:.1f}")
        if metrics2.cache_hit_rate is not None:
            print(f"  Cache hit rate: {metrics2.cache_hit_rate:.2f}")
        
        # Compare performance
        speedup = metrics.total_generation_time / metrics2.total_generation_time if metrics2.total_generation_time > 0 else 1
        print(f"  Performance change: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


if __name__ == "__main__":
    success = test_metrics_collection()
    if success:
        print("\n✅ vLLM metrics collection test completed successfully!")
    else:
        print("\n❌ vLLM metrics collection test failed!")
        sys.exit(1)