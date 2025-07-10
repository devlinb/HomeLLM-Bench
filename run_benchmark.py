#!/usr/bin/env python3
"""
Complete LLM benchmark runner with enhanced system metrics and output formatting
"""
import time
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from metrics.vllm_collector import VLLMMetricsCollector
from metrics.system_collector import SystemMetricsCollector
from metrics.schemas import GenerationMetrics, ConversationBenchmarkResult
from schemas.conversation import Conversation, Message
from templates.phi3 import Phi3Template
from engines.vllm_engine import VLLMEngine
from output.formatters import BenchmarkFormatter


class BenchmarkRunner:
    """Complete benchmark runner with system monitoring"""
    
    def __init__(self, 
                 model_path: str,
                 host: str = "127.0.0.1",
                 port: int = 8001,
                 output_dir: str = "results"):
        
        self.model_path = model_path
        self.host = host
        self.port = port
        
        # Initialize components
        self.system_collector = SystemMetricsCollector()
        self.formatter = BenchmarkFormatter(output_dir)
        self.template = Phi3Template()
        
        # Will be initialized when benchmark starts
        self.engine: Optional[VLLMEngine] = None
        
    def prepare_conversations(self) -> List[Conversation]:
        """Prepare test conversations for benchmarking"""
        conversations = []
        
        # Conversation 1: Simple Q&A
        conversations.append(Conversation(
            name="Simple Q&A",
            messages=[
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="2+2 equals 4."),
                Message(role="user", content="What about 3+3?"),
                Message(role="assistant", content="3+3 equals 6."),
                Message(role="user", content="Can you explain basic addition?")
            ]
        ))
        
        # Conversation 2: Code Discussion
        conversations.append(Conversation(
            name="Code Discussion",
            messages=[
                Message(role="user", content="Can you write a Python function to calculate fibonacci numbers?"),
                Message(role="assistant", content="Here's a Python function for fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"),
                Message(role="user", content="That's recursive. Can you show me an iterative version?"),
                Message(role="assistant", content="Sure! Here's an iterative version:\n\n```python\ndef fibonacci_iterative(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```"),
                Message(role="user", content="Which approach is more efficient for large numbers?")
            ]
        ))
        
        # Conversation 3: Long Context
        conversations.append(Conversation(
            name="Long Context",
            messages=[
                Message(role="user", content="I'm working on a project that involves natural language processing. I need to understand the differences between various transformer architectures like BERT, GPT, and T5. Can you explain the key differences and when to use each?"),
                Message(role="assistant", content="Great question! Here are the key differences between these transformer architectures:\n\n**BERT (Bidirectional Encoder Representations from Transformers):**\n- Uses only the encoder part of the transformer\n- Bidirectional attention (can see both left and right context)\n- Pre-trained with masked language modeling\n- Best for: Classification, sentiment analysis, question answering\n\n**GPT (Generative Pre-trained Transformer):**\n- Uses only the decoder part of the transformer\n- Unidirectional attention (can only see left context)\n- Pre-trained with autoregressive language modeling\n- Best for: Text generation, completion tasks\n\n**T5 (Text-to-Text Transfer Transformer):**\n- Uses both encoder and decoder\n- Treats all tasks as text-to-text problems\n- Pre-trained with span corruption\n- Best for: Translation, summarization, general text transformation"),
                Message(role="user", content="That's very helpful! Can you dive deeper into the training objectives for each? I'm particularly interested in how the masking strategies differ."),
                Message(role="assistant", content="Absolutely! The training objectives and masking strategies are crucial differences:\n\n**BERT Masking Strategy:**\n- Random masking: 15% of tokens are selected\n- Of those 15%: 80% replaced with [MASK], 10% replaced with random token, 10% kept unchanged\n- Model learns to predict original tokens\n- Bidirectional context helps with understanding\n\n**GPT Training:**\n- No masking - uses causal attention\n- Predicts next token given all previous tokens\n- Each position can only attend to previous positions\n- Creates strong generative capabilities\n\n**T5 Approach:**\n- Span corruption: corrupts random spans of text\n- Spans replaced with sentinel tokens\n- Model learns to generate the corrupted spans\n- More flexible than single-token prediction"),
                Message(role="user", content="Now I'm curious about the computational requirements. How do they compare in terms of memory usage and training time?")
            ]
        ))
        
        return conversations
    
    def run_conversation_benchmark(self, conversation: Conversation) -> ConversationBenchmarkResult:
        """Run benchmark for a single conversation"""
        print(f"\n🔄 Benchmarking conversation: {conversation.name}")
        print("-" * 50)
        
        turn_metrics = []
        total_start_time = time.time()
        
        # Convert conversation to chat format, excluding the last user message which we'll generate
        messages_for_template = []
        for i, message in enumerate(conversation.messages[:-1]):  # Exclude last message
            messages_for_template.append({
                "role": message.role.value,
                "content": message.content
            })
        
        # Process each turn (we'll generate responses for user messages)
        current_messages = []
        for i, message in enumerate(conversation.messages):
            current_messages.append({
                "role": message.role.value,
                "content": message.content
            })
            
            # If this is a user message and not the last message, generate assistant response
            if message.role.value == "user" and i < len(conversation.messages) - 1:
                # Format conversation up to this point
                prompt = self.template.format_messages(current_messages)
                
                print(f"Turn {len(turn_metrics) + 1}: Generating response...")
                
                try:
                    # Generate response with metrics
                    generated_text, metrics = self.engine.generate_with_metrics(
                        prompt=prompt,
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    turn_metrics.append(metrics)
                    
                    print(f"  Generated {metrics.completion_tokens} tokens in {metrics.total_generation_time:.3f}s")
                    print(f"  Speed: {metrics.tokens_per_second:.1f} tok/s")
                    
                except Exception as e:
                    print(f"  ❌ Generation failed: {e}")
                    # Create dummy metrics for failed generation
                    turn_metrics.append(GenerationMetrics(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        time_to_first_token=0,
                        total_generation_time=0,
                        tokens_per_second=0,
                        memory_usage_mb=0,
                        engine_name="vllm",
                        model_name=os.path.basename(self.model_path)
                    ))
        
        total_time = time.time() - total_start_time
        
        # Calculate aggregate metrics
        total_tokens_generated = sum(m.completion_tokens for m in turn_metrics)
        avg_tokens_per_second = (
            sum(m.tokens_per_second for m in turn_metrics) / len(turn_metrics)
            if turn_metrics else 0
        )
        
        # Calculate cache effectiveness (if available)
        cache_hit_rates = [m.cache_hit_rate for m in turn_metrics if m.cache_hit_rate is not None]
        cache_effectiveness = sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else None
        
        return ConversationBenchmarkResult(
            conversation_name=conversation.name,
            total_turns=len(turn_metrics),
            total_time=total_time,
            turn_metrics=turn_metrics,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens_generated=total_tokens_generated,
            cache_effectiveness=cache_effectiveness,
            timestamp=datetime.now()
        )
    
    def run_complete_benchmark(self, conversations: Optional[List[Conversation]] = None) -> Dict[str, str]:
        """Run complete benchmark suite"""
        print("🚀 Starting Complete LLM Benchmark")
        print("=" * 60)
        
        # Collect initial system information
        print("📊 Collecting system information...")
        system_info = self.system_collector.get_complete_system_info()
        
        # Prepare conversations
        if conversations is None:
            conversations = self.prepare_conversations()
        
        print(f"📝 Prepared {len(conversations)} test conversations")
        
        # Start vLLM engine
        print("🔧 Starting vLLM engine...")
        self.engine = VLLMEngine(
            model_path=self.model_path,
            host=self.host,
            port=self.port,
            max_model_len=2048,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.8,
            max_num_batched_tokens=512,
            max_num_seqs=2,
            enforce_eager=True  # Avoid compilation cache issues
        )
        
        if not self.engine.start_server():
            raise RuntimeError("Failed to start vLLM server")
        
        try:
            # Warmup
            print("🔥 Warming up model...")
            self.engine.warmup()
            
            # Run benchmarks
            results = []
            for conversation in conversations:
                result = self.run_conversation_benchmark(conversation)
                results.append(result)
            
            # Collect final system metrics
            final_system_metrics = self.system_collector.get_runtime_metrics()
            
            # Prepare configuration info
            config_info = {
                "model_path": self.model_path,
                "server_host": self.host,
                "server_port": self.port,
                "max_model_len": 2048,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.8,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 2,
                "enforce_eager": True
            }
            
            # Combine system info
            combined_system_info = {
                **system_info,
                "final_metrics": final_system_metrics
            }
            
            # Save results in multiple formats
            print("\n💾 Saving benchmark results...")
            files_created = self.formatter.save_results(
                conversation_results=results,
                system_info=combined_system_info,
                config_info=config_info
            )
            
            print("\n✅ Benchmark completed successfully!")
            print("\nFiles created:")
            for format_type, path in files_created.items():
                print(f"  📄 {format_type}: {path}")
            
            return files_created
            
        finally:
            # Always stop the engine
            if self.engine:
                self.engine.stop_server()


def main():
    """Main benchmark execution"""
    # Check if model exists
    model_path = "./phi-3.5-mini-Q4_K.gguf"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please download the model first or update the path.")
        sys.exit(1)
    
    print("🎯 LLM Benchmark Suite")
    print("=" * 40)
    print(f"Model: {model_path}")
    print(f"Output: results/")
    
    # Create and run benchmark
    runner = BenchmarkRunner(model_path)
    
    try:
        files_created = runner.run_complete_benchmark()
        
        print(f"\n🎉 Benchmark complete! Check the results directory.")
        print(f"📊 Open the markdown file for a human-readable report.")
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()