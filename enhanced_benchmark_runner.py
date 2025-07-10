#!/usr/bin/env python3
"""
Enhanced benchmark runner with context-aware conversation selection and RAG simulation
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
from schemas.conversation import Conversation, Message, MessageRole, MessageType
from templates.phi3 import Phi3ChatTemplate
from engines.vllm_engine import VLLMEngine
from output.formatters import BenchmarkFormatter
from data.conversation_loader import ConversationLoader


class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with dynamic conversation generation and RAG simulation"""
    
    def __init__(self, 
                 model_path: str,
                 model_context_size: int = 128000,  # Phi-3.5 supports 128K context
                 host: str = "127.0.0.1",
                 port: int = 8001,
                 output_dir: str = "results"):
        
        self.model_path = model_path
        self.model_context_size = model_context_size
        self.host = host
        self.port = port
        
        # Initialize components
        self.system_collector = SystemMetricsCollector()
        self.formatter = BenchmarkFormatter(output_dir)
        self.template = Phi3ChatTemplate()
        self.conversation_loader = ConversationLoader()
        
        # Will be initialized when benchmark starts
        self.engine: Optional[VLLMEngine] = None
        
        print(f"🤖 Model context size: {self.model_context_size:,} tokens")
    
    def select_conversations(self, 
                           include_tags: Optional[List[str]] = None,
                           exclude_tags: Optional[List[str]] = None,
                           max_conversations: Optional[int] = None) -> List[Conversation]:
        """Select appropriate conversations based on context size and filters"""
        
        conversations = self.conversation_loader.get_test_suite(
            model_context_size=self.model_context_size,
            include_tags=include_tags,
            exclude_tags=exclude_tags
        )
        
        if max_conversations and len(conversations) > max_conversations:
            # Prioritize conversations by complexity/importance
            conversations = conversations[:max_conversations]
        
        print(f"📝 Selected {len(conversations)} conversations for benchmarking:")
        for conv in conversations:
            estimated_tokens = conv.estimate_total_tokens()
            tags_str = ", ".join(conv.tags)
            rag_indicator = "🎯" if any(msg.message_type == MessageType.RAG_DATA for msg in conv.messages) else ""
            print(f"   • {conv.name} ({estimated_tokens:,} tokens) [{tags_str}] {rag_indicator}")
        
        return conversations
    
    def run_conversation_benchmark(self, conversation: Conversation) -> ConversationBenchmarkResult:
        """Run benchmark for a single conversation with enhanced turn handling"""
        print(f"\\n🔄 Benchmarking: {conversation.name}")
        print(f"📖 Description: {conversation.description}")
        print("-" * 70)
        
        turn_metrics = []
        total_start_time = time.time()
        
        # Track conversation state for multi-turn processing
        conversation_history = []
        rag_data_active = False
        
        # Start with any initial system/RAG messages
        for msg in conversation.messages:
            if msg.role == MessageRole.SYSTEM:
                conversation_history.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
                if msg.message_type == MessageType.RAG_DATA:
                    rag_data_active = True
                    print("   📋 Initial RAG data loaded into conversation context")
        
        # Process messages and generate responses dynamically
        user_messages = [msg for msg in conversation.messages if msg.role == MessageRole.USER]
        
        for turn_idx, user_message in enumerate(user_messages):
            turn_number = turn_idx + 1
            print(f"\\n🔵 Turn {turn_number}/{len(user_messages)}")
            
            # Handle RAG data removal before processing this message
            if (user_message.message_type == MessageType.RAG_REMOVAL and 
                user_message.message_metadata.get("remove_rag_before_this") and 
                rag_data_active):
                
                # Remove RAG data from history
                conversation_history = [
                    msg for msg in conversation_history 
                    if not msg["content"].startswith("[RETRIEVED INFORMATION]")
                ]
                rag_data_active = False
                print("   🗑️ RAG data removed from conversation context")
            
            # Add current user message to history
            conversation_history.append({
                "role": user_message.role.value,
                "content": user_message.content
            })
            
            # Format the conversation for the model
            prompt = self.template.format_messages(conversation_history)
            
            # Estimate context usage
            estimated_prompt_tokens = len(prompt) // 4  # Rough estimate
            print(f"   📏 Estimated prompt tokens: {estimated_prompt_tokens:,}")
            
            # Check context limit
            if estimated_prompt_tokens > self.model_context_size * 0.8:  # 80% threshold
                print(f"   ⚠️ Warning: Approaching context limit ({estimated_prompt_tokens:,}/{self.model_context_size:,})")
            
            try:
                # Generate response with metrics
                print(f"   🤖 Generating response...")
                generated_text, metrics = self.engine.generate_with_metrics(
                    prompt=prompt,
                    max_tokens=min(500, self.model_context_size - estimated_prompt_tokens - 100),  # Adaptive max tokens
                    temperature=0.7
                )
                
                # Add assistant response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": generated_text.strip()
                })
                
                # Enhance metrics with turn-specific information
                metrics.prompt_tokens = estimated_prompt_tokens  # Update with our estimate
                
                # Add turn metadata to metrics
                if hasattr(metrics, 'turn_metadata'):
                    metrics.turn_metadata = {}
                else:
                    metrics.turn_metadata = {}
                
                metrics.turn_metadata.update({
                    "turn_number": turn_number,
                    "message_type": user_message.message_type.value,
                    "rag_active": rag_data_active,
                    "context_usage_percent": (estimated_prompt_tokens / self.model_context_size) * 100
                })
                
                turn_metrics.append(metrics)
                
                print(f"   ✅ Generated {metrics.completion_tokens} tokens in {metrics.total_generation_time:.3f}s")
                print(f"   ⚡ Speed: {metrics.tokens_per_second:.1f} tok/s")
                if metrics.cache_hit_rate is not None:
                    print(f"   💾 Cache hit rate: {metrics.cache_hit_rate:.2%}")
                
                # Brief preview of generated content
                preview = generated_text.strip()[:100] + "..." if len(generated_text.strip()) > 100 else generated_text.strip()
                print(f"   💬 Preview: {preview}")
                
            except Exception as e:
                print(f"   ❌ Generation failed: {e}")
                # Create dummy metrics for failed generation
                turn_metrics.append(GenerationMetrics(
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=0,
                    total_tokens=estimated_prompt_tokens,
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
        
        # Create enhanced result with RAG simulation metadata
        result = ConversationBenchmarkResult(
            conversation_name=conversation.name,
            total_turns=len(turn_metrics),
            total_time=total_time,
            turn_metrics=turn_metrics,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens_generated=total_tokens_generated,
            cache_effectiveness=cache_effectiveness,
            timestamp=datetime.now()
        )
        
        # Add enhanced metadata
        if hasattr(result, 'conversation_metadata'):
            result.conversation_metadata = {}
        else:
            result.conversation_metadata = {}
        
        result.conversation_metadata.update({
            "description": conversation.description,
            "tags": conversation.tags,
            "estimated_final_tokens": conversation.estimated_final_tokens,
            "actual_final_context": sum(len(msg["content"]) for msg in conversation_history) // 4,
            "rag_simulation": any(msg.message_type == MessageType.RAG_DATA for msg in conversation.messages),
            "max_context_usage": max((m.turn_metadata.get("context_usage_percent", 0) for m in turn_metrics), default=0)
        })
        
        print(f"\\n📊 Conversation Summary:")
        print(f"   🎯 Total turns: {len(turn_metrics)}")
        print(f"   ⏱️ Total time: {total_time:.3f}s")
        print(f"   📝 Tokens generated: {total_tokens_generated}")
        print(f"   ⚡ Avg speed: {avg_tokens_per_second:.1f} tok/s")
        if cache_effectiveness:
            print(f"   💾 Cache effectiveness: {cache_effectiveness:.2%}")
        
        return result
    
    def run_complete_benchmark(self, 
                             include_tags: Optional[List[str]] = None,
                             exclude_tags: Optional[List[str]] = None,
                             max_conversations: Optional[int] = None) -> Dict[str, str]:
        """Run complete enhanced benchmark suite"""
        print("🚀 Enhanced LLM Benchmark Suite")
        print("=" * 80)
        print(f"🤖 Model: {self.model_path}")
        print(f"📏 Context size: {self.model_context_size:,} tokens")
        print(f"📊 Output: results/")
        
        # Collect initial system information
        print("\\n📊 Collecting system information...")
        system_info = self.system_collector.get_complete_system_info()
        
        # Select appropriate conversations
        print("\\n📝 Selecting conversations...")
        conversations = self.select_conversations(
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            max_conversations=max_conversations
        )
        
        if not conversations:
            print("❌ No suitable conversations found for this context size")
            return {}
        
        # Start vLLM engine
        print("\\n🔧 Starting vLLM engine...")
        self.engine = VLLMEngine(
            model_path=self.model_path,
            host=self.host,
            port=self.port,
            max_model_len=min(self.model_context_size, 32768),  # Cap for stability
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
            for i, conversation in enumerate(conversations, 1):
                print(f"\\n{'='*80}")
                print(f"🎯 Running benchmark {i}/{len(conversations)}")
                result = self.run_conversation_benchmark(conversation)
                results.append(result)
            
            # Collect final system metrics
            final_system_metrics = self.system_collector.get_runtime_metrics()
            
            # Prepare enhanced configuration info
            config_info = {
                "model_path": self.model_path,
                "model_context_size": self.model_context_size,
                "server_host": self.host,
                "server_port": self.port,
                "actual_max_model_len": min(self.model_context_size, 32768),
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.8,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 2,
                "enforce_eager": True,
                "conversation_selection": {
                    "include_tags": include_tags,
                    "exclude_tags": exclude_tags,
                    "max_conversations": max_conversations,
                    "total_selected": len(conversations)
                }
            }
            
            # Combine system info
            combined_system_info = {
                **system_info,
                "final_metrics": final_system_metrics,
                "conversation_summary": {
                    "total_conversations": len(results),
                    "total_turns": sum(r.total_turns for r in results),
                    "total_tokens_generated": sum(r.total_tokens_generated for r in results),
                    "total_time": sum(r.total_time for r in results),
                    "rag_simulations": sum(1 for r in results if r.conversation_metadata.get("rag_simulation", False))
                }
            }
            
            # Save results in multiple formats
            print("\\n💾 Saving enhanced benchmark results...")
            files_created = self.formatter.save_results(
                conversation_results=results,
                system_info=combined_system_info,
                config_info=config_info
            )
            
            # Print summary
            print("\\n✅ Enhanced benchmark completed successfully!")
            print("\\n📈 Summary:")
            print(f"   🎯 Conversations: {len(results)}")
            print(f"   🔄 Total turns: {sum(r.total_turns for r in results)}")
            print(f"   📝 Tokens generated: {sum(r.total_tokens_generated for r in results):,}")
            print(f"   ⏱️ Total time: {sum(r.total_time for r in results):.1f}s")
            print(f"   ⚡ Avg speed: {sum(r.avg_tokens_per_second for r in results)/len(results):.1f} tok/s")
            
            rag_count = sum(1 for r in results if r.conversation_metadata.get("rag_simulation", False))
            if rag_count > 0:
                print(f"   🎯 RAG simulations: {rag_count}")
            
            print("\\nFiles created:")
            for format_type, path in files_created.items():
                print(f"   📄 {format_type}: {path}")
            
            return files_created
            
        finally:
            # Always stop the engine
            if self.engine:
                self.engine.stop_server()


def main():
    """Main enhanced benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LLM Benchmark Runner")
    parser.add_argument("--model", type=str, default="./phi-3.5-mini-Q4_K.gguf", help="Model path")
    parser.add_argument("--context-size", type=int, default=128000, help="Model context size")
    parser.add_argument("--include-tags", nargs="+", help="Include only conversations with these tags")
    parser.add_argument("--exclude-tags", nargs="+", help="Exclude conversations with these tags")
    parser.add_argument("--max-conversations", type=int, help="Maximum number of conversations to run")
    parser.add_argument("--list-conversations", action="store_true", help="List available conversations and exit")
    
    args = parser.parse_args()
    
    # Check if we should just list conversations
    if args.list_conversations:
        loader = ConversationLoader()
        loader.list_available_conversations(args.context_size)
        return
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please download the model first using:")
        print("  python download_models.py --setup")
        sys.exit(1)
    
    print("🎯 Enhanced LLM Benchmark Suite")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Context: {args.context_size:,} tokens")
    if args.include_tags:
        print(f"Include tags: {args.include_tags}")
    if args.exclude_tags:
        print(f"Exclude tags: {args.exclude_tags}")
    if args.max_conversations:
        print(f"Max conversations: {args.max_conversations}")
    
    # Create and run enhanced benchmark
    runner = EnhancedBenchmarkRunner(
        model_path=str(model_path),
        model_context_size=args.context_size
    )
    
    try:
        files_created = runner.run_complete_benchmark(
            include_tags=args.include_tags,
            exclude_tags=args.exclude_tags,
            max_conversations=args.max_conversations
        )
        
        print(f"\\n🎉 Enhanced benchmark complete!")
        print(f"📊 Open the markdown file for a detailed report.")
        
    except Exception as e:
        print(f"\\n❌ Enhanced benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()