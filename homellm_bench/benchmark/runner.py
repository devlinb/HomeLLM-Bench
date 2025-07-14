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

from ..metrics.schemas import GenerationMetrics, ConversationBenchmarkResult
from ..schemas.conversation import Conversation, Message, MessageRole, MessageType
from ..utils.benchmark_dependencies import BenchmarkDependencies
from ..utils.exceptions import safe_execute, handle_server_connection_error, handle_model_config_error
from ..utils.server_detection import detect_vllm_server, wait_for_server_ready, get_vllm_process_info
from ..utils.compilation_cache import get_compilation_cache
from ..utils.vllm_server_manager import VLLMServerManager
from ..config.vllm_config import DEFAULT_VLLM_PORT
from ..config.constants import (
    CONTEXT_WARNING_THRESHOLD, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    TOKEN_ESTIMATION_DIVISOR
)


class BenchmarkRunner:
    """Enhanced benchmark runner with dynamic conversation generation and RAG simulation"""
    
    def __init__(self, 
                 model_name: str,
                 model_context_size: Optional[int] = None,
                 host: str = "127.0.0.1",
                 port: int = DEFAULT_VLLM_PORT,
                 output_dir: str = "results",
                 deps: Optional[BenchmarkDependencies] = None):
        
        self.model_name = model_name
        self.host = host
        self.port = port
        
        # Initialize dependencies
        self.deps = deps or BenchmarkDependencies(output_dir)
        
        # Get model configuration
        self.model_config = self.deps.model_registry.get_model_config(model_name)
        if not self.model_config:
            handle_model_config_error(model_name)
        
        # Set context size from config or parameter
        self.model_context_size = model_context_size or self.model_config.context_size
        
        # Initialize compilation cache (uses default ~/.cache/vllm/torch_compile_cache)
        self.compilation_cache = get_compilation_cache()
        
        # Get appropriate template for model
        self.template = self.deps.get_template(self.model_config.chat_template)
        if not self.template:
            print(f"Warning: Template '{self.model_config.chat_template}' not found, falling back to phi3")
            self.template = self.deps.get_template("phi3")
        
        # Initialize engine client (connects to external server)
        self.engine = self.deps.create_engine(model_name=model_name, host=host, port=port)
        
        print(f"Model: {model_name}")
        print(f"Model type: {self.model_config.model_type.value}")
        print(f"Chat template: {self.model_config.chat_template}")
        print(f"Context size: {self.model_context_size:,} tokens")
        print(f"Connecting to vLLM server at {host}:{port}")
    
    def select_conversations(self, 
                           include_tags: Optional[List[str]] = None,
                           exclude_tags: Optional[List[str]] = None,
                           max_conversations: Optional[int] = None) -> List[Conversation]:
        """Select appropriate conversations based on context size and filters"""
        
        conversations = self.deps.conversation_loader.get_test_suite(
            model_context_size=self.model_context_size,
            include_tags=include_tags,
            exclude_tags=exclude_tags
        )
        
        if max_conversations and len(conversations) > max_conversations:
            # Prioritize conversations by complexity/importance
            conversations = conversations[:max_conversations]
        
        print(f"Selected {len(conversations)} conversations for benchmarking:")
        for conv in conversations:
            estimated_tokens = conv.estimate_total_tokens()
            tags_str = ", ".join(conv.tags)
            rag_indicator = "[RAG]" if any(msg.message_type == MessageType.RAG_DATA for msg in conv.messages) else ""
            print(f"   - {conv.name} ({estimated_tokens:,} tokens) [{tags_str}] {rag_indicator}")
        
        return conversations
    
    def run_conversation_benchmark(self, conversation: Conversation) -> ConversationBenchmarkResult:
        """Run benchmark for a single conversation with enhanced turn handling"""
        print(f"\\nBenchmarking: {conversation.name}")
        print(f"Description: {conversation.description}")
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
                    print("   Initial RAG data loaded into conversation context")
        
        # Process messages and generate responses dynamically
        user_messages = [msg for msg in conversation.messages if msg.role == MessageRole.USER]
        
        for turn_idx, user_message in enumerate(user_messages):
            turn_number = turn_idx + 1
            print(f"\\nTurn {turn_number}/{len(user_messages)}")
            
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
                print("   Removed: RAG data removed from conversation context")
            
            # Add current user message to history
            conversation_history.append({
                "role": user_message.role.value,
                "content": user_message.content
            })
            
            # Format the conversation for the model
            prompt = self.template.format_messages(conversation_history)
            
            # Estimate context usage
            estimated_prompt_tokens = len(prompt) // 4  # Rough estimate
            print(f"   Estimated prompt tokens: {estimated_prompt_tokens:,}")
            
            # Check context limit
            if estimated_prompt_tokens > self.model_context_size * 0.8:  # 80% threshold
                print(f"   Warning: Approaching context limit ({estimated_prompt_tokens:,}/{self.model_context_size:,})")
            
            try:
                # Generate response with metrics
                print(f"   Generating response...")
                generated_text, metrics = self.engine.generate_chat_with_metrics(
                    messages=conversation_history,
                    max_tokens=min(DEFAULT_MAX_TOKENS, self.model_context_size - estimated_prompt_tokens - 100),  # Adaptive max tokens
                    temperature=DEFAULT_TEMPERATURE
                )
                
                # Add assistant response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": generated_text.strip()
                })
                
                # Enhance metrics with turn-specific information
                metrics.prompt_tokens = estimated_prompt_tokens  # Update with our estimate
                
                # Add turn metadata to metrics
                metrics.turn_metadata = {}
                
                metrics.turn_metadata = {
                    "turn_number": turn_number,
                    "message_type": user_message.message_type.value,
                    "rag_active": rag_data_active,
                    "context_usage_percent": (estimated_prompt_tokens / self.model_context_size) * 100
                }
                
                turn_metrics.append(metrics)
                
                print(f"   Generated {metrics.completion_tokens} tokens in {metrics.total_generation_time:.3f}s")
                print(f"   Speed: {metrics.tokens_per_second:.1f} tok/s")
                
                # Brief preview of generated content
                preview = generated_text.strip()[:100] + "..." if len(generated_text.strip()) > 100 else generated_text.strip()
                print(f"   Preview: {preview}")
                
            except Exception as e:
                print(f"   Failed: Generation failed: {e}")
                # Create dummy metrics for failed generation
                turn_metrics.append(GenerationMetrics(
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=0,
                    total_tokens=estimated_prompt_tokens,
                    time_to_first_token=0,
                    total_generation_time=0,
                    tokens_per_second=0,
                    engine_name="vllm",
                    model_name=os.path.basename(self.model_name)
                ))
        
        total_time = time.time() - total_start_time
        
        # Calculate aggregate metrics
        total_tokens_generated = sum(m.completion_tokens for m in turn_metrics)
        avg_tokens_per_second = (
            sum(m.tokens_per_second for m in turn_metrics) / len(turn_metrics)
            if turn_metrics else 0
        )
        
        # Calculate cache effectiveness as timing delta between first and second message
        cache_effectiveness = None
        if len(turn_metrics) >= 2:
            first_ttft = turn_metrics[0].time_to_first_token
            second_ttft = turn_metrics[1].time_to_first_token
            if first_ttft is not None and second_ttft is not None and first_ttft > 0:
                cache_effectiveness = first_ttft - second_ttft
        
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
        result.conversation_metadata = {
            "description": conversation.description,
            "tags": conversation.tags,
            "estimated_final_tokens": conversation.estimated_final_tokens,
            "actual_final_context": sum(len(msg["content"]) for msg in conversation_history) // 4,
            "rag_simulation": any(msg.message_type == MessageType.RAG_DATA for msg in conversation.messages),
            "max_context_usage": max((m.turn_metadata.get("context_usage_percent", 0) for m in turn_metrics), default=0)
        }
        
        print(f"\\nSummary:")
        print(f"   Total turns: {len(turn_metrics)}")
        print(f"   Time: {total_time:.3f}s")
        print(f"   Tokens generated: {total_tokens_generated}")
        print(f"   Speed: {avg_tokens_per_second:.1f} tok/s")
        if cache_effectiveness:
            print(f"   Cache effectiveness: {cache_effectiveness:.3f}s")
        
        return result
    
    def _get_server_config_dict(self) -> Dict[str, Any]:
        """Get server configuration dictionary for cache hash generation."""
        return {
            'gpu_memory_utilization': 0.6,  # Common default
            'max_model_len': 32768,  # Common default
            'quantization': 'gptq_marlin' if 'gptq' in self.model_name.lower() else None,
            'enable_prefix_caching': True,
            'dtype': 'auto',
            'tensor_parallel_size': 1,
            'enforce_eager': False,
        }
    
    def check_compilation_status(self) -> Dict[str, Any]:
        """Check detailed compilation status for the current model."""
        config_dict = self._get_server_config_dict()
        return self.compilation_cache.get_cache_info(self.model_name, config_dict)
    
    def is_compiled(self) -> bool:
        """Check if current model configuration is compiled."""
        config_dict = self._get_server_config_dict()
        return self.compilation_cache.is_compiled(self.model_name, config_dict)
    
    def start_vllm_server(self, auto_start: bool = False) -> bool:
        """Start vLLM server with proper monitoring and error handling.
        
        Args:
            auto_start: If True, start server automatically. If False, ask user.
            
        Returns:
            True if server is ready, False otherwise
        """
        # Check if server is already running
        status, pid, message = detect_vllm_server(self.port, self.host)
        if status == "ready":
            print(f"Server already running on {self.host}:{self.port}")
            return True
        
        if not auto_start:
            print(f"Start vLLM server? (y/n)")
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                return False
        
        # Start server using the new manager
        print(f"Starting vLLM server for {self.model_name}...")
        print(f"Server will be available at http://{self.host}:{self.port}")
        
        config_dict = self._get_server_config_dict()
        
        try:
            manager = VLLMServerManager(self.host, self.port)
            result = manager.start_server(
                model_path=self.model_name,
                gpu_memory_utilization=config_dict.get('gpu_memory_utilization', 0.6),
                max_model_len=config_dict.get('max_model_len', 32768),
                enable_prefix_caching=config_dict.get('enable_prefix_caching', True),
                enforce_eager=config_dict.get('enforce_eager', False),
                disable_log_stats=True,
                disable_log_requests=True,
            )
            
            if result.success:
                print(f"Server started successfully in {result.startup_time:.1f}s")
                server_info = manager.get_server_info()
                print(f"Model: {server_info.get('model', 'Unknown')}")
                print(f"Process ID: {server_info.get('process_id')}")
                return True
            else:
                print(f"Failed to start server: {result.error_message}")
                return False
                
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def run_complete_benchmark(self, 
                             include_tags: Optional[List[str]] = None,
                             exclude_tags: Optional[List[str]] = None,
                             max_conversations: Optional[int] = None) -> Dict[str, str]:
        """Run complete enhanced benchmark suite"""
        print("Enhanced LLM Benchmark Suite")
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Context size: {self.model_context_size:,} tokens")
        print(f"Output: results/")
        
        # Collect initial system information
        print("\\nCollecting system information...")
        system_info = self.deps.system_collector.get_complete_system_info()
        
        # Select appropriate conversations
        print("\\nSelecting conversations...")
        conversations = self.select_conversations(
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            max_conversations=max_conversations
        )
        
        if not conversations:
            print("Failed: No suitable conversations found for this context size")
            return {}
        
        # Check compilation status
        print("\\nChecking compilation status...")
        config_dict = self._get_server_config_dict()
        if self.compilation_cache.is_compiled(self.model_name, config_dict):
            print(f"Found compiled cache for {self.model_name}")
        else:
            print(f"No compiled cache found for {self.model_name}")
            print("Note: First run will include compilation time")
        
        # Check vLLM server is running
        print("\\nChecking vLLM server...")
        status, pid, message = detect_vllm_server(self.port, self.host)
        
        if status == "not_running":
            print(f"No server running on {self.host}:{self.port}")
            auto_start = getattr(self, 'auto_start_server', False)
            if not self.start_vllm_server(auto_start=auto_start):
                print("Failed: Cannot run benchmark without vLLM server")
                return {}
        
        elif status == "starting_up":
            print(f"STARTING: {message}")
            print(f"Waiting for server to become ready...")
            if not wait_for_server_ready(self.port, self.host, max_wait=120):
                print("Failed: Server did not become ready within 2 minutes")
                return {}
        
        elif status == "port_busy":
            print(f"ERROR: {message}")
            print(f"Stop the other service or use a different port with --server-port")
            return {}
        
        elif status == "ready":
            print(f"READY: {message}")
            # Show server info
            if pid:
                info = get_vllm_process_info(pid)
                if info:
                    print(f"   Model: {info['model']}")
                    print(f"   Memory: {info['memory_mb']:.1f}MB")
        
        print(f"Server ready on {self.host}:{self.port}")
        
        # Warmup
        print("Warming up model...")
        if not self.engine.warmup():
            print("Warning: Warmup failed, but continuing...")
        
        # Run benchmarks
        results = []
        for i, conversation in enumerate(conversations, 1):
            print(f"\\n{'='*80}")
            print(f"Running benchmark {i}/{len(conversations)}")
            result = self.run_conversation_benchmark(conversation)
            results.append(result)
        
        # Collect final system metrics
        final_system_metrics = self.deps.system_collector.get_runtime_metrics()
        
        # Prepare enhanced configuration info
        config_info = {
            "model_name": self.model_name,
            "model_context_size": self.model_context_size,
            "server_host": self.host,
            "server_port": self.port,
            "server_type": "external_vllm",
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
        print("\\nSaving enhanced benchmark results...")
        files_created = self.deps.formatter.save_results(
            conversation_results=results,
            system_info=combined_system_info,
            config_info=config_info
        )
        
        # Print summary
        print("\\nComplete: Enhanced benchmark completed successfully!")
        print("\\nSummary:")
        print(f"   Conversations: {len(results)}")
        print(f"   Total turns: {sum(r.total_turns for r in results)}")
        print(f"   Tokens generated: {sum(r.total_tokens_generated for r in results):,}")
        print(f"   Time: {sum(r.total_time for r in results):.1f}s")
        print(f"   Speed: {sum(r.avg_tokens_per_second for r in results)/len(results):.1f} tok/s")
        
        rag_count = sum(1 for r in results if r.conversation_metadata.get("rag_simulation", False))
        if rag_count > 0:
            print(f"   RAG simulations: {rag_count}")
        
        print("\\nFiles created:")
        for format_type, path in files_created.items():
            print(f"   File {format_type}: {path}")
            
        return files_created


def main():
    """Main enhanced benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LLM Benchmark Runner")
    parser.add_argument("--model", type=str, default="./phi-3.5-mini-Q4_K.gguf", help="Model name (same as used to start vLLM server)")
    parser.add_argument("--context-size", type=int, default=128000, help="Model context size")
    parser.add_argument("--include-tags", nargs="+", help="Include only conversations with these tags")
    parser.add_argument("--exclude-tags", nargs="+", help="Exclude conversations with these tags")
    parser.add_argument("--max-conversations", type=int, help="Maximum number of conversations to run")
    parser.add_argument("--list-conversations", action="store_true", help="List available conversations and exit")
    parser.add_argument("--server-port", type=int, default=DEFAULT_VLLM_PORT, help=f"Port of vLLM server (default: {DEFAULT_VLLM_PORT})")
    parser.add_argument("--gpu-runtime", 
                       choices=["cuda", "rocm", "xpu"],
                       default="cuda",
                       help="GPU runtime to use (only cuda supported currently)")
    parser.add_argument("--check-compilation", action="store_true", 
                       help="Check compilation status and exit")
    parser.add_argument("--compilation-info", action="store_true", 
                       help="Show detailed compilation cache information")
    parser.add_argument("--auto-start-server", action="store_true",
                       help="Automatically start vLLM server if not running")
    
    args = parser.parse_args()
    
    # Check if we should just list conversations
    if args.list_conversations:
        deps = BenchmarkDependencies()
        deps.conversation_loader.list_available_conversations(args.context_size)
        return
    
    # Check compilation status if requested
    if args.check_compilation or args.compilation_info:
        runner = BenchmarkRunner(args.model, args.context_size, port=args.server_port)
        
        if args.check_compilation:
            if runner.is_compiled():
                print(f"Model {args.model} is compiled")
            else:
                print(f"Model {args.model} is not compiled")
            return
        
        if args.compilation_info:
            info = runner.check_compilation_status()
            print(f"Compilation Info for {args.model}:")
            print(f"  Cache Hash: {info['cache_hash']}")
            print(f"  Cache Path: {info['cache_path']}")
            print(f"  Cache Exists: {info['exists']}")
            print(f"  Is Compiled: {info['is_compiled']}")
            if info['files']:
                print(f"  Cache Files:")
                for file_info in info['files']:
                    print(f"    {file_info['name']}: {file_info['size_mb']:.1f}MB")
            return
    
    # Initialize system info
    from ..utils.system_info import initialize_system, GPURuntime
    initialize_system(GPURuntime(args.gpu_runtime))
    
    # Note: We don't check if model file exists since server handles the model
    model_name = args.model
    
    print("Enhanced LLM Benchmark Suite")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Context: {args.context_size:,} tokens")
    if args.include_tags:
        print(f"Include tags: {args.include_tags}")
    if args.exclude_tags:
        print(f"Exclude tags: {args.exclude_tags}")
    if args.max_conversations:
        print(f"Max conversations: {args.max_conversations}")
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        model_name=model_name,
        model_context_size=args.context_size,
        port=args.server_port
    )
    
    # Set auto-start server flag
    runner.auto_start_server = args.auto_start_server
    
    files_created = safe_execute(
        runner.run_complete_benchmark,
        "benchmark execution",
        include_tags=args.include_tags,
        exclude_tags=args.exclude_tags,
        max_conversations=args.max_conversations
    )
    
    print(f"\\nComplete: Enhanced benchmark complete!")
    print(f"Open the markdown file for a detailed report.")


if __name__ == "__main__":
    main()