#!/usr/bin/env python3
"""
HomeLLM Benchmark Runner - Connect to any OpenAI-compatible chat completions endpoint
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
from ..utils.exceptions import safe_execute
from ..config.constants import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, TOKEN_ESTIMATION_DIVISOR, DEFAULT_PORT, DEFAULT_CONTEXT_SIZE, VLLM_DEFAULT_PORT, OLLAMA_DEFAULT_PORT

# Additional constants for benchmark runner
CONTEXT_LIMIT_WARNING_THRESHOLD = 0.8  # Warn when approaching 80% of context limit
CONTEXT_SAFETY_MARGIN = 100  # Reserve 100 tokens for safety


class BenchmarkRunner:
    """Benchmark runner that connects to any OpenAI-compatible chat completions endpoint"""
    
    def __init__(self, 
                 host: str = "127.0.0.1",
                 port: int = DEFAULT_PORT,
                 context_size: int = DEFAULT_CONTEXT_SIZE,
                 engine_type: str = "vllm",
                 model_name: str = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
                 output_dir: str = "results",
                 enable_tts: bool = False,
                 tts_output_dir: str = "tts_output"):
        
        self.host = host
        self.port = port
        self.context_size = context_size
        self.engine_type = engine_type.lower()
        self.model_name = model_name
        self.enable_tts = enable_tts
        self.tts_output_dir = tts_output_dir
        
        # Initialize dependencies
        self.deps = BenchmarkDependencies(output_dir)
        
        # Initialize TTS coordinator if enabled
        self.tts_coordinator = None
        if enable_tts:
            from ..tts.streaming_tts_coordinator import StreamingTTSCoordinator
            self.tts_coordinator = StreamingTTSCoordinator(
                output_dir=str(Path(output_dir) / tts_output_dir),
                max_concurrent_tts=1,  # Single conversation at a time
                buffer_size=300,
                min_chunk_size=20,
                max_chunk_size=150
            )
        
        # Initialize engine client based on type
        if self.engine_type == "vllm":
            from ..engines.vllm_engine import VLLMEngine
            self.engine = VLLMEngine(model_name=model_name, host=host, port=port)
        elif self.engine_type == "ollama":
            from ..engines.ollama_engine import OllamaEngine
            self.engine = OllamaEngine(model_name=model_name, host=host, port=port)
        else:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        print(f"Benchmark runner initialized:")
        print(f"  Endpoint: {host}:{port}")
        print(f"  Engine: {engine_type}")
        print(f"  Model: {model_name}")
        print(f"  Context: {context_size:,} tokens")
        if enable_tts:
            print(f"  TTS: Enabled (output: {tts_output_dir})")
        else:
            print(f"  TTS: Disabled")
    
    def estimate_conversation_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Character-based token estimation for OpenAI format messages"""
        total_chars = sum(len(msg['content']) for msg in messages)
        return total_chars // TOKEN_ESTIMATION_DIVISOR
    
    def check_server_health(self) -> bool:
        """Check if the server is responding"""
        try:
            return self.engine.is_running()
        except Exception:
            return False
    
    def select_conversations(self, 
                           include_tags: Optional[List[str]] = None,
                           exclude_tags: Optional[List[str]] = None,
                           max_conversations: Optional[int] = None) -> List[Conversation]:
        """Select appropriate conversations based on context size and filters"""
        
        conversations = self.deps.conversation_loader.get_test_suite(
            model_context_size=self.context_size,
            include_tags=include_tags,
            exclude_tags=exclude_tags
        )
        
        if max_conversations and len(conversations) > max_conversations:
            conversations = conversations[:max_conversations]
        
        print(f"Selected {len(conversations)} conversations:")
        for conv in conversations:
            estimated_tokens = conv.estimate_total_tokens()
            tags_str = ", ".join(conv.tags)
            rag_indicator = "[RAG]" if any(msg.message_type == MessageType.RAG_DATA for msg in conv.messages) else ""
            print(f"  - {conv.name} (~{estimated_tokens:,} tokens) [{tags_str}] {rag_indicator}")
        
        return conversations
    
    async def generate_with_streaming_tts(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float):
        """Generate text with streaming TTS enabled"""
        
        # Initialize TTS coordinator if not already done
        if not self.tts_coordinator.is_initialized:
            await self.tts_coordinator.initialize()
        
        # Create async generator for streaming tokens
        async def token_stream():
            # Get the streaming generator from the engine
            stream_generator = self.engine.generate_chat_streaming(messages, max_tokens, temperature)
            generated_tokens = []
            
            for item in stream_generator:
                if isinstance(item, str):
                    generated_tokens.append(item)
                    yield item
                else:
                    # This is the final metrics object
                    self._last_generation_metrics = item
                    self._generated_text = "".join(generated_tokens)
                    break
        
        # Process through TTS coordinator
        result = await self.tts_coordinator.process_token_stream(token_stream())
        
        # Get final text and metrics
        final_text = getattr(self, '_generated_text', "")
        base_metrics = getattr(self, '_last_generation_metrics', None)
        
        # Create enhanced metrics with TTS information
        if base_metrics:
            # Create a new metrics object with TTS metadata
            tts_metadata = {
                "audio_files_created": len(result.audio_files),
                "tts_processing_time": result.total_processing_time,
                "audio_chunks": result.successful_chunks,
                "tts_rtf": result.metrics.avg_audio_rtf
            }
            
            # Create new metrics object with TTS metadata
            metrics = GenerationMetrics(
                prompt_tokens=base_metrics.prompt_tokens,
                completion_tokens=base_metrics.completion_tokens,
                total_tokens=base_metrics.total_tokens,
                time_to_first_token=base_metrics.time_to_first_token,
                total_generation_time=base_metrics.total_generation_time,
                tokens_per_second=base_metrics.tokens_per_second,
                engine_metrics=base_metrics.engine_metrics,
                timestamp=base_metrics.timestamp,
                engine_name=base_metrics.engine_name,
                model_name=base_metrics.model_name,
                turn_metadata=base_metrics.turn_metadata,
                tts_metadata=tts_metadata
            )
        else:
            # Create fallback metrics with TTS metadata
            tts_metadata = {
                "audio_files_created": len(result.audio_files),
                "tts_processing_time": result.total_processing_time,
                "audio_chunks": result.successful_chunks,
                "tts_rtf": result.metrics.avg_audio_rtf
            }
            
            metrics = GenerationMetrics(
                prompt_tokens=len(" ".join(msg['content'] for msg in messages)) // 4,
                completion_tokens=result.metrics.total_tokens_processed,
                total_tokens=result.metrics.total_tokens_processed,
                time_to_first_token=0.1,
                total_generation_time=result.total_processing_time,
                tokens_per_second=result.metrics.avg_tokens_per_second,
                engine_name=self.engine_type,
                model_name=self.model_name,
                tts_metadata=tts_metadata
            )
        
        return final_text, metrics, result

    async def run_conversation_benchmark(self, conversation: Conversation) -> ConversationBenchmarkResult:
        """Run benchmark for a single conversation"""
        print(f"\\nRunning: {conversation.name}")
        print(f"Description: {conversation.description}")
        print("-" * 60)
        
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
                    print("  RAG data loaded")
        
        # Process user messages and generate responses
        user_messages = [msg for msg in conversation.messages if msg.role == MessageRole.USER]
        
        for turn_idx, user_message in enumerate(user_messages):
            turn_number = turn_idx + 1
            print(f"\\nTurn {turn_number}/{len(user_messages)}")
            
            # Handle RAG data removal
            if (user_message.message_type == MessageType.RAG_REMOVAL and 
                user_message.message_metadata.get("remove_rag_before_this") and 
                rag_data_active):
                
                conversation_history = [
                    msg for msg in conversation_history 
                    if not msg["content"].startswith("[RETRIEVED INFORMATION]")
                ]
                rag_data_active = False
                print("  RAG data removed")
            
            # Add current user message
            conversation_history.append({
                "role": user_message.role.value,
                "content": user_message.content
            })
            
            # Estimate context usage
            estimated_tokens = self.estimate_conversation_tokens(conversation_history)
            print(f"  Context: ~{estimated_tokens:,} tokens")
            
            # Check context limit
            if estimated_tokens > self.context_size * CONTEXT_LIMIT_WARNING_THRESHOLD:
                print(f"  Warning: Approaching context limit ({estimated_tokens:,}/{self.context_size:,})")
            
            try:
                # Generate response
                print(f"  Generating...")
                
                if self.enable_tts:
                    # Use streaming TTS generation
                    print(f"  TTS enabled - streaming generation...")
                    generated_text, metrics, tts_result = await self.generate_with_streaming_tts(
                        messages=conversation_history,
                        max_tokens=min(DEFAULT_MAX_TOKENS, self.context_size - estimated_tokens - CONTEXT_SAFETY_MARGIN),
                        temperature=DEFAULT_TEMPERATURE
                    )
                    
                    # Print TTS info
                    print(f"  TTS: {len(tts_result.audio_files)} audio files created")
                    print(f"  TTS RTF: {tts_result.metrics.avg_audio_rtf:.4f}")
                    
                else:
                    # Use regular generation
                    generated_text, metrics = self.engine.generate_chat_with_metrics(
                        messages=conversation_history,
                        max_tokens=min(DEFAULT_MAX_TOKENS, self.context_size - estimated_tokens - CONTEXT_SAFETY_MARGIN),
                        temperature=DEFAULT_TEMPERATURE
                    )
                
                # Add response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": generated_text.strip()
                })
                
                # Add turn metadata
                metrics.turn_metadata = {
                    "turn_number": turn_number,
                    "message_type": user_message.message_type.value,
                    "rag_active": rag_data_active,
                    "context_usage_percent": (estimated_tokens / self.context_size) * 100
                }
                
                turn_metrics.append(metrics)
                
                print(f"  Generated {metrics.completion_tokens} tokens in {metrics.total_generation_time:.2f}s")
                print(f"  Speed: {metrics.tokens_per_second:.1f} tok/s")
                
            except Exception as e:
                print(f"  Error: {e}")
                # Create dummy metrics for failed generation
                turn_metrics.append(GenerationMetrics(
                    prompt_tokens=estimated_tokens,
                    completion_tokens=0,
                    total_tokens=estimated_tokens,
                    time_to_first_token=0,
                    total_generation_time=0,
                    tokens_per_second=0,
                    engine_name=self.engine_type,
                    model_name="unknown"
                ))
        
        total_time = time.time() - total_start_time
        
        # Calculate summary metrics
        total_tokens_generated = sum(m.completion_tokens for m in turn_metrics)
        avg_tokens_per_second = (
            sum(m.tokens_per_second for m in turn_metrics) / len(turn_metrics)
            if turn_metrics else 0
        )
        
        result = ConversationBenchmarkResult(
            conversation_name=conversation.name,
            total_turns=len(turn_metrics),
            total_time=total_time,
            turn_metrics=turn_metrics,
            avg_tokens_per_second=avg_tokens_per_second,
            total_tokens_generated=total_tokens_generated,
            cache_effectiveness=None,
            timestamp=datetime.now()
        )
        
        # Add metadata
        result.conversation_metadata = {
            "description": conversation.description,
            "tags": conversation.tags,
            "rag_simulation": any(msg.message_type == MessageType.RAG_DATA for msg in conversation.messages),
            "final_context_tokens": self.estimate_conversation_tokens(conversation_history)
        }
        
        print(f"\\nCompleted: {len(turn_metrics)} turns, {total_tokens_generated} tokens, {avg_tokens_per_second:.1f} tok/s")
        
        return result
    
    async def run_benchmark(self, 
                     include_tags: Optional[List[str]] = None,
                     exclude_tags: Optional[List[str]] = None,
                     max_conversations: Optional[int] = None) -> Dict[str, str]:
        """Run the complete benchmark suite"""
        
        print("HomeLLM Benchmark Suite")
        print("=" * 50)
        
        # Check server health
        if not self.check_server_health():
            print(f"Error: No server responding at {self.host}:{self.port}")
            print("Start your server first:")
            print(f"  vllm serve <model> --host {self.host} --port {self.port}")
            return {}
        
        print(f"Server healthy at {self.host}:{self.port}")
        
        # Collect system info
        system_info = self.deps.system_collector.get_complete_system_info()
        
        # Select conversations
        conversations = self.select_conversations(
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            max_conversations=max_conversations
        )
        
        if not conversations:
            print("No conversations selected")
            return {}
        
        # Run benchmarks
        results = []
        for i, conversation in enumerate(conversations, 1):
            print(f"\\n{'='*60}")
            print(f"Benchmark {i}/{len(conversations)}")
            result = await self.run_conversation_benchmark(conversation)
            results.append(result)
        
        # Save results
        config_info = {
            "endpoint": f"{self.host}:{self.port}",
            "engine_type": self.engine_type,
            "context_size": self.context_size,
            "conversation_selection": {
                "include_tags": include_tags,
                "exclude_tags": exclude_tags,
                "max_conversations": max_conversations,
                "total_selected": len(conversations)
            }
        }
        
        print("\\nSaving results...")
        files_created = self.deps.formatter.save_results(
            conversation_results=results,
            system_info=system_info,
            config_info=config_info
        )
        
        # Print summary
        total_tokens = sum(r.total_tokens_generated for r in results)
        total_time = sum(r.total_time for r in results)
        avg_speed = sum(r.avg_tokens_per_second for r in results) / len(results)
        
        print(f"\\nBenchmark Complete!")
        print(f"  Conversations: {len(results)}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average speed: {avg_speed:.1f} tok/s")
        
        print(f"\\nFiles created:")
        for format_type, path in files_created.items():
            print(f"  {format_type}: {path}")
        
        # Cleanup TTS coordinator if enabled
        if self.enable_tts and self.tts_coordinator:
            await self.tts_coordinator.shutdown()
            print(f"TTS coordinator shut down")
        
        return files_created


async def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="HomeLLM Benchmark Runner - Connect to any OpenAI-compatible chat completions endpoint"
    )
    
    # Server connection
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, help="Server port (default: engine-specific)")
    parser.add_argument("--engine", choices=["vllm", "ollama"], default="vllm", 
                       help="Engine type for metrics collection (default: vllm, port 8000; ollama uses port 11434)")
    parser.add_argument("--model", help="Model name for API calls (default: engine-specific)")
    
    # Benchmark configuration
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE, 
                       help=f"Context size in tokens (default: {DEFAULT_CONTEXT_SIZE})")
    parser.add_argument("--max-conversations", type=int, 
                       help="Maximum number of conversations to run")
    
    # Conversation filtering
    parser.add_argument("--include-tags", nargs="+", 
                       help="Include only conversations with these tags")
    parser.add_argument("--exclude-tags", nargs="+", 
                       help="Exclude conversations with these tags")
    
    # TTS options
    parser.add_argument("--enable-tts", action="store_true", 
                       help="Enable streaming TTS generation with F5-TTS")
    parser.add_argument("--tts-output-dir", default="tts_output", 
                       help="Output directory for TTS audio files (default: tts_output)")
    
    # Utilities
    parser.add_argument("--list-conversations", action="store_true", 
                       help="List available conversations and exit")
    
    args = parser.parse_args()
    
    # Set engine-specific defaults
    if args.port is None:
        args.port = OLLAMA_DEFAULT_PORT if args.engine == "ollama" else VLLM_DEFAULT_PORT
    
    if args.model is None:
        args.model = "llama3.2:3b" if args.engine == "ollama" else "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
    
    # List conversations if requested
    if args.list_conversations:
        deps = BenchmarkDependencies()
        deps.conversation_loader.list_available_conversations(args.context_size)
        return
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        host=args.host,
        port=args.port,
        context_size=args.context_size,
        engine_type=args.engine,
        model_name=args.model,
        enable_tts=args.enable_tts,
        tts_output_dir=args.tts_output_dir
    )
    
    # Run benchmark with async support
    import asyncio
    files_created = await runner.run_benchmark(
        include_tags=args.include_tags,
        exclude_tags=args.exclude_tags,
        max_conversations=args.max_conversations
    )
    
    if files_created:
        print(f"\\nOpen the markdown file for detailed results.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())