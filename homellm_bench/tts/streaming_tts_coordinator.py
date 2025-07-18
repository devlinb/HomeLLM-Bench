#!/usr/bin/env python3
"""
Streaming TTS Coordinator

This module coordinates the streaming text buffer with the async TTS service
to provide end-to-end streaming TTS functionality. It handles the flow from
LLM streaming tokens to final audio output files.
"""

import asyncio
import time
import logging
import tempfile
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading

try:
    from .streaming_buffer import StreamingBuffer, BufferChunk
    from .async_tts_service import AsyncTTSService, TTSResult
except ImportError:
    # For standalone testing
    from streaming_buffer import StreamingBuffer, BufferChunk
    from async_tts_service import AsyncTTSService, TTSResult


# Constants
DEFAULT_COORDINATOR_BUFFER_SIZE = 500
DEFAULT_COORDINATOR_TIMEOUT = 1.5
DEFAULT_MIN_CHUNK_SIZE = 20
DEFAULT_MAX_CHUNK_SIZE = 150
DEFAULT_OUTPUT_DIR = str(Path(tempfile.gettempdir()) / "streaming_tts_output")
DEFAULT_AUDIO_PREFIX = "stream_chunk"
COORDINATOR_SHUTDOWN_TIMEOUT = 10.0


class CoordinatorState(Enum):
    """States of the streaming TTS coordinator"""
    IDLE = "idle"
    READY = "ready"
    STREAMING = "streaming"
    FINISHING = "finishing"
    STOPPED = "stopped"


@dataclass
class StreamingTTSMetrics:
    """Metrics for streaming TTS performance"""
    total_tokens_processed: int = 0
    total_chunks_generated: int = 0
    total_audio_files_created: int = 0
    total_streaming_time: float = 0.0
    total_tts_time: float = 0.0
    total_audio_duration: float = 0.0
    avg_chunk_processing_time: float = 0.0
    avg_tts_generation_time: float = 0.0
    avg_tokens_per_second: float = 0.0
    avg_audio_rtf: float = 0.0
    buffer_metrics: Optional[Dict[str, Any]] = None
    tts_metrics: Optional[Dict[str, Any]] = None
    
    def update_from_tts_result(self, result: TTSResult):
        """Update metrics from TTS result"""
        if result.success and result.metrics:
            self.total_audio_files_created += 1
            self.total_tts_time += result.processing_time
            self.total_audio_duration += result.metrics.get("audio_duration", 0.0)
            
            # Calculate averages with zero-division protection
            if self.total_audio_files_created > 0:
                self.avg_tts_generation_time = self.total_tts_time / self.total_audio_files_created
                # Use epsilon to prevent division by zero
                audio_duration_safe = max(self.total_audio_duration, 1e-10)
                self.avg_audio_rtf = self.total_tts_time / audio_duration_safe


@dataclass
class StreamingTTSOutput:
    """Output from streaming TTS processing"""
    audio_files: List[str] = field(default_factory=list)
    metrics: StreamingTTSMetrics = field(default_factory=StreamingTTSMetrics)
    successful_chunks: int = 0
    failed_chunks: int = 0
    total_processing_time: float = 0.0


class StreamingTTSCoordinator:
    """
    Coordinates streaming text buffer with async TTS service.
    
    This class provides the main interface for streaming TTS functionality,
    managing the flow from LLM tokens to audio files while maintaining
    low latency and high throughput.
    """
    
    def __init__(self,
                 buffer_size: int = DEFAULT_COORDINATOR_BUFFER_SIZE,
                 flush_timeout: float = DEFAULT_COORDINATOR_TIMEOUT,
                 min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
                 max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 audio_prefix: str = DEFAULT_AUDIO_PREFIX,
                 max_concurrent_tts: int = 2,
                 tts_model_name: str = "F5TTS_v1_Base",
                 tts_device: str = "auto"):
        """
        Initialize the streaming TTS coordinator.
        
        Args:
            buffer_size: Maximum buffer size for text accumulation
            flush_timeout: Timeout for buffer flushing
            min_chunk_size: Minimum chunk size for emission
            max_chunk_size: Maximum chunk size for emission
            output_dir: Directory for audio output files
            audio_prefix: Prefix for audio filenames
            max_concurrent_tts: Maximum concurrent TTS jobs
            tts_model_name: F5-TTS model name
            tts_device: Device for TTS processing
        """
        self.output_dir = Path(output_dir)
        self.audio_prefix = audio_prefix
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.buffer = StreamingBuffer(
            max_buffer_size=buffer_size,
            flush_timeout=flush_timeout,
            min_emit_size=min_chunk_size,
            max_emit_size=max_chunk_size
        )
        
        self.tts_service = AsyncTTSService(
            max_queue_size=max_concurrent_tts * 2,
            output_dir=str(self.output_dir),
            audio_prefix=audio_prefix,
            max_concurrent_jobs=max_concurrent_tts,
            model_name=tts_model_name,
            device=tts_device
        )
        
        # State management
        self.state = CoordinatorState.IDLE
        self.is_initialized = False
        self.is_processing = False
        self._shutdown_event = asyncio.Event()
        
        # Metrics and tracking
        self.metrics = StreamingTTSMetrics()
        self.audio_files: List[str] = []
        self.pending_jobs: Dict[str, BufferChunk] = {}
        
        # Callbacks
        self.on_chunk_ready: Optional[Callable[[BufferChunk], None]] = None
        self.on_audio_ready: Optional[Callable[[str, TTSResult], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # Background tasks
        self._result_processor_task: Optional[asyncio.Task] = None
        self._buffer_monitor_task: Optional[asyncio.Task] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Setup buffer callback
        self.buffer.set_chunk_callback(self._on_buffer_chunk_ready)
        
    async def initialize(self):
        """Initialize the coordinator and its components"""
        if self.is_initialized:
            return
            
        self.logger.info("Initializing Streaming TTS Coordinator...")
        
        # Initialize TTS service
        await self.tts_service.initialize()
        
        # Set up TTS callbacks
        self.tts_service.set_completion_callback(self._on_tts_completion)
        self.tts_service.set_failure_callback(self._on_tts_failure)
        
        # Start TTS processing
        self.tts_service.start_processing()
        
        self.is_initialized = True
        self.state = CoordinatorState.READY
        
        self.logger.info("Streaming TTS Coordinator initialized successfully")
        
    async def start_streaming(self):
        """Start streaming processing"""
        if not self.is_initialized:
            raise RuntimeError("Coordinator not initialized. Call initialize() first.")
            
        if self.is_processing:
            return
            
        self.logger.info("Starting streaming TTS processing...")
        
        self.is_processing = True
        self.state = CoordinatorState.STREAMING
        self._shutdown_event.clear()
        
        # Start background tasks
        self._result_processor_task = asyncio.create_task(self._process_results())
        self._buffer_monitor_task = asyncio.create_task(self._monitor_buffer())
        
        # Reset metrics
        self.metrics = StreamingTTSMetrics()
        self.audio_files.clear()
        self.pending_jobs.clear()
        
        self.logger.info("Streaming TTS processing started")
        
    async def stop_streaming(self):
        """Stop streaming processing"""
        if not self.is_processing:
            return
            
        self.logger.info("Stopping streaming TTS processing...")
        
        self.state = CoordinatorState.FINISHING
        
        # Flush any remaining buffer content
        await self._flush_buffer()
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._result_processor_task:
            self._result_processor_task.cancel()
            try:
                await self._result_processor_task
            except asyncio.CancelledError:
                pass
                
        if self._buffer_monitor_task:
            self._buffer_monitor_task.cancel()
            try:
                await self._buffer_monitor_task
            except asyncio.CancelledError:
                pass
                
        # Wait for pending TTS jobs to complete
        await self._wait_for_pending_jobs()
        
        self.is_processing = False
        self.state = CoordinatorState.READY
        
        self.logger.info("Streaming TTS processing stopped")
        
    async def process_token(self, token: str) -> List[str]:
        """
        Process a single token through the streaming pipeline.
        
        Args:
            token: Token to process
            
        Returns:
            List of audio file paths for any completed audio
        """
        if not self.is_processing:
            raise RuntimeError("Streaming not started. Call start_streaming() first.")
            
        # Update metrics
        self.metrics.total_tokens_processed += 1
        
        # Add token to buffer
        chunks = self.buffer.add_token(token)
        
        # Submit chunks to TTS service
        audio_files = []
        for chunk in chunks:
            try:
                job_id = await self.tts_service.submit_chunk(chunk)
                self.pending_jobs[job_id] = chunk
                self.metrics.total_chunks_generated += 1
                
                # Call callback if set
                if self.on_chunk_ready:
                    self.on_chunk_ready(chunk)
                    
            except Exception as e:
                self.logger.error(f"Failed to submit chunk to TTS: {e}")
                if self.on_error:
                    self.on_error(e)
                    
        return audio_files
        
    async def process_token_stream(self, token_stream: AsyncGenerator[str, None]) -> StreamingTTSOutput:
        """
        Process a stream of tokens through the streaming pipeline.
        
        Args:
            token_stream: Async generator of tokens
            
        Returns:
            StreamingTTSOutput with results and metrics
        """
        if not self.is_processing:
            await self.start_streaming()
            
        start_time = time.time()
        
        try:
            # Process tokens
            async for token in token_stream:
                await self.process_token(token)
                
                # Check for timeout-based buffer flushes
                timeout_chunks = self.buffer.check_timeout()
                for chunk in timeout_chunks:
                    try:
                        job_id = await self.tts_service.submit_chunk(chunk)
                        self.pending_jobs[job_id] = chunk
                        self.metrics.total_chunks_generated += 1
                    except Exception as e:
                        self.logger.error(f"Failed to submit timeout chunk: {e}")
                        
            # Flush remaining buffer content
            await self._flush_buffer()
            
            # Wait for all TTS jobs to complete
            await self._wait_for_pending_jobs()
            
            # Calculate final metrics
            self.metrics.total_streaming_time = time.time() - start_time
            self.metrics.buffer_metrics = self.buffer.get_metrics().__dict__
            self.metrics.tts_metrics = self.tts_service.get_metrics()
            
            # Calculate averages
            if self.metrics.total_tokens_processed > 0:
                self.metrics.avg_tokens_per_second = (
                    self.metrics.total_tokens_processed / self.metrics.total_streaming_time
                )
                
            if self.metrics.total_chunks_generated > 0:
                self.metrics.avg_chunk_processing_time = (
                    self.metrics.total_streaming_time / self.metrics.total_chunks_generated
                )
                
            # Create output
            output = StreamingTTSOutput(
                audio_files=self.audio_files.copy(),
                metrics=self.metrics,
                successful_chunks=len(self.audio_files),
                failed_chunks=self.metrics.total_chunks_generated - len(self.audio_files),
                total_processing_time=self.metrics.total_streaming_time
            )
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error in token stream processing: {e}")
            if self.on_error:
                self.on_error(e)
            
            # Return partial results instead of raising
            self.metrics.total_streaming_time = time.time() - start_time
            self.metrics.buffer_metrics = self.buffer.get_metrics().__dict__
            self.metrics.tts_metrics = self.tts_service.get_metrics()
            
            output = StreamingTTSOutput(
                audio_files=self.audio_files.copy(),
                metrics=self.metrics,
                successful_chunks=len(self.audio_files),
                failed_chunks=self.metrics.total_chunks_generated - len(self.audio_files),
                total_processing_time=self.metrics.total_streaming_time
            )
            
            return output
            
    async def _flush_buffer(self):
        """Flush any remaining buffer content"""
        final_chunks = self.buffer.flush_all()
        
        for chunk in final_chunks:
            try:
                job_id = await self.tts_service.submit_chunk(chunk)
                self.pending_jobs[job_id] = chunk
                self.metrics.total_chunks_generated += 1
            except Exception as e:
                self.logger.error(f"Failed to submit final chunk: {e}")
                
    async def _wait_for_pending_jobs(self, timeout: float = COORDINATOR_SHUTDOWN_TIMEOUT):
        """Wait for all pending TTS jobs to complete"""
        if not self.pending_jobs:
            return
            
        self.logger.info(f"Waiting for {len(self.pending_jobs)} pending TTS jobs...")
        
        start_time = time.time()
        while self.pending_jobs and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
            
        if self.pending_jobs:
            self.logger.warning(f"Timeout waiting for {len(self.pending_jobs)} TTS jobs")
            
    def _on_buffer_chunk_ready(self, chunk: BufferChunk):
        """Handle buffer chunk ready callback"""
        chunk_preview = chunk.text[:50] if len(chunk.text) > 50 else chunk.text
        self.logger.debug(f"Buffer chunk ready: {chunk_preview}...")
        
    def _on_tts_completion(self, result: TTSResult):
        """Handle TTS completion callback"""
        self.logger.debug(f"TTS job {result.job_id} completed: {result.output_path}")
        
        # Update metrics
        self.metrics.update_from_tts_result(result)
        
        # Add to audio files
        if result.output_path:
            self.audio_files.append(result.output_path)
            
        # Remove from pending jobs
        chunk = self.pending_jobs.pop(result.job_id, None)
        
        # Call callback if set
        if self.on_audio_ready and chunk:
            self.on_audio_ready(result.output_path, result)
            
    def _on_tts_failure(self, result: TTSResult):
        """Handle TTS failure callback"""
        self.logger.error(f"TTS job {result.job_id} failed: {result.error}")
        
        # Remove from pending jobs
        self.pending_jobs.pop(result.job_id, None)
        
        # Call error callback if set
        if self.on_error:
            self.on_error(Exception(f"TTS job failed: {result.error}"))
            
    async def _process_results(self):
        """Background task to process TTS results"""
        while not self._shutdown_event.is_set():
            try:
                # This is handled by callbacks, so just sleep
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in result processor: {e}")
                
    async def _monitor_buffer(self):
        """Background task to monitor buffer timeouts"""
        while not self._shutdown_event.is_set():
            try:
                # Check for timeout-based flushes
                timeout_chunks = self.buffer.check_timeout()
                
                for chunk in timeout_chunks:
                    try:
                        job_id = await self.tts_service.submit_chunk(chunk)
                        self.pending_jobs[job_id] = chunk
                        self.metrics.total_chunks_generated += 1
                    except Exception as e:
                        self.logger.error(f"Failed to submit timeout chunk: {e}")
                        
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in buffer monitor: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status"""
        return {
            "state": self.state.value,
            "is_initialized": self.is_initialized,
            "is_processing": self.is_processing,
            "pending_jobs": len(self.pending_jobs),
            "audio_files_created": len(self.audio_files),
            "buffer_status": self.buffer.get_buffer_status(),
            "tts_status": self.tts_service.get_queue_status()
        }
        
    def get_metrics(self) -> StreamingTTSMetrics:
        """Get current metrics"""
        return self.metrics
        
    def set_chunk_callback(self, callback: Callable[[BufferChunk], None]):
        """Set callback for chunk ready events"""
        self.on_chunk_ready = callback
        
    def set_audio_callback(self, callback: Callable[[str, TTSResult], None]):
        """Set callback for audio ready events"""
        self.on_audio_ready = callback
        
    def set_error_callback(self, callback: Callable[[Exception], None]):
        """Set callback for error events"""
        self.on_error = callback
        
    async def shutdown(self):
        """Shutdown the coordinator and its components"""
        self.logger.info("Shutting down Streaming TTS Coordinator...")
        
        # Stop streaming if active
        if self.is_processing:
            await self.stop_streaming()
            
        # Shutdown TTS service
        await self.tts_service.shutdown()
        
        # Reset state
        self.is_initialized = False
        self.state = CoordinatorState.STOPPED
        
        self.logger.info("Streaming TTS Coordinator shut down")
        
    def __del__(self):
        """Cleanup when coordinator is destroyed"""
        if self.is_processing:
            # Cannot call async method in destructor
            self.logger.warning("StreamingTTSCoordinator destroyed while processing - use shutdown() for clean cleanup")


# Example usage and testing
async def main():
    """Test the streaming TTS coordinator"""
    print("Testing Streaming TTS Coordinator")
    print("=" * 60)
    
    # Create coordinator
    coordinator = StreamingTTSCoordinator(
        buffer_size=200,
        flush_timeout=1.0,
        min_chunk_size=15,
        max_chunk_size=100,
        output_dir=str(Path(tempfile.gettempdir()) / "streaming_tts_coordinator_test"),
        max_concurrent_tts=1
    )
    
    # Initialize
    await coordinator.initialize()
    
    # Set callbacks
    def on_chunk_ready(chunk):
        print(f"Chunk ready: {chunk.text}")
        
    def on_audio_ready(audio_path, result):
        print(f"Audio ready: {audio_path}")
        print(f"   RTF: {result.metrics.get('rtf', 'N/A'):.4f}")
        
    def on_error(error):
        print(f"Error: {error}")
        
    coordinator.set_chunk_callback(on_chunk_ready)
    coordinator.set_audio_callback(on_audio_ready)
    coordinator.set_error_callback(on_error)
    
    # Create test token stream
    async def token_stream():
        test_text = "Hello world! This is a test of the streaming TTS coordinator. How are you doing today?"
        tokens = test_text.split()
        
        for token in tokens:
            yield token + " "
            await asyncio.sleep(0.1)  # Simulate streaming delay
            
    # Process stream
    print("\nStarting token stream processing...")
    result = await coordinator.process_token_stream(token_stream())
    
    # Display results
    print(f"\nResults:")
    print(f"  Audio files created: {len(result.audio_files)}")
    print(f"  Successful chunks: {result.successful_chunks}")
    print(f"  Failed chunks: {result.failed_chunks}")
    print(f"  Total processing time: {result.total_processing_time:.2f}s")
    
    # Display metrics
    print(f"\nMetrics:")
    print(f"  Tokens processed: {result.metrics.total_tokens_processed}")
    print(f"  Chunks generated: {result.metrics.total_chunks_generated}")
    print(f"  Audio files created: {result.metrics.total_audio_files_created}")
    print(f"  Avg tokens/sec: {result.metrics.avg_tokens_per_second:.1f}")
    print(f"  Avg TTS time: {result.metrics.avg_tts_generation_time:.2f}s")
    print(f"  Avg audio RTF: {result.metrics.avg_audio_rtf:.4f}")
    
    # List audio files
    print(f"\nAudio files:")
    for i, audio_file in enumerate(result.audio_files):
        print(f"  {i+1}. {audio_file}")
        
    # Shutdown
    await coordinator.shutdown()
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())