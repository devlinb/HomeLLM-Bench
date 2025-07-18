#!/usr/bin/env python3
"""
Async TTS Service Wrapper

This module provides an asynchronous wrapper around the F5-TTS service that
enables concurrent processing of TTS requests. It handles queue management,
audio file naming, and provides metrics for TTS performance.
"""

import asyncio
import time
import logging
import tempfile
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import uuid

try:
    from ..tts.streaming_buffer import BufferChunk
except ImportError:
    # For standalone testing
    from streaming_buffer import BufferChunk

# Import F5-TTS service
try:
    from f5tts.f5tts_service import F5TTSService
except ImportError:
    # Create a mock service for testing
    class F5TTSService:
        def __init__(self, *args, **kwargs):
            self.is_loaded = False
            self.logger = logging.getLogger(__name__)
        
        def load_model(self):
            self.is_loaded = True
            
        def generate_speech(self, text: str, output_path: str, **kwargs) -> dict:
            # Mock implementation
            import time
            mock_processing_time = 0.1
            mock_audio_duration = 1.0
            mock_sample_rate = 22050
            time.sleep(mock_processing_time)
            return {
                "generation_time": mock_processing_time,
                "audio_duration": mock_audio_duration,
                "rtf": mock_processing_time / mock_audio_duration,
                "output_path": output_path,
                "text_length": len(text),
                "sample_rate": mock_sample_rate,
                "audio_samples": int(mock_sample_rate * mock_audio_duration)
            }


# Constants
DEFAULT_MAX_QUEUE_SIZE = 10
DEFAULT_AUDIO_OUTPUT_DIR = str(Path(tempfile.gettempdir()) / "tts_output")
DEFAULT_AUDIO_PREFIX = "tts_chunk"
DEFAULT_AUDIO_EXTENSION = ".wav"
DEFAULT_TIMEOUT = 30.0
MAX_CONCURRENT_JOBS = 3
CLEANUP_INTERVAL = 60.0  # seconds
ZERO_DIVISION_EPSILON = 1e-10  # Prevent division by zero
THREAD_SHUTDOWN_TIMEOUT = 10.0  # seconds


@dataclass
class TTSJob:
    """Represents a TTS generation job"""
    id: str
    text: str
    output_path: str
    chunk: BufferChunk
    created_at: float
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class TTSResult:
    """Represents the result of a TTS generation"""
    job_id: str
    success: bool
    output_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    queue_time: float = 0.0


class AsyncTTSService:
    """
    Asynchronous TTS service that provides concurrent processing capabilities.
    
    This service wraps the F5TTSService and provides:
    - Async/await interface for TTS generation
    - Queue management for batch processing
    - Audio file naming and organization
    - Performance metrics and monitoring
    - Concurrent job processing
    """
    
    def __init__(self,
                 max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
                 output_dir: str = DEFAULT_AUDIO_OUTPUT_DIR,
                 audio_prefix: str = DEFAULT_AUDIO_PREFIX,
                 audio_extension: str = DEFAULT_AUDIO_EXTENSION,
                 max_concurrent_jobs: int = MAX_CONCURRENT_JOBS,
                 model_name: str = "F5TTS_v1_Base",
                 device: str = "auto"):
        """
        Initialize the async TTS service.
        
        Args:
            max_queue_size: Maximum number of jobs to queue
            output_dir: Directory for audio output files
            audio_prefix: Prefix for audio filenames
            audio_extension: Extension for audio files
            max_concurrent_jobs: Maximum concurrent TTS jobs
            model_name: F5-TTS model name
            device: Device to run on
        """
        self.max_queue_size = max_queue_size
        self.output_dir = Path(output_dir)
        self.audio_prefix = audio_prefix
        self.audio_extension = audio_extension
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize F5-TTS service
        self.tts_service = F5TTSService(model_name=model_name, device=device)
        
        # Queue and processing
        self.job_queue: Queue[TTSJob] = Queue(maxsize=max_queue_size)
        self.result_queue: Queue[TTSResult] = Queue()
        self.active_jobs: Dict[str, TTSJob] = {}
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs, thread_name_prefix="AsyncTTSWorker")
        
        # Thread safety
        self._metrics_lock = threading.Lock()
        self._state_lock = threading.Lock()
        
        # Control flags
        self.is_running = False
        self.is_loaded = False
        self._shutdown_event = threading.Event()
        
        # Metrics
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.total_processing_time = 0.0
        self.total_queue_time = 0.0
        
        # Callbacks
        self.on_job_complete: Optional[Callable[[TTSResult], None]] = None
        self.on_job_failed: Optional[Callable[[TTSResult], None]] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Background worker threads
        self._worker_threads: List[threading.Thread] = []
        
    async def initialize(self):
        """Initialize the service and load the TTS model"""
        if self.is_loaded:
            return
            
        self.logger.info("Initializing Async TTS Service...")
        
        # Load TTS model in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.tts_service.load_model)
        
        self.is_loaded = True
        self.logger.info("Async TTS Service initialized successfully")
        
    def start_processing(self):
        """Start the background processing workers"""
        if self.is_running:
            return
            
        self.logger.info("Starting TTS processing workers...")
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.max_concurrent_jobs):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"TTSWorker-{i}",
                daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)
            
        self.logger.info(f"Started {len(self._worker_threads)} TTS worker threads")
        
    def stop_processing(self):
        """Stop the background processing workers"""
        if not self.is_running:
            return
            
        self.logger.info("Stopping TTS processing workers...")
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5.0)
            
        self._worker_threads.clear()
        self.logger.info("TTS processing workers stopped")
        
    def _worker_thread(self):
        """Background worker thread for processing TTS jobs"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Get job from queue with timeout
                job = self.job_queue.get(timeout=1.0)
                
                # Record queue time
                queue_time = time.time() - job.created_at
                
                # Process the job
                self._process_job(job, queue_time)
                
                # Mark job as done
                self.job_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker thread error: {e}")
                
    def _process_job(self, job: TTSJob, queue_time: float):
        """Process a single TTS job"""
        self.logger.debug(f"Processing job {job.id}: {job.text[:50]}...")
        
        start_time = time.time()
        
        try:
            # Add to active jobs (thread-safe)
            with self._state_lock:
                self.active_jobs[job.id] = job
            
            # Generate speech
            metrics = self.tts_service.generate_speech(
                text=job.text,
                output_path=job.output_path,
                **job.metadata
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = TTSResult(
                job_id=job.id,
                success=True,
                output_path=job.output_path,
                metrics=metrics,
                processing_time=processing_time,
                queue_time=queue_time
            )
            
            # Update statistics (thread-safe)
            with self._metrics_lock:
                self.jobs_processed += 1
                self.total_processing_time += processing_time
                self.total_queue_time += queue_time
            
            # Add to result queue
            self.result_queue.put(result)
            
            # Call callback if set
            if self.on_job_complete:
                self.on_job_complete(result)
                
            self.logger.debug(f"Job {job.id} completed in {processing_time:.2f}s")
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Create error result
            result = TTSResult(
                job_id=job.id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                queue_time=queue_time
            )
            
            # Update statistics (thread-safe)
            with self._metrics_lock:
                self.jobs_failed += 1
            
            # Add to result queue
            self.result_queue.put(result)
            
            # Call callback if set
            if self.on_job_failed:
                self.on_job_failed(result)
                
            self.logger.error(f"Job {job.id} failed: {e}")
            
        finally:
            # Remove from active jobs (thread-safe)
            with self._state_lock:
                self.active_jobs.pop(job.id, None)
            
    def _generate_audio_filename(self, job_id: str) -> str:
        """Generate a unique filename for audio output"""
        timestamp = int(time.time() * 1000)
        filename = f"{self.audio_prefix}_{timestamp}_{job_id}{self.audio_extension}"
        return str(self.output_dir / filename)
        
    async def submit_chunk(self, chunk: BufferChunk, **kwargs) -> str:
        """
        Submit a text chunk for TTS processing.
        
        Args:
            chunk: BufferChunk to process
            **kwargs: Additional parameters for TTS generation
            
        Returns:
            Job ID for tracking
            
        Raises:
            RuntimeError: If service is not initialized
            asyncio.QueueFull: If queue is full
        """
        if not self.is_loaded:
            raise RuntimeError("Service not initialized. Call initialize() first.")
            
        if not self.is_running:
            raise RuntimeError("Service not running. Call start_processing() first.")
            
        # Create job
        job_id = str(uuid.uuid4())
        output_path = self._generate_audio_filename(job_id)
        
        job = TTSJob(
            id=job_id,
            text=chunk.text,
            output_path=output_path,
            chunk=chunk,
            created_at=time.time(),
            priority=int(chunk.confidence * 100),  # Higher confidence = higher priority
            metadata=kwargs
        )
        
        # Submit to queue
        try:
            self.job_queue.put_nowait(job)
            chunk_preview = chunk.text[:50] if len(chunk.text) > 50 else chunk.text
            self.logger.debug(f"Submitted job {job_id} for text: {chunk_preview}...")
            return job_id
        except Exception as e:
            # Check if it's actually a queue full error
            if "Full" in str(e):
                raise asyncio.QueueFull(f"TTS queue is full: {e}")
            else:
                raise RuntimeError(f"Failed to submit TTS job: {e}")
            
    async def get_result(self, timeout: float = DEFAULT_TIMEOUT) -> Optional[TTSResult]:
        """
        Get the next completed result.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            TTSResult or None if timeout
        """
        loop = asyncio.get_event_loop()
        
        try:
            # Use asyncio timeout with thread-safe queue
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.result_queue.get),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return None
            
    async def get_results(self, count: int = 1, timeout: float = DEFAULT_TIMEOUT) -> List[TTSResult]:
        """
        Get multiple results.
        
        Args:
            count: Number of results to get
            timeout: Timeout in seconds
            
        Returns:
            List of TTSResult objects
        """
        results = []
        end_time = time.time() + timeout
        
        while len(results) < count and time.time() < end_time:
            remaining_time = end_time - time.time()
            if remaining_time <= 0:
                break
                
            result = await self.get_result(timeout=remaining_time)
            if result:
                results.append(result)
            else:
                break
                
        return results
        
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_size": self.job_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_jobs": len(self.active_jobs),
            "is_running": self.is_running,
            "is_loaded": self.is_loaded
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics (thread-safe)"""
        with self._metrics_lock:
            total_jobs = self.jobs_processed + self.jobs_failed
            
            avg_processing_time = (
                self.total_processing_time / max(self.jobs_processed, ZERO_DIVISION_EPSILON)
                if self.jobs_processed > 0 else 0.0
            )
            
            avg_queue_time = (
                self.total_queue_time / max(self.jobs_processed, ZERO_DIVISION_EPSILON)
                if self.jobs_processed > 0 else 0.0
            )
            
            success_rate = (
                self.jobs_processed / max(total_jobs, ZERO_DIVISION_EPSILON)
                if total_jobs > 0 else 0.0
            )
            
            return {
                "jobs_processed": self.jobs_processed,
                "jobs_failed": self.jobs_failed,
                "success_rate": success_rate,
                "avg_processing_time": avg_processing_time,
                "avg_queue_time": avg_queue_time,
                "total_processing_time": self.total_processing_time,
                "total_queue_time": self.total_queue_time
            }
        
    def set_completion_callback(self, callback: Callable[[TTSResult], None]):
        """Set callback for job completion"""
        self.on_job_complete = callback
        
    def set_failure_callback(self, callback: Callable[[TTSResult], None]):
        """Set callback for job failure"""
        self.on_job_failed = callback
        
    async def shutdown(self):
        """Shutdown the service gracefully"""
        self.logger.info("Shutting down Async TTS Service...")
        
        # Stop processing
        self.stop_processing()
        
        # Clear queues
        while not self.job_queue.empty():
            try:
                self.job_queue.get_nowait()
            except Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Empty:
                break
                
        # Shutdown executor with timeout
        try:
            self.executor.shutdown(wait=False)
            # Wait for threads to finish with timeout
            start_time = time.time()
            while (time.time() - start_time) < THREAD_SHUTDOWN_TIMEOUT:
                if not any(t.is_alive() for t in self.executor._threads):
                    break
                time.sleep(0.1)
            else:
                self.logger.warning("TTS executor threads did not shutdown cleanly")
        except Exception as e:
            self.logger.error(f"Error during executor shutdown: {e}")
        
        # Unload TTS model with error handling
        if self.is_loaded:
            try:
                self.tts_service.unload_model()
            except Exception as e:
                self.logger.error(f"Error unloading TTS model: {e}")
            finally:
                self.is_loaded = False
            
        self.logger.info("Async TTS Service shut down")
        
    def __del__(self):
        """Cleanup when service is destroyed"""
        # Only do synchronous cleanup - cannot call async methods
        if self.is_running:
            self.logger.warning("AsyncTTSService destroyed while still running - use shutdown() for clean cleanup")
            try:
                self.stop_processing()
            except Exception as e:
                self.logger.error(f"Error during destructor cleanup: {e}")


# Example usage and testing
async def main():
    """Test the async TTS service"""
    print("Testing Async TTS Service")
    print("=" * 50)
    
    # Create service
    service = AsyncTTSService(
        max_queue_size=5,
        max_concurrent_jobs=2,
        output_dir=str(Path(tempfile.gettempdir()) / "async_tts_test")
    )
    
    # Initialize
    await service.initialize()
    
    # Start processing
    service.start_processing()
    
    # Create test chunks
    from streaming_buffer import BufferChunk, BoundaryType
    
    test_chunks = [
        BufferChunk(
            text="Hello world!",
            confidence=0.9,
            boundary_type=BoundaryType.SENTENCE_END,
            timestamp=time.time()
        ),
        BufferChunk(
            text="This is a test.",
            confidence=0.8,
            boundary_type=BoundaryType.SENTENCE_END,
            timestamp=time.time()
        ),
        BufferChunk(
            text="How are you?",
            confidence=0.7,
            boundary_type=BoundaryType.SENTENCE_END,
            timestamp=time.time()
        )
    ]
    
    # Submit chunks
    job_ids = []
    for chunk in test_chunks:
        job_id = await service.submit_chunk(chunk)
        job_ids.append(job_id)
        print(f"Submitted job {job_id}: {chunk.text}")
        
    # Get results
    print("\nWaiting for results...")
    results = await service.get_results(count=len(job_ids), timeout=30.0)
    
    # Display results
    for result in results:
        if result.success:
            print(f"SUCCESS Job {result.job_id}: {result.output_path}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            print(f"  Queue time: {result.queue_time:.2f}s")
        else:
            print(f"FAILED Job {result.job_id}: {result.error}")
            
    # Show metrics
    print("\nService metrics:")
    metrics = service.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
        
    # Shutdown
    await service.shutdown()
    print("\nTest completed!")


if __name__ == "__main__":
    asyncio.run(main())