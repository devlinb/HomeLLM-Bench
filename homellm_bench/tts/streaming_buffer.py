#!/usr/bin/env python3
"""
Streaming Text Buffer for TTS Integration

This module provides a streaming text buffer that accumulates tokens from LLM
streaming output and emits complete sentences for TTS processing. It handles
partial sentences, buffering, and intelligent sentence boundary detection.
"""

import time
from typing import List, Optional, Iterator, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

try:
    from .sentence_splitter import SentenceSplitter, BoundaryType
except ImportError:
    # For standalone testing
    from sentence_splitter import SentenceSplitter, BoundaryType


# Constants
DEFAULT_BUFFER_SIZE = 1000
DEFAULT_FLUSH_TIMEOUT = 2.0  # seconds
DEFAULT_MIN_EMIT_SIZE = 15
DEFAULT_MAX_EMIT_SIZE = 200
PARTIAL_SENTENCE_THRESHOLD = 0.3  # Confidence threshold for partial sentences
BUFFER_EFFICIENCY_THRESHOLD = 0.8  # When to trigger incremental processing
MAX_BOUNDARY_CACHE_SIZE = 50  # Maximum cached boundaries to keep
ZERO_DIVISION_EPSILON = 1e-10  # Small value to prevent division by zero


class BufferState(Enum):
    """States of the streaming buffer"""
    EMPTY = "empty"
    ACCUMULATING = "accumulating"
    READY_TO_EMIT = "ready_to_emit"
    FLUSHING = "flushing"
    FINISHED = "finished"


@dataclass
class BufferChunk:
    """Represents a chunk of text ready for TTS processing"""
    text: str
    confidence: float
    boundary_type: BoundaryType
    timestamp: float
    is_partial: bool = False
    token_count: int = 0
    
    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = max(1, len(self.text.split()))


@dataclass
class BufferMetrics:
    """Metrics for buffer performance"""
    total_tokens_received: int = 0
    chunks_emitted: int = 0
    partial_chunks_emitted: int = 0
    buffer_flushes: int = 0
    avg_chunk_size: float = 0.0
    avg_processing_time: float = 0.0
    buffer_overflows: int = 0
    processing_times: List[float] = field(default_factory=list)
    
    def update_chunk_metrics(self, chunk: BufferChunk, processing_time: float):
        """Update metrics when a chunk is emitted"""
        self.chunks_emitted += 1
        if chunk.is_partial:
            self.partial_chunks_emitted += 1
        
        # Update averages with zero-division protection
        if self.chunks_emitted > 0:
            self.avg_chunk_size = (
                (self.avg_chunk_size * (self.chunks_emitted - 1) + len(chunk.text)) / 
                self.chunks_emitted
            )
        
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 0:
            self.avg_processing_time = sum(self.processing_times) / len(self.processing_times)


class BoundarySelector:
    """Handles boundary selection logic"""
    
    def __init__(self, prefer_sentence_ends: bool = True):
        self.prefer_sentence_ends = prefer_sentence_ends
    
    def select_best_boundary(self, boundaries: List[Any], min_position: int) -> Optional[Any]:
        """Select the best boundary for emission"""
        if not boundaries:
            return None
        
        # Filter boundaries by minimum position
        valid_boundaries = [
            b for b in boundaries 
            if b.position >= min_position and b.confidence >= PARTIAL_SENTENCE_THRESHOLD
        ]
        
        if not valid_boundaries:
            return None
        
        # Prefer sentence endings over clause breaks
        if self.prefer_sentence_ends:
            sentence_endings = [
                b for b in valid_boundaries 
                if b.boundary_type == BoundaryType.SENTENCE_END
            ]
            if sentence_endings:
                return sentence_endings[0]
        
        # Fall back to clause breaks
        clause_breaks = [
            b for b in valid_boundaries 
            if b.boundary_type == BoundaryType.CLAUSE_BREAK
        ]
        if clause_breaks:
            return clause_breaks[0]
        
        # Return first valid boundary as last resort
        return valid_boundaries[0]


class BufferProcessor:
    """Handles buffer processing and chunk creation"""
    
    def __init__(self, sentence_splitter: SentenceSplitter, boundary_selector: BoundarySelector):
        self.sentence_splitter = sentence_splitter
        self.boundary_selector = boundary_selector
        self._boundary_cache = {}
        self._last_processed_length = 0
    
    def process_buffer(self, buffer: str, min_emit_size: int) -> List[BufferChunk]:
        """Process buffer and return ready chunks"""
        if len(buffer) < min_emit_size:
            return []
        
        # Use incremental processing for efficiency
        boundaries = self._get_boundaries_incremental(buffer)
        
        # Select best boundary
        best_boundary = self.boundary_selector.select_best_boundary(boundaries, min_emit_size)
        if not best_boundary:
            return []
        
        # Create chunk
        chunk_text = buffer[:best_boundary.position].strip()
        if not chunk_text:
            return []
        
        chunk = BufferChunk(
            text=chunk_text,
            confidence=best_boundary.confidence,
            boundary_type=best_boundary.boundary_type,
            timestamp=time.time(),
            is_partial=False
        )
        
        return [chunk]
    
    def _get_boundaries_incremental(self, buffer: str) -> List[Any]:
        """Get boundaries using incremental processing for efficiency"""
        # If buffer hasn't grown significantly, use cached results
        if (len(buffer) - self._last_processed_length) < (len(buffer) * (1 - BUFFER_EFFICIENCY_THRESHOLD)):
            cache_key = hash(buffer)
            if cache_key in self._boundary_cache:
                return self._boundary_cache[cache_key]
        
        # Process new boundaries
        boundaries = self.sentence_splitter.detect_boundaries(buffer)
        
        # Cache results with size limit
        cache_key = hash(buffer)
        if len(self._boundary_cache) >= MAX_BOUNDARY_CACHE_SIZE:
            # Remove oldest entries
            old_keys = list(self._boundary_cache.keys())[:MAX_BOUNDARY_CACHE_SIZE // 2]
            for key in old_keys:
                del self._boundary_cache[key]
        
        self._boundary_cache[cache_key] = boundaries
        self._last_processed_length = len(buffer)
        
        return boundaries
    
    def force_split_buffer(self, buffer: str, max_chunk_size: int) -> List[BufferChunk]:
        """Force split buffer into chunks when overflow occurs"""
        chunks = self.sentence_splitter.split_into_chunks(buffer.strip())
        
        buffer_chunks = []
        for chunk_text in chunks:
            if chunk_text.strip():
                chunk = BufferChunk(
                    text=chunk_text.strip(),
                    confidence=0.5,  # Medium confidence for forced split
                    boundary_type=BoundaryType.FORCED_BREAK,
                    timestamp=time.time(),
                    is_partial=True
                )
                buffer_chunks.append(chunk)
        
        return buffer_chunks


class StreamingBuffer:
    """
    Streaming text buffer that accumulates tokens and emits complete sentences.
    
    This class handles the buffering of streaming tokens from LLM output,
    detects sentence boundaries, and emits complete sentences for TTS processing.
    """
    
    def __init__(self,
                 max_buffer_size: int = DEFAULT_BUFFER_SIZE,
                 flush_timeout: float = DEFAULT_FLUSH_TIMEOUT,
                 min_emit_size: int = DEFAULT_MIN_EMIT_SIZE,
                 max_emit_size: int = DEFAULT_MAX_EMIT_SIZE,
                 sentence_splitter: Optional[SentenceSplitter] = None,
                 prefer_sentence_ends: bool = True):
        """
        Initialize the streaming buffer.
        
        Args:
            max_buffer_size: Maximum characters to buffer before forcing flush
            flush_timeout: Time in seconds before forcing a flush
            min_emit_size: Minimum characters required to emit a chunk
            max_emit_size: Maximum characters per emitted chunk
            sentence_splitter: Custom sentence splitter (creates default if None)
            prefer_sentence_ends: Whether to prefer sentence endings over clause breaks
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_parameters(max_buffer_size, flush_timeout, min_emit_size, max_emit_size)
        
        self.max_buffer_size = max_buffer_size
        self.flush_timeout = flush_timeout
        self.min_emit_size = min_emit_size
        self.max_emit_size = max_emit_size
        
        # Initialize processing components
        self.sentence_splitter = sentence_splitter or SentenceSplitter(
            min_chunk_size=min_emit_size,
            max_chunk_size=max_emit_size
        )
        
        self.boundary_selector = BoundarySelector(prefer_sentence_ends)
        self.buffer_processor = BufferProcessor(self.sentence_splitter, self.boundary_selector)
        
        # Buffer state - using deque for efficient string building
        self._buffer_parts = deque()
        self._buffer_length = 0
        self.state = BufferState.EMPTY
        self.last_token_time = 0.0
        self.buffer_start_time = 0.0
        
        # Metrics
        self.metrics = BufferMetrics()
        
        # Callbacks
        self.on_chunk_ready: Optional[Callable[[BufferChunk], None]] = None
        self.on_buffer_overflow: Optional[Callable[[str], None]] = None
        self.on_flush: Optional[Callable[[str], None]] = None
    
    def _validate_parameters(self, max_buffer_size: int, flush_timeout: float, 
                           min_emit_size: int, max_emit_size: int):
        """Validate constructor parameters"""
        if max_buffer_size < 10:
            raise ValueError("max_buffer_size must be at least 10")
        if flush_timeout < 0.1:
            raise ValueError("flush_timeout must be at least 0.1 seconds")
        if min_emit_size < 1:
            raise ValueError("min_emit_size must be at least 1")
        if max_emit_size < min_emit_size:
            raise ValueError("max_emit_size must be >= min_emit_size")
    
    def _get_buffer_text(self) -> str:
        """Get current buffer as string"""
        return ''.join(self._buffer_parts)
    
    def _transition_state(self, new_state: BufferState):
        """Safely transition buffer state"""
        # Allow staying in the same state (no-op transitions)
        if self.state == new_state:
            return
        
        # Validate state transitions
        valid_transitions = {
            BufferState.EMPTY: [BufferState.ACCUMULATING, BufferState.FINISHED],
            BufferState.ACCUMULATING: [BufferState.READY_TO_EMIT, BufferState.FLUSHING, BufferState.EMPTY],
            BufferState.READY_TO_EMIT: [BufferState.ACCUMULATING, BufferState.FLUSHING],
            BufferState.FLUSHING: [BufferState.EMPTY, BufferState.FINISHED],
            BufferState.FINISHED: [BufferState.EMPTY, BufferState.FLUSHING]  # Allow reset or final flush
        }
        
        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(f"Invalid state transition from {self.state.value} to {new_state.value}")
        
        self.state = new_state
    
    def set_chunk_callback(self, callback: Callable[[BufferChunk], None]):
        """Set callback for when chunks are ready"""
        self.on_chunk_ready = callback
    
    def set_overflow_callback(self, callback: Callable[[str], None]):
        """Set callback for buffer overflow events"""
        self.on_buffer_overflow = callback
    
    def set_flush_callback(self, callback: Callable[[str], None]):
        """Set callback for flush events"""
        self.on_flush = callback
    
    def add_token(self, token: str) -> List[BufferChunk]:
        """
        Add a token to the buffer and return any ready chunks.
        
        Args:
            token: Token to add to the buffer
            
        Returns:
            List of chunks ready for TTS processing
            
        Raises:
            TypeError: If token is not a string
        """
        if not isinstance(token, str):
            raise TypeError("Token must be a string")
        
        start_time = time.time()
        current_time = time.time()
        
        # Update metrics
        self.metrics.total_tokens_received += 1
        self.last_token_time = current_time
        
        # Initialize buffer timing if empty
        if self.state == BufferState.EMPTY:
            self.buffer_start_time = current_time
            self._transition_state(BufferState.ACCUMULATING)
        
        # Add token to buffer using efficient deque
        self._buffer_parts.append(token)
        self._buffer_length += len(token)
        
        # Check for buffer overflow
        if self._buffer_length > self.max_buffer_size:
            self.metrics.buffer_overflows += 1
            chunks = self._handle_buffer_overflow()
            if self.on_buffer_overflow:
                self.on_buffer_overflow(self._get_buffer_text())
            return self._emit_chunks(chunks, start_time)
        
        # Check for complete sentences
        chunks = self._check_for_complete_sentences()
        return self._emit_chunks(chunks, start_time)
    
    def _emit_chunks(self, chunks: List[BufferChunk], start_time: float) -> List[BufferChunk]:
        """Emit chunks and handle callbacks"""
        if not chunks:
            return []
        
        processing_time = time.time() - start_time
        
        for chunk in chunks:
            # Update metrics
            self.metrics.update_chunk_metrics(chunk, processing_time)
            
            # Call callback if set
            if self.on_chunk_ready:
                self.on_chunk_ready(chunk)
        
        return chunks
    
    def _check_for_complete_sentences(self) -> List[BufferChunk]:
        """Check buffer for complete sentences and emit them"""
        if self._buffer_length < self.min_emit_size:
            return []
        
        buffer_text = self._get_buffer_text()
        chunks = self.buffer_processor.process_buffer(buffer_text, self.min_emit_size)
        
        if chunks:
            # Update buffer by removing processed text
            chunk = chunks[0]  # Only one chunk returned by design
            chunk_length = len(chunk.text)
            
            # Remove processed text from buffer
            self._remove_processed_text(chunk_length)
            
            # Update state
            if self._buffer_length == 0:
                self._transition_state(BufferState.EMPTY)
            else:
                self._transition_state(BufferState.ACCUMULATING)
        
        return chunks
    
    def _remove_processed_text(self, length: int):
        """Remove processed text from buffer efficiently"""
        remaining = length
        while remaining > 0 and self._buffer_parts:
            part = self._buffer_parts.popleft()
            if len(part) <= remaining:
                # Remove entire part
                remaining -= len(part)
                self._buffer_length -= len(part)
            else:
                # Split part and keep remainder
                self._buffer_parts.appendleft(part[remaining:])
                self._buffer_length -= remaining
                remaining = 0
    
    def _handle_buffer_overflow(self) -> List[BufferChunk]:
        """Handle buffer overflow by force-splitting"""
        buffer_text = self._get_buffer_text()
        chunks = self.buffer_processor.force_split_buffer(buffer_text, self.max_emit_size)
        
        # Clear buffer
        self._buffer_parts.clear()
        self._buffer_length = 0
        self._transition_state(BufferState.EMPTY)
        
        return chunks
    
    def check_timeout(self) -> List[BufferChunk]:
        """
        Check for timeout and flush if necessary.
        
        Returns:
            List of chunks if timeout flush occurred, empty list otherwise
        """
        if self.state == BufferState.EMPTY:
            return []
        
        current_time = time.time()
        time_since_last_token = current_time - self.last_token_time
        
        if time_since_last_token >= self.flush_timeout:
            return self._timeout_flush()
        
        return []
    
    def _timeout_flush(self) -> List[BufferChunk]:
        """Flush buffer due to timeout"""
        if self._buffer_length == 0:
            return []
        
        self.metrics.buffer_flushes += 1
        self._transition_state(BufferState.FLUSHING)
        
        buffer_text = self._get_buffer_text().strip()
        
        # Try to find a reasonable break point
        boundaries = self.sentence_splitter.detect_boundaries(buffer_text)
        
        if boundaries:
            # Use the first reasonable boundary
            boundary = boundaries[0]
            chunk_text = buffer_text[:boundary.position].strip()
            
            if chunk_text:
                chunk = BufferChunk(
                    text=chunk_text,
                    confidence=boundary.confidence,
                    boundary_type=boundary.boundary_type,
                    timestamp=time.time(),
                    is_partial=boundary.confidence < PARTIAL_SENTENCE_THRESHOLD
                )
                
                # Update buffer
                self._remove_processed_text(len(chunk_text))
                
                if self._buffer_length == 0:
                    self._transition_state(BufferState.EMPTY)
                else:
                    self._transition_state(BufferState.ACCUMULATING)
                
                if self.on_flush:
                    self.on_flush(chunk_text)
                
                return [chunk]
        
        # No boundaries found, emit the whole buffer as partial
        chunk = BufferChunk(
            text=buffer_text,
            confidence=0.1,  # Low confidence for forced flush
            boundary_type=BoundaryType.FORCED_BREAK,
            timestamp=time.time(),
            is_partial=True
        )
        
        self._buffer_parts.clear()
        self._buffer_length = 0
        self._transition_state(BufferState.EMPTY)
        
        if self.on_flush:
            self.on_flush(buffer_text)
        
        return [chunk]
    
    def flush_all(self) -> List[BufferChunk]:
        """
        Flush all remaining content in the buffer.
        
        Returns:
            List of final chunks
        """
        if self._buffer_length == 0:
            self._transition_state(BufferState.FINISHED)
            return []
        
        self._transition_state(BufferState.FLUSHING)
        
        buffer_text = self._get_buffer_text().strip()
        chunks = self.sentence_splitter.split_into_chunks(buffer_text)
        
        buffer_chunks = []
        for chunk_text in chunks:
            if chunk_text.strip():
                chunk = BufferChunk(
                    text=chunk_text.strip(),
                    confidence=0.8,  # Good confidence for final flush
                    boundary_type=BoundaryType.SENTENCE_END,
                    timestamp=time.time(),
                    is_partial=False
                )
                buffer_chunks.append(chunk)
        
        # Clear buffer and mark as finished
        self._buffer_parts.clear()
        self._buffer_length = 0
        self._transition_state(BufferState.FINISHED)
        
        return buffer_chunks
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status"""
        buffer_text = self._get_buffer_text()
        return {
            "state": self.state.value,
            "buffer_length": self._buffer_length,
            "buffer_content": buffer_text[:100] + "..." if len(buffer_text) > 100 else buffer_text,
            "time_since_last_token": time.time() - self.last_token_time if self.last_token_time else 0,
            "buffer_age": time.time() - self.buffer_start_time if self.buffer_start_time else 0
        }
    
    def get_metrics(self) -> BufferMetrics:
        """Get buffer performance metrics"""
        return self.metrics
    
    def reset(self):
        """Reset the buffer to initial state"""
        self._buffer_parts.clear()
        self._buffer_length = 0
        self._transition_state(BufferState.EMPTY)
        self.last_token_time = 0.0
        self.buffer_start_time = 0.0
        self.metrics = BufferMetrics()
    
    def process_token_stream(self, token_stream: Iterator[str], add_spaces: bool = True) -> Iterator[BufferChunk]:
        """
        Process a stream of tokens and yield chunks as they become ready.
        
        Args:
            token_stream: Iterator of tokens
            add_spaces: Whether to add spaces between tokens
            
        Yields:
            BufferChunk objects ready for TTS processing
        """
        for token in token_stream:
            # Process the token (add space if requested)
            token_to_add = token + " " if add_spaces else token
            chunks = self.add_token(token_to_add)
            
            # Yield any ready chunks
            for chunk in chunks:
                yield chunk
            
            # Check for timeout-based flushes
            timeout_chunks = self.check_timeout()
            for chunk in timeout_chunks:
                yield chunk
        
        # Flush any remaining content
        final_chunks = self.flush_all()
        for chunk in final_chunks:
            yield chunk


# Test constants
TEST_OUTPUT_LIMIT = 10
TEST_TIMEOUT_DURATION = 0.2
TEST_BUFFER_SIZE = 50


def main():
    """Test the streaming buffer with various scenarios"""
    
    print("Streaming Buffer Test")
    print("=" * 60)
    
    # Test case 1: Normal sentence streaming
    print("\nTest 1: Normal sentence streaming")
    print("-" * 40)
    
    buffer = StreamingBuffer(min_emit_size=10, max_emit_size=100)
    
    # Set up callback to track emitted chunks
    emitted_chunks = []
    
    def chunk_callback(chunk):
        emitted_chunks.append(chunk)
        print(f"Emitted: '{chunk.text}' (confidence: {chunk.confidence:.2f}, "
              f"type: {chunk.boundary_type.value})")
    
    buffer.set_chunk_callback(chunk_callback)
    
    # Simulate streaming tokens
    text = "Hello world! This is a test sentence. How are you doing today?"
    tokens = text.split()
    
    for token in tokens:
        buffer.add_token(token + " ")
    
    # Flush remaining
    final_chunks = buffer.flush_all()
    
    print(f"Total chunks emitted: {len(emitted_chunks) + len(final_chunks)}")
    
    # Test case 2: Timeout handling
    print("\nTest 2: Timeout handling")
    print("-" * 40)
    
    buffer2 = StreamingBuffer(flush_timeout=0.1)
    
    # Add partial sentence
    buffer2.add_token("This is incomplete")
    
    # Simulate timeout
    time.sleep(TEST_TIMEOUT_DURATION)
    timeout_chunks = buffer2.check_timeout()
    
    for chunk in timeout_chunks:
        print(f"Timeout flush: '{chunk.text}' (partial: {chunk.is_partial})")
    
    # Test case 3: Buffer overflow
    print("\nTest 3: Buffer overflow")
    print("-" * 40)
    
    buffer3 = StreamingBuffer(max_buffer_size=TEST_BUFFER_SIZE)
    
    # Add text that exceeds buffer size
    long_text = "This is a very long text that will definitely exceed the buffer size limit " * 3
    
    overflow_chunks = buffer3.add_token(long_text)
    
    for chunk in overflow_chunks:
        print(f"Overflow: '{chunk.text[:50]}...' (forced: {chunk.is_partial})")
    
    # Test metrics
    print("\nBuffer Metrics:")
    print("-" * 40)
    
    metrics = buffer.get_metrics()
    print(f"Tokens received: {metrics.total_tokens_received}")
    print(f"Chunks emitted: {metrics.chunks_emitted}")
    print(f"Partial chunks: {metrics.partial_chunks_emitted}")
    print(f"Buffer flushes: {metrics.buffer_flushes}")
    print(f"Average chunk size: {metrics.avg_chunk_size:.1f}")
    
    print("\n" + "=" * 60)
    print("Streaming Buffer tests completed!")


if __name__ == "__main__":
    main()