"""
TTS (Text-to-Speech) integration package for HomeLLM-Bench.

This package provides streaming TTS functionality that works with LLM engines
to generate audio output with low latency.
"""

from .sentence_splitter import SentenceSplitter, BoundaryType
from .streaming_buffer import StreamingBuffer, BufferChunk, BufferState, BufferMetrics
from .async_tts_service import AsyncTTSService, TTSJob, TTSResult
from .streaming_tts_coordinator import StreamingTTSCoordinator, StreamingTTSMetrics, StreamingTTSOutput

__all__ = [
    'SentenceSplitter', 'BoundaryType',
    'StreamingBuffer', 'BufferChunk', 'BufferState', 'BufferMetrics',
    'AsyncTTSService', 'TTSJob', 'TTSResult',
    'StreamingTTSCoordinator', 'StreamingTTSMetrics', 'StreamingTTSOutput'
]