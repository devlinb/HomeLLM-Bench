from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime


class GenerationMetrics(BaseModel):
    """Metrics collected during text generation"""
    
    # Basic token metrics
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens generated")
    total_tokens: int = Field(..., description="Total tokens (prompt + completion)")
    
    # Timing metrics
    time_to_first_token: float = Field(..., description="Time to generate first token (seconds)")
    total_generation_time: float = Field(..., description="Total generation time (seconds)")
    tokens_per_second: float = Field(..., description="Generation speed (tokens/second)")
    
    
    # Engine-specific metrics
    engine_metrics: Dict[str, Any] = Field(default_factory=dict, description="Engine-specific metrics")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="When metrics were collected")
    engine_name: str = Field(..., description="Name of the inference engine")
    model_name: str = Field(..., description="Name of the model")
    turn_metadata: Dict[str, Any] = Field(default_factory=dict, description="Turn-specific metadata")
    tts_metadata: Optional[Dict[str, Any]] = Field(default=None, description="TTS-specific metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VLLMMetrics(BaseModel):
    """vLLM-specific metrics parsed from server responses"""
    
    # vLLM performance metrics
    prefill_time: Optional[float] = Field(default=None, description="Time spent on prefill phase")
    decode_time: Optional[float] = Field(default=None, description="Time spent on decode phase")
    queue_time: Optional[float] = Field(default=None, description="Time spent waiting in queue")
    
    # vLLM cache metrics
    num_cached_tokens: Optional[int] = Field(default=None, description="Number of cached tokens used")
    num_generated_tokens: Optional[int] = Field(default=None, description="Number of tokens generated")
    
    # vLLM system metrics
    running_requests: Optional[int] = Field(default=None, description="Number of running requests")
    waiting_requests: Optional[int] = Field(default=None, description="Number of waiting requests")
    gpu_cache_usage: Optional[float] = Field(default=None, description="GPU cache usage percentage")
    
    # Raw metrics data
    raw_metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw metrics from vLLM")


class OllamaMetrics(BaseModel):
    """Ollama-specific metrics parsed from server responses"""
    
    # Ollama timing metrics (converted from nanoseconds)
    total_duration_seconds: Optional[float] = Field(default=None, description="Total request duration")
    load_duration_seconds: Optional[float] = Field(default=None, description="Model loading time")
    prompt_eval_duration_seconds: Optional[float] = Field(default=None, description="Prompt evaluation time")
    eval_duration_seconds: Optional[float] = Field(default=None, description="Response generation time")
    
    # Ollama token counts
    prompt_eval_count: Optional[int] = Field(default=None, description="Tokens in prompt evaluation")
    eval_count: Optional[int] = Field(default=None, description="Tokens in response generation")
    
    # Ollama raw metrics (in nanoseconds)
    total_duration_ns: Optional[int] = Field(default=None, description="Total duration in nanoseconds")
    load_duration_ns: Optional[int] = Field(default=None, description="Load duration in nanoseconds")
    prompt_eval_duration_ns: Optional[int] = Field(default=None, description="Prompt eval duration in nanoseconds")
    eval_duration_ns: Optional[int] = Field(default=None, description="Eval duration in nanoseconds")
    
    # Raw metrics data
    raw_metrics: Dict[str, Any] = Field(default_factory=dict, description="Raw metrics from Ollama")


class ConversationBenchmarkResult(BaseModel):
    """Results from benchmarking a single conversation"""
    
    conversation_name: str = Field(..., description="Name of the conversation")
    total_turns: int = Field(..., description="Number of turns in the conversation")
    total_time: float = Field(..., description="Total time for entire conversation")
    
    # Per-turn metrics
    turn_metrics: list[GenerationMetrics] = Field(..., description="Metrics for each turn")
    
    # Aggregate metrics
    avg_tokens_per_second: float = Field(..., description="Average tokens per second across all turns")
    total_tokens_generated: int = Field(..., description="Total tokens generated in conversation")
    cache_effectiveness: Optional[float] = Field(default=None, description="Overall cache effectiveness")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    conversation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }