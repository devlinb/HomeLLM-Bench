#!/usr/bin/env python3
"""
Unit tests for conversation processing with OpenAI format (template system removed)
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homellm_bench.schemas.conversation import Conversation, Message, MessageRole, MessageType
from homellm_bench.benchmark.runner import BenchmarkRunner
from homellm_bench.config.constants import TOKEN_ESTIMATION_DIVISOR


class TestConversationProcessing:
    """Test conversation processing with OpenAI format"""

    def test_estimate_conversation_tokens(self):
        """Test token estimation for OpenAI format messages"""
        # Create a mock runner
        runner = Mock(spec=BenchmarkRunner)
        runner.estimate_conversation_tokens = BenchmarkRunner.estimate_conversation_tokens.__get__(runner, BenchmarkRunner)
        
        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]
        
        # Calculate expected tokens
        total_chars = sum(len(msg['content']) for msg in messages)
        expected_tokens = total_chars // TOKEN_ESTIMATION_DIVISOR
        
        # Test token estimation
        estimated_tokens = runner.estimate_conversation_tokens(messages)
        assert estimated_tokens == expected_tokens
        
    def test_estimate_conversation_tokens_empty(self):
        """Test token estimation with empty messages"""
        runner = Mock(spec=BenchmarkRunner)
        runner.estimate_conversation_tokens = BenchmarkRunner.estimate_conversation_tokens.__get__(runner, BenchmarkRunner)
        
        messages = []
        estimated_tokens = runner.estimate_conversation_tokens(messages)
        assert estimated_tokens == 0

    def test_estimate_conversation_tokens_single_message(self):
        """Test token estimation with single message"""
        runner = Mock(spec=BenchmarkRunner)
        runner.estimate_conversation_tokens = BenchmarkRunner.estimate_conversation_tokens.__get__(runner, BenchmarkRunner)
        
        messages = [{"role": "user", "content": "Test message"}]
        expected_tokens = len("Test message") // TOKEN_ESTIMATION_DIVISOR
        
        estimated_tokens = runner.estimate_conversation_tokens(messages)
        assert estimated_tokens == expected_tokens

    def test_conversation_to_chatml_format(self):
        """Test conversation conversion to ChatML format"""
        # Create test messages
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
                message_type=MessageType.NORMAL
            ),
            Message(
                role=MessageRole.USER,
                content="Hello, how are you?",
                message_type=MessageType.NORMAL
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="I'm doing well!",
                message_type=MessageType.NORMAL
            )
        ]
        
        # Create conversation
        conversation = Conversation(
            name="test_conversation",
            description="Test conversation",
            messages=messages,
            tags=["test"]
        )
        
        # Convert to ChatML format
        chatml_messages = conversation.to_chatml_format()
        
        # Verify format
        assert len(chatml_messages) == 3
        assert chatml_messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert chatml_messages[1] == {"role": "user", "content": "Hello, how are you?"}
        assert chatml_messages[2] == {"role": "assistant", "content": "I'm doing well!"}

    def test_rag_message_processing(self):
        """Test RAG message processing with OpenAI format"""
        # Create test messages with RAG data
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="[RETRIEVED INFORMATION]\n\nSome documentation...\n\n[END RETRIEVED INFORMATION]",
                message_type=MessageType.RAG_DATA
            ),
            Message(
                role=MessageRole.USER,
                content="What is the documentation about?",
                message_type=MessageType.RAG_QUERY
            ),
            Message(
                role=MessageRole.USER,
                content="Now, ignoring the previous documentation...",
                message_type=MessageType.RAG_REMOVAL,
                message_metadata={"remove_rag_before_this": True}
            )
        ]
        
        # Create conversation
        conversation = Conversation(
            name="rag_test",
            description="Test RAG processing",
            messages=messages,
            tags=["rag"]
        )
        
        # Convert to ChatML format
        chatml_messages = conversation.to_chatml_format()
        
        # Verify all messages are preserved (RAG removal is handled by runner)
        assert len(chatml_messages) == 3
        assert chatml_messages[0]["role"] == "system"
        assert "[RETRIEVED INFORMATION]" in chatml_messages[0]["content"]
        assert chatml_messages[1]["role"] == "user"
        assert chatml_messages[2]["role"] == "user"

    def test_message_type_preservation(self):
        """Test that message types are preserved in Message objects"""
        # Create messages with different types
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="Normal system message",
                message_type=MessageType.NORMAL
            ),
            Message(
                role=MessageRole.SYSTEM,
                content="RAG data",
                message_type=MessageType.RAG_DATA
            ),
            Message(
                role=MessageRole.USER,
                content="RAG query",
                message_type=MessageType.RAG_QUERY
            ),
            Message(
                role=MessageRole.USER,
                content="Remove RAG",
                message_type=MessageType.RAG_REMOVAL
            ),
            Message(
                role=MessageRole.USER,
                content="Continue conversation",
                message_type=MessageType.CONTINUATION
            )
        ]
        
        # Verify message types are preserved
        assert messages[0].message_type == MessageType.NORMAL
        assert messages[1].message_type == MessageType.RAG_DATA
        assert messages[2].message_type == MessageType.RAG_QUERY
        assert messages[3].message_type == MessageType.RAG_REMOVAL
        assert messages[4].message_type == MessageType.CONTINUATION

    def test_conversation_token_estimation(self):
        """Test conversation-level token estimation"""
        # Create test messages
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
                message_type=MessageType.NORMAL
            ),
            Message(
                role=MessageRole.USER,
                content="Hello, how are you?",
                message_type=MessageType.NORMAL
            )
        ]
        
        # Create conversation
        conversation = Conversation(
            name="test_conversation",
            description="Test conversation",
            messages=messages,
            tags=["test"]
        )
        
        # Test token estimation
        estimated_tokens = conversation.estimate_total_tokens()
        expected_tokens = sum(len(msg.content) // TOKEN_ESTIMATION_DIVISOR for msg in messages)
        assert estimated_tokens == expected_tokens

    def test_rag_data_detection(self):
        """Test RAG data detection in conversations"""
        # Create conversation with RAG data
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="[RETRIEVED INFORMATION]\n\nSome data...\n\n[END RETRIEVED INFORMATION]",
                message_type=MessageType.RAG_DATA
            ),
            Message(
                role=MessageRole.USER,
                content="What is this about?",
                message_type=MessageType.RAG_QUERY
            )
        ]
        
        conversation = Conversation(
            name="rag_conversation",
            description="RAG test conversation",
            messages=messages,
            tags=["rag"]
        )
        
        # Test RAG data detection
        has_rag_data = any(msg.message_type == MessageType.RAG_DATA for msg in conversation.messages)
        assert has_rag_data

    def test_context_usage_calculation(self):
        """Test context usage percentage calculation"""
        # Test data
        estimated_tokens = 1000
        context_size = 4000
        
        # Calculate context usage
        context_usage_percent = (estimated_tokens / context_size) * 100
        
        # Verify calculation
        assert context_usage_percent == 25.0

    def test_openai_format_compatibility(self):
        """Test that ChatML format is compatible with OpenAI API"""
        # Create test messages
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a helpful assistant.",
                message_type=MessageType.NORMAL
            ),
            Message(
                role=MessageRole.USER,
                content="Hello!",
                message_type=MessageType.NORMAL
            )
        ]
        
        conversation = Conversation(
            name="openai_test",
            description="OpenAI compatibility test",
            messages=messages,
            tags=["openai"]
        )
        
        # Convert to ChatML format
        chatml_messages = conversation.to_chatml_format()
        
        # Verify OpenAI API compatibility
        for msg in chatml_messages:
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["role"], str)
            assert isinstance(msg["content"], str)
            assert msg["role"] in ["system", "user", "assistant"]


class TestBenchmarkRunnerConfiguration:
    """Test benchmark runner configuration"""
    
    def test_benchmark_runner_initialization(self):
        """Test that BenchmarkRunner initializes properly"""
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8000,
            context_size=16000,
            engine_type="vllm"
        )
        
        # Verify configuration
        assert runner.host == "127.0.0.1"
        assert runner.port == 8000
        assert runner.context_size == 16000
        assert runner.engine_type == "vllm"

    def test_benchmark_runner_defaults(self):
        """Test benchmark runner default values"""
        runner = BenchmarkRunner()
        
        # Should use defaults from constants
        assert runner.host == "127.0.0.1"
        assert runner.port == 8000
        assert runner.context_size == 32000
        assert runner.engine_type == "vllm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])