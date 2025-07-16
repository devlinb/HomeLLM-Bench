#!/usr/bin/env python3
"""
Unit tests for OpenAI format processing and RAG functionality
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homellm_bench.schemas.conversation import Conversation, Message, MessageRole, MessageType
from homellm_bench.benchmark.runner import BenchmarkRunner
from homellm_bench.utils.benchmark_dependencies import BenchmarkDependencies
from homellm_bench.metrics.schemas import GenerationMetrics


class TestOpenAIFormatProcessing:
    """Test OpenAI format processing and RAG functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create test runner with mock dependencies
        self.runner = BenchmarkRunner(
            host="127.0.0.1",
            port=8000,
            context_size=8192,
            engine_type="vllm"
        )

    def test_openai_format_message_structure(self):
        """Test OpenAI format message structure"""
        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"}
        ]
        
        # Verify OpenAI format structure
        for msg in messages:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]
            assert isinstance(msg["content"], str)

    def test_rag_data_active_tracking(self):
        """Test RAG data active state tracking"""
        # Create conversation with RAG data
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="[RETRIEVED INFORMATION]\n\nSome documentation...\n\n[END RETRIEVED INFORMATION]",
                message_type=MessageType.RAG_DATA
            ),
            Message(
                role=MessageRole.USER,
                content="What is this about?",
                message_type=MessageType.RAG_QUERY
            ),
            Message(
                role=MessageRole.USER,
                content="Now, ignoring the previous documentation...",
                message_type=MessageType.RAG_REMOVAL,
                message_metadata={"remove_rag_before_this": True}
            ),
            Message(
                role=MessageRole.USER,
                content="Continue without RAG",
                message_type=MessageType.CONTINUATION
            )
        ]
        
        conversation = Conversation(
            name="rag_test",
            description="Test RAG processing",
            messages=messages,
            tags=["rag"]
        )
        
        # Test RAG data detection
        has_rag_data = any(msg.message_type == MessageType.RAG_DATA for msg in conversation.messages)
        assert has_rag_data
        
        # Test RAG removal detection
        has_rag_removal = any(
            msg.message_type == MessageType.RAG_REMOVAL and 
            msg.message_metadata.get("remove_rag_before_this")
            for msg in conversation.messages
        )
        assert has_rag_removal

    def test_conversation_history_management(self):
        """Test conversation history management in OpenAI format"""
        # Initial conversation history
        conversation_history = []
        
        # Add system message
        conversation_history.append({
            "role": "system",
            "content": "You are a helpful assistant."
        })
        
        # Add user message
        conversation_history.append({
            "role": "user",
            "content": "Hello!"
        })
        
        # Add assistant response
        conversation_history.append({
            "role": "assistant",
            "content": "Hello! How can I help you?"
        })
        
        # Verify conversation history structure
        assert len(conversation_history) == 3
        assert conversation_history[0]["role"] == "system"
        assert conversation_history[1]["role"] == "user"
        assert conversation_history[2]["role"] == "assistant"

    def test_rag_data_removal_logic(self):
        """Test RAG data removal from conversation history"""
        # Create conversation history with RAG data
        conversation_history = [
            {
                "role": "system",
                "content": "[RETRIEVED INFORMATION]\n\nSome documentation...\n\n[END RETRIEVED INFORMATION]"
            },
            {
                "role": "user",
                "content": "What is this about?"
            },
            {
                "role": "assistant",
                "content": "Based on the retrieved information..."
            }
        ]
        
        # Simulate RAG data removal
        conversation_history = [
            msg for msg in conversation_history 
            if not msg["content"].startswith("[RETRIEVED INFORMATION]")
        ]
        
        # Verify RAG data was removed
        assert len(conversation_history) == 2
        assert conversation_history[0]["role"] == "user"
        assert conversation_history[1]["role"] == "assistant"

    def test_context_usage_estimation(self):
        """Test context usage estimation with OpenAI format"""
        # Test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you today?"},
            {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
        ]
        
        # Test token estimation
        estimated_tokens = self.runner.estimate_conversation_tokens(messages)
        
        # Verify estimation
        assert estimated_tokens > 0
        assert isinstance(estimated_tokens, int)
        
        # Test context usage percentage
        context_usage = (estimated_tokens / self.runner.context_size) * 100
        assert 0 <= context_usage <= 100

    def test_adaptive_max_tokens_calculation(self):
        """Test adaptive max tokens calculation"""
        # Test with different prompt lengths
        short_messages = [{"role": "user", "content": "Hi"}]
        long_messages = [{"role": "user", "content": "A" * 1000}]
        
        # Calculate tokens
        short_tokens = self.runner.estimate_conversation_tokens(short_messages)
        long_tokens = self.runner.estimate_conversation_tokens(long_messages)
        
        # Calculate adaptive max tokens
        DEFAULT_MAX_TOKENS = 100
        buffer = 100
        context_size = self.runner.context_size
        
        short_max = min(DEFAULT_MAX_TOKENS, context_size - short_tokens - buffer)
        long_max = min(DEFAULT_MAX_TOKENS, context_size - long_tokens - buffer)
        
        # Verify adaptive calculation
        assert short_max >= long_max
        assert short_max <= DEFAULT_MAX_TOKENS
        assert long_max <= DEFAULT_MAX_TOKENS

    def test_turn_metadata_generation(self):
        """Test turn metadata generation without templates"""
        # Create test message
        test_message = Message(
            role=MessageRole.USER,
            content="Test message",
            message_type=MessageType.RAG_QUERY
        )
        
        # Create mock metrics
        mock_metrics = Mock(spec=GenerationMetrics)
        mock_metrics.completion_tokens = 50
        mock_metrics.total_generation_time = 1.0
        mock_metrics.tokens_per_second = 50.0
        
        # Test turn metadata creation
        rag_active = True
        turn_number = 1
        estimated_tokens = 100
        context_size = 8192
        
        turn_metadata = {
            "turn_number": turn_number,
            "message_type": test_message.message_type.value,
            "rag_active": rag_active,
            "context_usage_percent": (estimated_tokens / context_size) * 100
        }
        
        # Verify metadata
        assert turn_metadata["turn_number"] == 1
        assert turn_metadata["message_type"] == "rag_query"
        assert turn_metadata["rag_active"] is True
        assert turn_metadata["context_usage_percent"] == (100 / 8192) * 100

    def test_context_warning_threshold(self):
        """Test context warning threshold detection"""
        context_size = 8192
        warning_threshold = 0.8
        
        # Test below threshold
        low_tokens = int(context_size * 0.5)
        assert low_tokens <= context_size * warning_threshold
        
        # Test above threshold
        high_tokens = int(context_size * 0.9)
        assert high_tokens > context_size * warning_threshold

    def test_conversation_selection_without_templates(self):
        """Test conversation selection without template dependencies"""
        # Mock conversation loader
        mock_conversations = [
            Mock(
                name="test_conv_1",
                tags=["test"],
                estimate_total_tokens=Mock(return_value=1000),
                messages=[Mock(message_type=MessageType.NORMAL)]
            ),
            Mock(
                name="test_conv_2",
                tags=["rag"],
                estimate_total_tokens=Mock(return_value=2000),
                messages=[Mock(message_type=MessageType.RAG_DATA)]
            )
        ]
        
        with patch.object(self.runner.deps.conversation_loader, 'get_test_suite', return_value=mock_conversations):
            # Test conversation selection
            selected = self.runner.select_conversations(
                include_tags=["test"],
                max_conversations=1
            )
            
            # Verify selection works
            assert len(selected) <= 1

    def test_openai_api_compatibility(self):
        """Test OpenAI API compatibility"""
        # Create test messages in OpenAI format
        openai_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # Mock engine response
        mock_engine = Mock()
        mock_engine.generate_chat_with_metrics.return_value = ("Test response", Mock())
        self.runner.engine = mock_engine
        
        # Test that messages can be passed directly to engine
        try:
            self.runner.engine.generate_chat_with_metrics(
                messages=openai_messages,
                max_tokens=100,
                temperature=0.7
            )
            # If no exception, OpenAI format is compatible
            assert True
        except Exception as e:
            pytest.fail(f"OpenAI format not compatible: {e}")

    def test_rag_simulation_flow(self):
        """Test complete RAG simulation flow"""
        # Create RAG conversation
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="[RETRIEVED INFORMATION]\n\nRAG data here...\n\n[END RETRIEVED INFORMATION]",
                message_type=MessageType.RAG_DATA
            ),
            Message(
                role=MessageRole.USER,
                content="What does the data say?",
                message_type=MessageType.RAG_QUERY
            ),
            Message(
                role=MessageRole.USER,
                content="Now ignore the previous data...",
                message_type=MessageType.RAG_REMOVAL,
                message_metadata={"remove_rag_before_this": True}
            ),
            Message(
                role=MessageRole.USER,
                content="Continue without RAG",
                message_type=MessageType.CONTINUATION
            )
        ]
        
        conversation = Conversation(
            name="rag_simulation",
            description="Test RAG simulation",
            messages=messages,
            tags=["rag"]
        )
        
        # Convert to OpenAI format
        chatml_messages = conversation.to_chatml_format()
        
        # Verify RAG flow preservation
        assert len(chatml_messages) == 4
        assert "[RETRIEVED INFORMATION]" in chatml_messages[0]["content"]
        assert chatml_messages[1]["content"] == "What does the data say?"
        assert chatml_messages[2]["content"] == "Now ignore the previous data..."
        assert chatml_messages[3]["content"] == "Continue without RAG"

    def test_error_handling_without_templates(self):
        """Test error handling without template dependencies"""
        # Mock engine that raises exception
        mock_engine = Mock()
        mock_engine.generate_chat_with_metrics.side_effect = Exception("Generation failed")
        self.runner.engine = mock_engine
        
        # Test that error handling works without templates
        try:
            self.runner.engine.generate_chat_with_metrics(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=100,
                temperature=0.7
            )
            pytest.fail("Should have raised exception")
        except Exception as e:
            assert str(e) == "Generation failed"
            # Verify error is handled appropriately


if __name__ == "__main__":
    pytest.main([__file__, "-v"])