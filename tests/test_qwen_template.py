#!/usr/bin/env python3
"""
Comprehensive tests for Qwen chat template conversion
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from templates.qwen import QwenChatTemplate
from schemas.conversation import Conversation, Message, MessageRole


class TestQwenChatTemplate(unittest.TestCase):
    """Test cases for Qwen chat template"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.template = QwenChatTemplate()
    
    def test_template_name(self):
        """Test that template name is correct"""
        self.assertEqual(self.template.name, "qwen")
    
    def test_single_user_message(self):
        """Test formatting a single user message"""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        result = self.template.format_messages(messages)
        expected = "<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant"
        
        self.assertEqual(result, expected)
    
    def test_system_user_conversation(self):
        """Test system + user message formatting"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                   "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
                   "<|im_start|>assistant")
        
        self.assertEqual(result, expected)
    
    def test_complete_conversation(self):
        """Test a complete conversation with multiple turns"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                   "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
                   "<|im_start|>assistant\n2+2 equals 4.<|im_end|>\n"
                   "<|im_start|>user\nWhat about 3+3?<|im_end|>\n"
                   "<|im_start|>assistant")
        
        self.assertEqual(result, expected)
    
    def test_conversation_ending_with_assistant(self):
        """Test conversation that ends with assistant message (no prompt added)"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>user\nHello<|im_end|>\n"
                   "<|im_start|>assistant\nHi there!<|im_end|>")
        
        self.assertEqual(result, expected)
    
    def test_multiline_content(self):
        """Test messages with multiline content"""
        messages = [
            {"role": "user", "content": "Please explain:\n1. What is AI?\n2. How does it work?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence.\n\nIt works by:\n- Processing data\n- Learning patterns"}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>user\nPlease explain:\n1. What is AI?\n2. How does it work?<|im_end|>\n"
                   "<|im_start|>assistant\nAI stands for Artificial Intelligence.\n\nIt works by:\n- Processing data\n- Learning patterns<|im_end|>")
        
        self.assertEqual(result, expected)
    
    def test_empty_content(self):
        """Test messages with empty content"""
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "I notice you sent an empty message."}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>user\n<|im_end|>\n"
                   "<|im_start|>assistant\nI notice you sent an empty message.<|im_end|>")
        
        self.assertEqual(result, expected)
    
    def test_special_characters(self):
        """Test messages with special characters and formatting"""
        messages = [
            {"role": "user", "content": "Can you format this code: print(\"Hello, World!\")"},
            {"role": "assistant", "content": "```python\nprint(\"Hello, World!\")\n```"}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>user\nCan you format this code: print(\"Hello, World!\")<|im_end|>\n"
                   "<|im_start|>assistant\n```python\nprint(\"Hello, World!\")\n```<|im_end|>")
        
        self.assertEqual(result, expected)
    
    def test_unknown_role_raises_error(self):
        """Test that unknown roles raise ValueError"""
        messages = [
            {"role": "unknown", "content": "This should fail"}
        ]
        
        with self.assertRaises(ValueError) as context:
            self.template.format_messages(messages)
        
        self.assertIn("Unknown role: unknown", str(context.exception))
    
    def test_format_conversation_method(self):
        """Test the format_conversation method with Conversation object"""
        conversation = Conversation(
            name="Test Conversation",
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="Hello!"),
                Message(role=MessageRole.ASSISTANT, content="Hi there!"),
                Message(role=MessageRole.USER, content="How are you?")
            ]
        )
        
        result = self.template.format_conversation(conversation)
        expected = ("<|im_start|>system\nYou are helpful.<|im_end|>\n"
                   "<|im_start|>user\nHello!<|im_end|>\n"
                   "<|im_start|>assistant\nHi there!<|im_end|>\n"
                   "<|im_start|>user\nHow are you?<|im_end|>\n"
                   "<|im_start|>assistant")
        
        self.assertEqual(result, expected)
    
    def test_long_conversation(self):
        """Test formatting a longer conversation"""
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"User message {i+1}"})
            messages.append({"role": "assistant", "content": f"Assistant response {i+1}"})
        
        # Add one more user message
        messages.append({"role": "user", "content": "Final user message"})
        
        result = self.template.format_messages(messages)
        
        # Verify it contains all expected parts
        self.assertIn("<|im_start|>user\nUser message 1<|im_end|>", result)
        self.assertIn("<|im_start|>assistant\nAssistant response 1<|im_end|>", result)
        self.assertIn("<|im_start|>user\nFinal user message<|im_end|>", result)
        self.assertTrue(result.endswith("<|im_start|>assistant"))
    
    def test_whitespace_preservation(self):
        """Test that whitespace in content is preserved"""
        messages = [
            {"role": "user", "content": "   Leading and trailing spaces   "},
            {"role": "assistant", "content": "Tabs\tand\nnewlines\rpreserved"}
        ]
        
        result = self.template.format_messages(messages)
        expected = ("<|im_start|>user\n   Leading and trailing spaces   <|im_end|>\n"
                   "<|im_start|>assistant\nTabs\tand\nnewlines\rpreserved<|im_end|>")
        
        self.assertEqual(result, expected)


class TestQwenTemplateComparison(unittest.TestCase):
    """Test Qwen template against known good examples"""
    
    def setUp(self):
        self.template = QwenChatTemplate()
    
    def test_qwen_official_format(self):
        """Test against Qwen's official chat format example"""
        # This is based on Qwen's official documentation format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about yourself."}
        ]
        
        result = self.template.format_messages(messages)
        
        # Verify the format matches Qwen's expected structure
        self.assertTrue(result.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_end|>\n<|im_start|>user\n", result)
        self.assertTrue(result.endswith("<|im_start|>assistant"))
        
        # Verify no extra newlines or formatting issues
        lines = result.split('\n')
        self.assertEqual(lines[0], "<|im_start|>system")
        self.assertEqual(lines[1], "You are a helpful assistant.<|im_end|>")
        self.assertEqual(lines[2], "<|im_start|>user")
        self.assertEqual(lines[3], "Tell me about yourself.<|im_end|>")
        self.assertEqual(lines[4], "<|im_start|>assistant")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)