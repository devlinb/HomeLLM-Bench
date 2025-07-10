import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.conversation import Conversation, Message, MessageRole
from templates.phi3 import Phi3ChatTemplate


class TestPhi3ChatTemplate(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.template = Phi3ChatTemplate()
    
    def test_template_name(self):
        """Test that template name is correct"""
        self.assertEqual(self.template.name, "phi3")
    
    def test_simple_user_message(self):
        """Test formatting a simple user message"""
        messages = [{"role": "user", "content": "Hello"}]
        result = self.template.format_messages(messages)
        expected = "<|user|>\nHello<|end|>\n<|assistant|>"
        self.assertEqual(result, expected)
    
    def test_system_user_assistant_flow(self):
        """Test a complete conversation flow with system, user, and assistant"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]
        result = self.template.format_messages(messages)
        expected = (
            "<|system|>\n"
            "You are a helpful assistant.<|end|>\n"
            "<|user|>\n"
            "What is 2+2?<|end|>\n"
            "<|assistant|>\n"
            "2+2 equals 4.<|end|>"
        )
        self.assertEqual(result, expected)
    
    def test_conversation_ending_with_user_message(self):
        """Test that assistant prompt is added when conversation ends with user message"""
        messages = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "I don't have access to weather data."},
            {"role": "user", "content": "What about tomorrow?"}
        ]
        result = self.template.format_messages(messages)
        expected = (
            "<|user|>\n"
            "What is the weather?<|end|>\n"
            "<|assistant|>\n"
            "I don't have access to weather data.<|end|>\n"
            "<|user|>\n"
            "What about tomorrow?<|end|>\n"
            "<|assistant|>"
        )
        self.assertEqual(result, expected)
    
    def test_conversation_ending_with_assistant_message(self):
        """Test that no extra assistant prompt is added when conversation ends with assistant"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        result = self.template.format_messages(messages)
        expected = (
            "<|user|>\n"
            "Hello<|end|>\n"
            "<|assistant|>\n"
            "Hi there!<|end|>"
        )
        self.assertEqual(result, expected)
    
    def test_format_conversation_method(self):
        """Test the format_conversation method with Conversation object"""
        conversation = Conversation(
            name="Test Conversation",
            messages=[
                Message(role=MessageRole.SYSTEM, content="You are helpful."),
                Message(role=MessageRole.USER, content="Hello"),
                Message(role=MessageRole.ASSISTANT, content="Hi!"),
                Message(role=MessageRole.USER, content="How are you?")
            ]
        )
        result = self.template.format_conversation(conversation)
        expected = (
            "<|system|>\n"
            "You are helpful.<|end|>\n"
            "<|user|>\n"
            "Hello<|end|>\n"
            "<|assistant|>\n"
            "Hi!<|end|>\n"
            "<|user|>\n"
            "How are you?<|end|>\n"
            "<|assistant|>"
        )
        self.assertEqual(result, expected)
    
    def test_empty_messages_list(self):
        """Test handling of empty messages list"""
        messages = []
        result = self.template.format_messages(messages)
        self.assertEqual(result, "")
    
    def test_invalid_role(self):
        """Test that invalid role raises ValueError"""
        messages = [{"role": "invalid_role", "content": "test"}]
        with self.assertRaises(ValueError) as context:
            self.template.format_messages(messages)
        self.assertIn("Unknown role: invalid_role", str(context.exception))
    
    def test_multiline_content(self):
        """Test handling of multiline message content"""
        messages = [
            {"role": "user", "content": "Can you explain:\n1. First point\n2. Second point"},
            {"role": "assistant", "content": "Sure!\n\nHere's the explanation:\n- Point 1 details\n- Point 2 details"}
        ]
        result = self.template.format_messages(messages)
        expected = (
            "<|user|>\n"
            "Can you explain:\n"
            "1. First point\n"
            "2. Second point<|end|>\n"
            "<|assistant|>\n"
            "Sure!\n"
            "\n"
            "Here's the explanation:\n"
            "- Point 1 details\n"
            "- Point 2 details<|end|>"
        )
        self.assertEqual(result, expected)
    
    def test_system_only_message(self):
        """Test formatting with only a system message"""
        messages = [{"role": "system", "content": "You are a coding assistant."}]
        result = self.template.format_messages(messages)
        expected = "<|system|>\nYou are a coding assistant.<|end|>\n<|assistant|>"
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)