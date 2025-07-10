from typing import List, Dict, TYPE_CHECKING
from .base import ChatTemplate

if TYPE_CHECKING:
    from schemas.conversation import Conversation


class QwenChatTemplate(ChatTemplate):
    """Chat template for Qwen models"""
    
    @property
    def name(self) -> str:
        return "qwen"
    
    def format_conversation(self, conversation: 'Conversation') -> str:
        """Convert a conversation to Qwen format"""
        return self.format_messages(conversation.to_chatml_format())
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages to Qwen chat format.
        
        Qwen uses this format:
        <|im_start|>system
        System message<|im_end|>
        <|im_start|>user
        User message<|im_end|>
        <|im_start|>assistant
        Assistant message<|im_end|>
        """
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            else:
                raise ValueError(f"Unknown role: {role}")
        
        # Add the final assistant prompt if the last message is not from assistant
        if messages and messages[-1]["role"] != "assistant":
            formatted_parts.append("<|im_start|>assistant")
        
        return "\n".join(formatted_parts)


if __name__ == "__main__":
    # Test the template
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from schemas.conversation import Conversation, Message, MessageRole
    
    # Test conversation
    conversation = Conversation(
        name="Template Test",
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="What is 2+2?"),
            Message(role=MessageRole.ASSISTANT, content="2+2 equals 4."),
            Message(role=MessageRole.USER, content="What about 3+3?")
        ]
    )
    
    template = QwenChatTemplate()
    formatted = template.format_conversation(conversation)
    
    print("Qwen Chat Template Test:")
    print("=" * 50)
    print(formatted)
    print("=" * 50)
    print(f"Template name: {template.name}")