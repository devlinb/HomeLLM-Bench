from typing import List, Dict, TYPE_CHECKING
from .base import ChatTemplate

if TYPE_CHECKING:
    from schemas.conversation import Conversation


class Phi3ChatTemplate(ChatTemplate):
    """Chat template for Phi-3.5 models"""
    
    @property
    def name(self) -> str:
        return "phi3"
    
    def format_conversation(self, conversation: 'Conversation') -> str:
        """Convert a conversation to Phi-3.5 format"""
        return self.format_messages(conversation.to_chatml_format())
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages to Phi-3.5 chat format.
        
        Phi-3.5 uses this format:
        <|system|>
        System message<|end|>
        <|user|>
        User message<|end|>
        <|assistant|>
        Assistant message<|end|>
        """
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"<|system|>\n{content}<|end|>")
            elif role == "user":
                formatted_parts.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|>\n{content}<|end|>")
            else:
                raise ValueError(f"Unknown role: {role}")
        
        # Add the final assistant prompt if the last message is not from assistant
        if messages and messages[-1]["role"] != "assistant":
            formatted_parts.append("<|assistant|>")
        
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
    
    template = Phi3ChatTemplate()
    formatted = template.format_conversation(conversation)
    
    print("Phi-3.5 Chat Template Test:")
    print("=" * 50)
    print(formatted)
    print("=" * 50)
    print(f"Template name: {template.name}")