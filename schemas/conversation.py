from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageType(str, Enum):
    """Type of message for special handling"""
    NORMAL = "normal"
    RAG_DATA = "rag_data"
    RAG_QUERY = "rag_query"
    RAG_REMOVAL = "rag_removal"
    CONTINUATION = "continuation"


class Message(BaseModel):
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    message_type: MessageType = Field(default=MessageType.NORMAL, description="Type of message for special handling")
    message_metadata: Dict[str, Any] = Field(default_factory=dict, description="Message-specific metadata")
    
    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (4 chars per token)"""
        return len(self.content) // 4


class ConversationTemplate(BaseModel):
    """Template for generating dynamic conversations with context awareness"""
    name: str = Field(..., description="Name/title of the conversation template")
    description: str = Field(..., description="Description of what this conversation tests")
    max_context_tokens: int = Field(..., description="Maximum context size this conversation should use")
    estimated_final_tokens: int = Field(..., description="Estimated final context size after all generations")
    turns: int = Field(..., description="Number of back-and-forth turns")
    base_messages: List[Message] = Field(..., description="Base template messages")
    rag_config: Optional[Dict[str, Any]] = Field(default=None, description="RAG simulation configuration")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    
    def estimate_total_tokens(self) -> int:
        """Estimate total tokens for all base messages"""
        return sum(msg.estimated_tokens for msg in self.base_messages)
    
    def fits_in_context(self, model_context_size: int, generation_buffer: int = 500) -> bool:
        """Check if conversation fits in model context window"""
        total_estimated = self.estimated_final_tokens + generation_buffer
        return total_estimated <= model_context_size


class Conversation(BaseModel):
    name: str = Field(..., description="Name/title of the conversation")
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    description: Optional[str] = Field(default=None, description="Description of what this conversation tests")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="When the conversation was created")
    max_context_tokens: int = Field(default=4096, description="Maximum context this conversation should use")
    estimated_final_tokens: int = Field(default=0, description="Estimated final context size")
    
    def estimate_total_tokens(self) -> int:
        """Estimate total tokens for all messages"""
        return sum(msg.estimated_tokens for msg in self.messages)
    
    def fits_in_context(self, model_context_size: int, generation_buffer: int = 500) -> bool:
        """Check if conversation fits in model context window"""
        total_estimated = self.estimated_final_tokens or self.estimate_total_tokens()
        return total_estimated + generation_buffer <= model_context_size
    
    def to_chatml_format(self) -> List[Dict[str, str]]:
        """Convert to ChatML format for processing"""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.messages
        ]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


if __name__ == "__main__":
    # Example usage
    conversation = Conversation(
        name="Simple Math Test",
        messages=[
            Message(role=MessageRole.USER, content="What is 2+2?"),
            Message(role=MessageRole.ASSISTANT, content="2+2 equals 4."),
            Message(role=MessageRole.USER, content="What about 3+3?"),
            Message(role=MessageRole.ASSISTANT, content="3+3 equals 6.")
        ],
        description="Tests basic arithmetic capabilities",
        tags=["math", "basic"]
    )
    
    print("Conversation schema created successfully!")
    print(f"Name: {conversation.name}")
    print(f"Messages: {len(conversation.messages)}")
    print(f"ChatML format: {conversation.to_chatml_format()}")