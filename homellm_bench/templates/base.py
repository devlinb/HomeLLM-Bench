from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from schemas.conversation import Conversation


class ChatTemplate(ABC):
    """Abstract base class for chat templates"""
    
    @abstractmethod
    def format_conversation(self, conversation: 'Conversation') -> str:
        """Convert a conversation to the model's expected format"""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert a list of messages to the model's expected format"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the template"""
        pass