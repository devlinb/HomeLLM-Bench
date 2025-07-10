"""
Conversation data loader with context awareness and RAG simulation
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..schemas.conversation import Conversation, ConversationTemplate, Message, MessageRole, MessageType


class ConversationLoader:
    """Loads and manages conversation data with context awareness"""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Use package data directory
            self.data_dir = Path(__file__).parent
        else:
            self.data_dir = Path(data_dir)
        self.conversations_file = self.data_dir / "conversations.json"
        self.rag_data_file = self.data_dir / "rag_data.json"
        
        # Load data files
        self.conversation_templates = self._load_conversations()
        self.rag_data = self._load_rag_data()
    
    def _load_conversations(self) -> Dict[str, ConversationTemplate]:
        """Load conversation templates from JSON file"""
        try:
            with open(self.conversations_file, 'r') as f:
                data = json.load(f)
            
            templates = {}
            for key, conv_data in data.items():
                # Convert base_messages to Message objects
                messages = []
                for msg_data in conv_data["base_messages"]:
                    message = Message(
                        role=MessageRole(msg_data["role"]),
                        content=msg_data["content"],
                        message_type=MessageType(msg_data.get("message_type", "normal")),
                        message_metadata=msg_data.get("message_metadata", {})
                    )
                    messages.append(message)
                
                template = ConversationTemplate(
                    name=conv_data["name"],
                    description=conv_data["description"],
                    max_context_tokens=conv_data["max_context_tokens"],
                    estimated_final_tokens=conv_data["estimated_final_tokens"],
                    turns=conv_data["turns"],
                    base_messages=messages,
                    rag_config=conv_data.get("rag_config"),
                    tags=conv_data.get("tags", [])
                )
                templates[key] = template
            
            return templates
            
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return {}
    
    def _load_rag_data(self) -> Dict[str, Dict[str, str]]:
        """Load RAG simulation data"""
        try:
            with open(self.rag_data_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading RAG data: {e}")
            return {}
    
    def get_conversations_for_context(self, 
                                    model_context_size: int,
                                    generation_buffer: int = 1000,
                                    tags: Optional[List[str]] = None) -> List[Conversation]:
        """Get conversations that fit within the model's context window"""
        
        suitable_conversations = []
        
        for key, template in self.conversation_templates.items():
            # Check if conversation fits in context
            if not template.fits_in_context(model_context_size, generation_buffer):
                print(f"Warning: Skipping '{template.name}' - estimated {template.estimated_final_tokens} tokens exceeds context limit")
                continue
            
            # Check tag filtering
            if tags and not any(tag in template.tags for tag in tags):
                continue
            
            # Generate the actual conversation
            conversation = self._generate_conversation_from_template(template)
            suitable_conversations.append(conversation)
        
        return suitable_conversations
    
    def _generate_conversation_from_template(self, template: ConversationTemplate) -> Conversation:
        """Generate a conversation from a template, handling simplified RAG simulation"""
        
        messages = []
        rag_message_indices = []  # Track all RAG message positions
        
        for i, base_message in enumerate(template.base_messages):
            # Check if this message triggers RAG data removal
            if (template.rag_config and 
                base_message.message_metadata.get("remove_rag_before_this") and
                rag_message_indices):
                
                # Remove all RAG data messages (in reverse order to maintain indices)
                for rag_idx in reversed(rag_message_indices):
                    if rag_idx < len(messages):
                        removed_msg = messages.pop(rag_idx)
                        print(f"   Removing RAG data message (was at position {rag_idx + 1})")
                
                rag_message_indices.clear()
                # Mark this message to track the removal
                base_message.message_metadata["rag_removed_before_this_turn"] = True
            
            # Add the base message
            messages.append(base_message)
            
            # Track RAG data message positions
            if base_message.message_type == MessageType.RAG_DATA:
                rag_message_indices.append(len(messages) - 1)
        
        return Conversation(
            name=template.name,
            messages=messages,
            description=template.description,
            tags=template.tags,
            max_context_tokens=template.max_context_tokens,
            estimated_final_tokens=template.estimated_final_tokens,
            metadata={
                "generated_from_template": True,
                "rag_simulation": template.rag_config is not None,
                "template_key": template.name.lower().replace(" ", "_")
            }
        )
    
    def _get_rag_content(self, data_type: str) -> str:
        """Get RAG content for injection"""
        if data_type in self.rag_data:
            data = self.rag_data[data_type]
            return f"# {data['title']}\n\n{data['content']}"
        else:
            return f"[RAG data for {data_type} would be inserted here]"
    
    def list_available_conversations(self, model_context_size: Optional[int] = None) -> None:
        """List all available conversations with context information"""
        print("Available Conversation Templates:")
        print("=" * 80)
        
        for key, template in self.conversation_templates.items():
            print(f"\\nKey: {key}")
            print(f"   Name: {template.name}")
            print(f"   Description: {template.description}")
            print(f"   Estimated tokens: {template.estimated_final_tokens:,}")
            print(f"   Turns: {template.turns}")
            print(f"   Tags: {', '.join(template.tags)}")
            
            if template.rag_config:
                has_initial = template.rag_config.get('has_initial_rag_data', False)
                remove_turn = template.rag_config.get('remove_rag_at_turn', 'N/A')
                print(f"   RAG simulation: Yes (initial data: {has_initial}, remove at turn {remove_turn})")
            else:
                print(f"   RAG simulation: No")
            
            if model_context_size:
                fits = template.fits_in_context(model_context_size)
                status = "Fits" if fits else "Too large"
                print(f"   Context fit ({model_context_size:,} tokens): {status}")
    
    def get_conversation_by_key(self, key: str, model_context_size: Optional[int] = None) -> Optional[Conversation]:
        """Get a specific conversation by key"""
        if key not in self.conversation_templates:
            print(f"Error: Conversation '{key}' not found")
            return None
        
        template = self.conversation_templates[key]
        
        if model_context_size and not template.fits_in_context(model_context_size):
            print(f"Warning: Conversation '{key}' may exceed context limit")
        
        return self._generate_conversation_from_template(template)
    
    def get_conversations_by_tags(self, 
                                 tags: List[str], 
                                 model_context_size: Optional[int] = None) -> List[Conversation]:
        """Get conversations matching any of the specified tags"""
        matching_conversations = []
        
        for key, template in self.conversation_templates.items():
            if any(tag in template.tags for tag in tags):
                if model_context_size and not template.fits_in_context(model_context_size):
                    print(f"Warning: Skipping '{template.name}' - exceeds context limit")
                    continue
                
                conversation = self._generate_conversation_from_template(template)
                matching_conversations.append(conversation)
        
        return matching_conversations
    
    def get_test_suite(self, 
                      model_context_size: int,
                      include_tags: Optional[List[str]] = None,
                      exclude_tags: Optional[List[str]] = None) -> List[Conversation]:
        """Get a complete test suite based on context size and tag filters"""
        
        conversations = []
        
        for key, template in self.conversation_templates.items():
            # Check context size
            if not template.fits_in_context(model_context_size):
                continue
            
            # Check include tags
            if include_tags and not any(tag in template.tags for tag in include_tags):
                continue
            
            # Check exclude tags
            if exclude_tags and any(tag in template.tags for tag in exclude_tags):
                continue
            
            conversation = self._generate_conversation_from_template(template)
            conversations.append(conversation)
        
        return conversations


if __name__ == "__main__":
    # Test the conversation loader
    loader = ConversationLoader()
    
    print("Testing Conversation Loader")
    print("=" * 50)
    
    # List all conversations
    loader.list_available_conversations(model_context_size=8192)
    
    # Get conversations for specific context size
    print(f"\\nConversations that fit in 4K context:")
    conversations_4k = loader.get_conversations_for_context(4096)
    for conv in conversations_4k:
        print(f"   {conv.name} ({conv.estimate_total_tokens()} estimated tokens)")
    
    print(f"\\nConversations that fit in 16K context:")
    conversations_16k = loader.get_conversations_for_context(16384)
    for conv in conversations_16k:
        print(f"   {conv.name} ({conv.estimate_total_tokens()} estimated tokens)")
    
    # Test RAG simulation
    print(f"\\nTarget: Testing RAG simulation:")
    rag_conv = loader.get_conversation_by_key("rag_simulation")
    if rag_conv:
        print(f"   RAG conversation loaded: {rag_conv.name}")
        rag_messages = [msg for msg in rag_conv.messages if msg.message_type == MessageType.RAG_DATA]
        print(f"   RAG messages found: {len(rag_messages)}")
    
    print(f"\\nSuccess: Conversation loader test completed!")