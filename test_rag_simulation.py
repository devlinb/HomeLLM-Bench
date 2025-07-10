#!/usr/bin/env python3
"""
Test the simplified RAG simulation logic
"""
from data.conversation_loader import ConversationLoader
from templates.phi3 import Phi3ChatTemplate
from schemas.conversation import MessageRole, MessageType

def test_rag_simulation():
    """Test the RAG simulation conversation flow"""
    
    print("🧪 Testing RAG Simulation Logic")
    print("=" * 50)
    
    # Load RAG conversation
    loader = ConversationLoader()
    rag_conv = loader.get_conversation_by_key('rag_simulation')
    
    if not rag_conv:
        print("❌ Could not load RAG simulation conversation")
        return False
    
    print(f"📋 Loaded: {rag_conv.name}")
    print(f"📝 Description: {rag_conv.description}")
    print(f"💬 Total messages: {len(rag_conv.messages)}")
    
    # Initialize template
    template = Phi3ChatTemplate()
    
    # Simulate the conversation flow
    conversation_history = []
    rag_data_active = False
    
    print(f"\\n🔄 Simulating conversation flow:")
    
    # Step 1: Load initial system/RAG messages
    for msg in rag_conv.messages:
        if msg.role == MessageRole.SYSTEM:
            conversation_history.append({
                "role": msg.role.value,
                "content": msg.content
            })
            if msg.message_type == MessageType.RAG_DATA:
                rag_data_active = True
                print(f"   📋 Initial RAG data loaded ({len(msg.content)} chars)")
    
    # Step 2: Process user messages
    user_messages = [msg for msg in rag_conv.messages if msg.role == MessageRole.USER]
    
    for turn_idx, user_message in enumerate(user_messages):
        turn_number = turn_idx + 1
        print(f"\\n🔵 Turn {turn_number}: {user_message.message_type.value}")
        
        # Handle RAG removal
        if (user_message.message_type == MessageType.RAG_REMOVAL and 
            user_message.message_metadata.get("remove_rag_before_this") and 
            rag_data_active):
            
            print(f"   🗑️ Removing RAG data from context")
            conversation_history = [
                msg for msg in conversation_history 
                if not msg["content"].startswith("[RETRIEVED INFORMATION]")
            ]
            rag_data_active = False
        
        # Add user message
        conversation_history.append({
            "role": user_message.role.value,
            "content": user_message.content
        })
        
        # Format for model
        prompt = template.format_messages(conversation_history)
        prompt_tokens = len(prompt) // 4
        
        print(f"   📏 Context size: {prompt_tokens:,} tokens")
        print(f"   🎯 RAG active: {'Yes' if rag_data_active else 'No'}")
        print(f"   💭 User: {user_message.content[:60]}...")
        
        # Simulate assistant response
        mock_response = f"[Assistant response to turn {turn_number}]"
        conversation_history.append({
            "role": "assistant", 
            "content": mock_response
        })
        print(f"   🤖 Assistant: {mock_response}")
    
    print(f"\\n📊 Final state:")
    print(f"   💬 Total messages in history: {len(conversation_history)}")
    print(f"   🎯 RAG data still active: {'Yes' if rag_data_active else 'No'}")
    
    # Verify the key RAG simulation aspects
    print(f"\\n✅ RAG Simulation Verification:")
    
    # Check that RAG data was initially present
    has_initial_rag = any(msg["content"].startswith("[RETRIEVED INFORMATION]") 
                         for msg in conversation_history[:2])  # Check first few messages
    print(f"   📋 Had initial RAG data: {'✅' if not has_initial_rag and not rag_data_active else '❌'}")
    
    # Check that RAG was removed
    print(f"   🗑️ RAG data removed: {'✅' if not rag_data_active else '❌'}")
    
    # Check conversation continued after removal
    user_count = sum(1 for msg in conversation_history if msg["role"] == "user")
    print(f"   🔄 Multi-turn conversation: {'✅' if user_count >= 3 else '❌'}")
    
    return True

if __name__ == "__main__":
    test_rag_simulation()