#!/usr/bin/env python3
"""
Demonstrate the Phi-3.5 chat template with specific examples
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from schemas.conversation import Conversation, Message, MessageRole
from templates.phi3 import Phi3ChatTemplate


def main():
    template = Phi3ChatTemplate()
    
    print("=== Phi-3.5 Chat Template Examples ===\n")
    
    # Example 1: Simple Q&A
    print("Example 1: Simple Q&A")
    print("-" * 30)
    conversation1 = Conversation(
        name="Basic Math",
        messages=[
            Message(role=MessageRole.USER, content="What is 5 * 7?"),
        ]
    )
    result1 = template.format_conversation(conversation1)
    print(result1)
    print("\n")
    
    # Example 2: System prompt with conversation
    print("Example 2: System Prompt + Conversation")
    print("-" * 40)
    conversation2 = Conversation(
        name="Coding Assistant",
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a Python programming expert."),
            Message(role=MessageRole.USER, content="How do I create a list in Python?"),
            Message(role=MessageRole.ASSISTANT, content="You can create a list using square brackets: my_list = [1, 2, 3]"),
            Message(role=MessageRole.USER, content="What about an empty list?"),
        ]
    )
    result2 = template.format_conversation(conversation2)
    print(result2)
    print("\n")
    
    # Example 3: Multiline content
    print("Example 3: Multiline Content")
    print("-" * 30)
    conversation3 = Conversation(
        name="Code Review",
        messages=[
            Message(role=MessageRole.USER, content="Review this code:\n\ndef add_numbers(a, b):\n    return a + b\n\nprint(add_numbers(2, 3))"),
            Message(role=MessageRole.ASSISTANT, content="The code looks good! Here's my review:\n\n✅ Function is well-named\n✅ Simple and clear logic\n✅ Proper usage example\n\nNo issues found."),
            Message(role=MessageRole.USER, content="What about error handling?"),
        ]
    )
    result3 = template.format_conversation(conversation3)
    print(result3)
    print("\n")
    
    print("=== All examples formatted correctly! ===")


if __name__ == "__main__":
    main()