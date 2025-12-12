#!/usr/bin/env python3
"""
Quick test script to debug LLM response issues
"""

import os
from dotenv import load_dotenv
from src.config import get_settings
from src.chat import initialize_chain, chat_response

load_dotenv()

def test_question():
    """Test with the problematic question"""
    
    print("="*80)
    print("Testing LLM Response Quality")
    print("="*80)
    
    # Initialize
    print("\n1. Initializing chain...")
    if not initialize_chain():
        print("❌ Failed to initialize chain")
        return
    
    print("✅ Chain initialized\n")
    
    # Test question
    test_q = "Comment est calculé le débit Upload dans la nouvelle offre Idoom Fibre Gamers ?"
    print(f"2. Testing question: {test_q}\n")
    
    # Get response
    history = []
    result = chat_response(test_q, history)
    
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    if result and len(result) > 0:
        user_msg, bot_msg = result[-1]
        print(f"Question: {user_msg}")
        print(f"\nAnswer:\n{bot_msg}")
    else:
        print("❌ No response received")
    
    print("="*80)

if __name__ == "__main__":
    test_question()
