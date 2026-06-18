"""
Memory System Demo for Jarvis Assistant

This demo shows how the memory system works and its capabilities.
Run this file independently to test memory functionality.
"""

import asyncio
import yaml
from memory_types import UserProfile, MemoryType
from memory_manager import MemoryManager

class MockLLMClient:
    """Mock LLM client for testing memory system"""
    
    async def generate(self, prompt):
        """Simple mock responses for testing"""
        if "summary" in prompt.lower():
            return "User discussed preferences for voice settings and asked about weather information."
        elif "facts" in prompt.lower():
            return "- User prefers female voice\n- User is interested in weather updates\n- User likes conversational style"
        else:
            return "This is a mock response from the LLM for testing purposes."

async def demo_memory_system():
    """Demonstrate memory system functionality"""
    print("🧠 Memory System Demo")
    print("=" * 50)
    
    # Load configuration
    try:
        with open('../config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except:
        # Fallback config for demo
        config = {
            'memory': {
                'enabled': True,
                'max_context_tokens': 2000,
                'user_profile_tokens': 200,
                'history_tokens': 500,
                'long_term_tokens': 300,
                'conversation_tokens': 500,
                'max_conversation_messages': 10,
                'auto_summarize_threshold': 6,  # Lower for demo
                'fact_extraction_threshold': 3,  # Lower for demo
                'cleanup_days': 30
            }
        }
    
    # Initialize memory manager
    print("1. Initializing Memory Manager...")
    memory_manager = MemoryManager(config)
    
    # Set mock LLM client
    mock_llm = MockLLMClient()
    memory_manager.set_llm_client(mock_llm)
    
    # Demo 1: Update user profile
    print("\n2. Updating User Profile...")
    memory_manager.update_user_profile(
        name="Demo User",
        language_preference="en",
        conversation_style="friendly",
        voice_preference="en-US-AriaNeural"
    )
    memory_manager.add_user_interest("technology")
    memory_manager.add_user_interest("AI")
    
    # Demo 2: Process some interactions
    print("\n3. Processing Sample Interactions...")
    interactions = [
        ("What's the weather like today?", "Today is sunny with a temperature of 22°C."),
        ("I prefer female voices for TTS", "I've noted that you prefer female voices. I'll remember that for future conversations."),
        ("Can you remind me about my meetings?", "I don't have access to your calendar, but I can help you set reminders."),
        ("I like having conversations in a casual style", "Got it! I'll keep our conversations casual and friendly."),
        ("What's my name again?", "Based on our conversation, you're Demo User."),
        ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!")
    ]
    
    for user_input, ai_response in interactions:
        print(f"   User: {user_input}")
        print(f"   AI: {ai_response}")
        await memory_manager.process_interaction(user_input, ai_response)
        print()
    
    # Demo 3: Show memory statistics
    print("\n4. Memory Statistics:")
    stats = memory_manager.get_memory_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # Demo 4: Search memories
    print("\n5. Searching Memories...")
    search_queries = ["voice preferences", "weather", "meetings"]
    
    for query in search_queries:
        print(f"   Query: '{query}'")
        results = memory_manager.search_memories(query, max_results=3)
        for i, (memory, score) in enumerate(results):
            print(f"     {i+1}. [{memory.type.value}] {memory.content[:60]}... (score: {score:.3f})")
        print()
    
    # Demo 5: Build enhanced prompt
    print("\n6. Enhanced Prompt Generation...")
    test_input = "What do you remember about my preferences?"
    enhanced_prompt = await memory_manager.build_enhanced_prompt(test_input)
    
    print(f"   Input: {test_input}")
    print(f"   Enhanced Prompt Length: {len(enhanced_prompt)} characters")
    print(f"   Enhanced Prompt Preview:")
    print("   " + "─" * 60)
    # Show first 300 characters
    preview = enhanced_prompt[:300] + "..." if len(enhanced_prompt) > 300 else enhanced_prompt
    for line in preview.split('\n'):
        print(f"   {line}")
    print("   " + "─" * 60)
    
    # Demo 6: Show user profile
    print("\n7. Current User Profile:")
    profile = memory_manager.user_profile
    print(f"   Name: {profile.name}")
    print(f"   Language: {profile.language_preference}")
    print(f"   Style: {profile.conversation_style}")
    print(f"   Voice: {profile.voice_preference}")
    print(f"   Interests: {', '.join(profile.interests)}")
    
    # Demo 7: Long-term memory
    print("\n8. Long-term Memory Content:")
    lt_memory = memory_manager.long_term_memory
    print(f"   Summaries ({len(lt_memory.summaries)}):")
    for i, summary in enumerate(lt_memory.summaries, 1):
        print(f"     {i}. {summary}")
    
    print(f"   Important Facts ({len(lt_memory.important_facts)}):")
    for i, fact in enumerate(lt_memory.important_facts, 1):
        print(f"     {i}. {fact}")
    
    # Cleanup
    print("\n9. Saving Memory Data...")
    memory_manager.save_persistent_data()
    
    print("\n✅ Demo completed! Memory data has been saved to memory_data/ directory.")
    print("🔍 Check the generated files to see how memories are stored.")

if __name__ == "__main__":
    print("Starting Memory System Demo...")
    print("Note: This demo uses mock LLM responses for testing.")
    print()
    
    try:
        asyncio.run(demo_memory_system())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc() 