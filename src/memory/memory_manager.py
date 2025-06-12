"""
Main memory management system for Jarvis Assistant
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import json
import os
from pathlib import Path

from .memory_types import (
    MemoryType, MemoryEntry, UserProfile, 
    ConversationMemory, LongTermMemory
)
from .vector_store import VectorStore
from .prompt_builder import PromptBuilder

class MemoryManager:
    """Main memory management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_config = config.get('memory', {})
        
        # Initialize components
        self.vector_store = VectorStore(config)
        self.prompt_builder = PromptBuilder(config)
        
        # Memory state
        self.user_profile = UserProfile()
        self.conversation_memory = ConversationMemory(
            max_messages=self.memory_config.get('max_conversation_messages', 10)
        )
        self.long_term_memory = LongTermMemory()
        
        # Configuration
        self.auto_summarize_threshold = self.memory_config.get('auto_summarize_threshold', 20)
        self.fact_extraction_threshold = self.memory_config.get('fact_extraction_threshold', 5)
        self.memory_cleanup_days = self.memory_config.get('cleanup_days', 30)
        
        # LLM client reference (will be set externally)
        self.llm_client = None
        
        # Load existing data
        self.load_persistent_data()
        
        print(f"Memory manager initialized with {len(self.vector_store.memory_entries)} stored memories")
    
    def set_llm_client(self, llm_client):
        """Set the LLM client for generating summaries and extracting facts"""
        self.llm_client = llm_client
    
    async def process_interaction(self, user_input: str, ai_response: str) -> str:
        """Process a complete user-AI interaction and return enhanced prompt for next interaction"""
        
        # Add to conversation memory
        self.conversation_memory.add_message("user", user_input)
        self.conversation_memory.add_message("assistant", ai_response)
        
        # Create memory entries for long-term storage
        await self._store_interaction_memory(user_input, ai_response)
        
        # Check if we should generate summaries or extract facts
        if len(self.conversation_memory.messages) >= self.auto_summarize_threshold:
            await self._auto_summarize_conversation()
        
        # Re-enable fact extraction but with higher threshold to reduce overhead
        if len(self.conversation_memory.messages) >= 10:  # Increased from 5 to 10
            await self._extract_user_facts()
        
        # Save updated data
        self.save_persistent_data()
        
        return "Interaction processed and stored in memory"
    
    async def build_enhanced_prompt(self, user_input: str) -> str:
        """Build an enhanced prompt with memory context for the given user input"""
        
        # Search for relevant memories
        relevant_memories = self.vector_store.search_similar_memories(
            user_input, 
            top_k=5,
            memory_types=[
                MemoryType.CONVERSATION.value,
                MemoryType.LONG_TERM.value,
                MemoryType.TASK_HISTORY.value
            ]
        )
        
        # Update access statistics for retrieved memories
        for memory_entry, score in relevant_memories:
            self.vector_store.update_memory_access(memory_entry.id)
        
        # Build the enhanced prompt
        enhanced_prompt = self.prompt_builder.build_enhanced_prompt(
            user_input=user_input,
            user_profile=self.user_profile,
            conversation_memory=self.conversation_memory,
            long_term_memory=self.long_term_memory,
            relevant_memories=relevant_memories
        )
        
        return enhanced_prompt
    
    async def _store_interaction_memory(self, user_input: str, ai_response: str):
        """Store interaction in long-term memory"""
        try:
            # Create memory entry for the interaction
            interaction_id = str(uuid.uuid4())
            interaction_content = f"User: {user_input}\nAssistant: {ai_response}"
            
            memory_entry = MemoryEntry(
                id=interaction_id,
                type=MemoryType.CONVERSATION,
                content=interaction_content,
                timestamp=datetime.now(),
                importance=self._calculate_importance(user_input, ai_response),
                metadata={
                    'user_input': user_input,
                    'ai_response': ai_response,
                    'interaction_type': 'conversation'
                }
            )
            
            # Add to vector store
            self.vector_store.add_memory(memory_entry)
            
        except Exception as e:
            print(f"Error storing interaction memory: {e}")
    
    def _calculate_importance(self, user_input: str, ai_response: str) -> float:
        """Calculate importance score for an interaction"""
        importance = 0.5  # Base importance
        
        # Boost importance for longer interactions
        total_length = len(user_input) + len(ai_response)
        if total_length > 200:
            importance += 0.2
        
        # Boost for certain keywords that indicate important information
        important_keywords = [
            'prefer', 'like', 'dislike', 'name', 'remember', 'important',
            'always', 'never', 'favorite', 'hate', 'love', 'want', 'need'
        ]
        
        combined_text = (user_input + " " + ai_response).lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in combined_text)
        importance += min(keyword_count * 0.1, 0.3)
        
        # Boost for questions about personal information
        if any(word in user_input.lower() for word in ['who', 'what', 'where', 'when', 'how', 'why']):
            importance += 0.1
        
        return min(importance, 1.0)
    
    async def _auto_summarize_conversation(self):
        """Automatically summarize conversation when it gets too long"""
        if not self.llm_client:
            print("Cannot summarize conversation: LLM client not available")
            return
        
        try:
            # Get conversation history for summarization
            conversation_text = self.conversation_memory.get_recent_context(max_tokens=1000)
            
            if not conversation_text:
                return
            
            # Generate summary
            summary_prompt = self.prompt_builder.build_summary_prompt(conversation_text)
            summary = await self.llm_client.generate(summary_prompt)
            
            if summary:
                self.long_term_memory.add_summary(summary)
                
                # Create a memory entry for the summary
                summary_id = str(uuid.uuid4())
                summary_entry = MemoryEntry(
                    id=summary_id,
                    type=MemoryType.LONG_TERM,
                    content=f"Conversation summary: {summary}",
                    timestamp=datetime.now(),
                    importance=0.8,
                    metadata={
                        'type': 'conversation_summary',
                        'message_count': len(self.conversation_memory.messages)
                    }
                )
                
                self.vector_store.add_memory(summary_entry)
                
                # Clear older conversation messages to prevent memory bloat
                if len(self.conversation_memory.messages) > self.conversation_memory.max_messages:
                    self.conversation_memory.messages = self.conversation_memory.messages[-5:]
                
                print(f"Generated conversation summary: {summary[:100]}...")
                
        except Exception as e:
            print(f"Error during auto-summarization: {e}")
    
    async def _extract_user_facts(self):
        """Extract important facts about the user from conversation"""
        if not self.llm_client:
            print("Cannot extract user facts: LLM client not available")
            return
        
        try:
            # Get recent conversation for fact extraction
            conversation_text = self.conversation_memory.get_recent_context(max_tokens=800)
            
            if not conversation_text:
                return
            
            # Extract facts
            fact_prompt = self.prompt_builder.build_fact_extraction_prompt(conversation_text)
            facts_response = await self.llm_client.generate(fact_prompt)
            
            if facts_response:
                # Parse facts (assume they're returned as separate lines)
                facts = [fact.strip() for fact in facts_response.split('\n') if fact.strip()]
                
                # Filter and combine facts to avoid too many individual entries
                substantial_facts = []
                for fact in facts:
                    if len(fact) > 15 and not any(existing in fact.lower() for existing in [
                        'blank_audio', 'test', 'example', 'placeholder'  # Filter out noise
                    ]):
                        substantial_facts.append(fact)
                
                # Only create one combined memory entry for all facts from this extraction
                if substantial_facts:
                    combined_facts = "; ".join(substantial_facts[:3])  # Limit to top 3 facts
                    self.long_term_memory.add_important_fact(combined_facts)
                    
                    # Create single memory entry for combined facts
                    fact_id = str(uuid.uuid4())
                    fact_entry = MemoryEntry(
                        id=fact_id,
                        type=MemoryType.PREFERENCE,
                        content=f"User facts: {combined_facts}",
                        timestamp=datetime.now(),
                        importance=0.9,
                        metadata={
                            'type': 'user_facts_batch',
                            'extracted_from': 'conversation',
                            'fact_count': len(substantial_facts)
                        }
                    )
                    
                    self.vector_store.add_memory(fact_entry)
                    print(f"Extracted and stored {len(substantial_facts)} user facts in single entry")
                
        except Exception as e:
            print(f"Error during fact extraction: {e}")
    
    def update_user_profile(self, **kwargs):
        """Update user profile information"""
        updated = False
        
        for key, value in kwargs.items():
            if hasattr(self.user_profile, key):
                setattr(self.user_profile, key, value)
                updated = True
                print(f"Updated user profile: {key} = {value}")
        
        if updated:
            # Store updated profile as memory entry
            profile_entry = self.user_profile.to_memory_entry()
            
            # Remove old profile entry if exists
            self.vector_store.memory_entries = [
                entry for entry in self.vector_store.memory_entries 
                if entry.id != "user_profile"
            ]
            
            # Add updated profile
            self.vector_store.add_memory(profile_entry)
            self.save_persistent_data()
    
    def add_user_interest(self, interest: str):
        """Add a new interest to user profile"""
        if interest not in self.user_profile.interests:
            self.user_profile.interests.append(interest)
            self.update_user_profile()  # This will trigger a save
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'total_memories': len(self.vector_store.memory_entries),
            'conversation_messages': len(self.conversation_memory.messages),
            'long_term_summaries': len(self.long_term_memory.summaries),
            'important_facts': len(self.long_term_memory.important_facts),
            'user_patterns': len(self.long_term_memory.user_patterns),
            'user_interests': len(self.user_profile.interests),
            'memory_types': {
                memory_type.value: sum(1 for entry in self.vector_store.memory_entries 
                                     if entry.type == memory_type)
                for memory_type in MemoryType
            }
        }
    
    def search_memories(self, query: str, max_results: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """Search memories by query"""
        return self.vector_store.search_similar_memories(query, top_k=max_results)
    
    def cleanup_old_memories(self):
        """Clean up old, unimportant memories"""
        self.vector_store.cleanup_old_memories(max_age_days=self.memory_cleanup_days)
        self.save_persistent_data()
    
    def save_persistent_data(self):
        """Save all persistent memory data"""
        try:
            memory_dir = Path("memory_data")
            memory_dir.mkdir(exist_ok=True)
            
            # Save user profile
            profile_file = memory_dir / "user_profile.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'name': self.user_profile.name,
                    'language_preference': self.user_profile.language_preference,
                    'voice_preference': self.user_profile.voice_preference,
                    'conversation_style': self.user_profile.conversation_style,
                    'interests': self.user_profile.interests,
                    'timezone': self.user_profile.timezone,
                    'preferences': self.user_profile.preferences
                }, f, ensure_ascii=False, indent=2)
            
            # Save conversation memory
            conversation_file = memory_dir / "conversation_memory.json"
            with open(conversation_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'messages': self.conversation_memory.messages,
                    'max_messages': self.conversation_memory.max_messages
                }, f, ensure_ascii=False, indent=2)
            
            # Save long-term memory
            longterm_file = memory_dir / "long_term_memory.json"
            with open(longterm_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summaries': self.long_term_memory.summaries,
                    'important_facts': self.long_term_memory.important_facts,
                    'user_patterns': self.long_term_memory.user_patterns
                }, f, ensure_ascii=False, indent=2)
            
            # Save vector store data
            self.vector_store.save_memory_data()
            
        except Exception as e:
            print(f"Error saving persistent data: {e}")
    
    def load_persistent_data(self):
        """Load all persistent memory data"""
        try:
            memory_dir = Path("memory_data")
            
            # Load user profile
            profile_file = memory_dir / "user_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                    self.user_profile = UserProfile(**profile_data)
                    print("Loaded user profile")
            
            # Load conversation memory
            conversation_file = memory_dir / "conversation_memory.json"
            if conversation_file.exists():
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    conv_data = json.load(f)
                    self.conversation_memory = ConversationMemory(
                        messages=conv_data.get('messages', []),
                        max_messages=conv_data.get('max_messages', 10)
                    )
                    print(f"Loaded {len(self.conversation_memory.messages)} conversation messages")
            
            # Load long-term memory
            longterm_file = memory_dir / "long_term_memory.json"
            if longterm_file.exists():
                with open(longterm_file, 'r', encoding='utf-8') as f:
                    lt_data = json.load(f)
                    self.long_term_memory = LongTermMemory(
                        summaries=lt_data.get('summaries', []),
                        important_facts=lt_data.get('important_facts', []),
                        user_patterns=lt_data.get('user_patterns', [])
                    )
                    print(f"Loaded long-term memory with {len(self.long_term_memory.summaries)} summaries")
            
        except Exception as e:
            print(f"Error loading persistent data: {e}")
    
    async def shutdown(self):
        """Shutdown memory manager gracefully"""
        print("Shutting down memory manager...")
        
        # Save all data before shutdown
        self.save_persistent_data()
        
        # Cleanup if needed
        if len(self.vector_store.memory_entries) > 1000:  # If too many memories
            self.cleanup_old_memories()
        
        print("Memory manager shutdown complete") 