"""
Structured prompt builder for memory-enhanced conversations
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from .memory_types import MemoryEntry, UserProfile, ConversationMemory, LongTermMemory
from .vector_store import VectorStore

class PromptBuilder:
    """Build structured prompts with memory context"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_total_tokens = config.get('memory', {}).get('max_context_tokens', 2000)
        self.user_profile_tokens = config.get('memory', {}).get('user_profile_tokens', 200)
        self.history_tokens = config.get('memory', {}).get('history_tokens', 500)
        self.long_term_tokens = config.get('memory', {}).get('long_term_tokens', 300)
        self.conversation_tokens = config.get('memory', {}).get('conversation_tokens', 500)
        
    def build_enhanced_prompt(self, 
                            user_input: str,
                            user_profile: Optional[UserProfile] = None,
                            conversation_memory: Optional[ConversationMemory] = None,
                            long_term_memory: Optional[LongTermMemory] = None,
                            relevant_memories: Optional[List[Tuple[MemoryEntry, float]]] = None) -> str:
        """Build a structured prompt with all memory context"""
        
        prompt_parts = []
        
        # System prompt with memory awareness
        system_prompt = self._build_system_prompt()
        prompt_parts.append(system_prompt)
        
        # User profile section
        if user_profile:
            profile_section = self._build_user_profile_section(user_profile)
            if profile_section:
                prompt_parts.append(profile_section)
        
        # Long-term memory summary
        if long_term_memory:
            long_term_section = self._build_long_term_section(long_term_memory)
            if long_term_section:
                prompt_parts.append(long_term_section)
        
        # Relevant historical memories
        if relevant_memories:
            relevant_section = self._build_relevant_memories_section(relevant_memories)
            if relevant_section:
                prompt_parts.append(relevant_section)
        
        # Recent conversation context
        if conversation_memory:
            conversation_section = self._build_conversation_section(conversation_memory)
            if conversation_section:
                prompt_parts.append(conversation_section)
        
        # Current user input
        current_input_section = self._build_current_input_section(user_input)
        prompt_parts.append(current_input_section)
        
        # Join all sections
        full_prompt = "\n\n".join(prompt_parts)
        
        # Ensure token limit compliance
        full_prompt = self._enforce_token_limit(full_prompt)
        
        return full_prompt
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt"""
        return """You are Jarvis, an intelligent AI assistant. You have access to information about the user's profile, conversation history, and relevant memories. Use this context to provide personalized, helpful responses that consider the user's preferences and previous interactions.

Guidelines:
- Reference relevant information from the user's profile and history when appropriate
- Maintain consistency with previous conversations
- Adapt your communication style to match the user's preferences
- Be helpful, accurate, and engaging"""
    
    def _build_user_profile_section(self, user_profile: UserProfile) -> str:
        """Build user profile section"""
        if not user_profile:
            return ""
        
        profile_info = []
        
        if user_profile.name:
            profile_info.append(f"Name: {user_profile.name}")
        
        profile_info.append(f"Language preference: {user_profile.language_preference}")
        profile_info.append(f"Communication style: {user_profile.conversation_style}")
        
        if user_profile.voice_preference:
            profile_info.append(f"Voice preference: {user_profile.voice_preference}")
        
        if user_profile.interests:
            profile_info.append(f"Interests: {', '.join(user_profile.interests)}")
        
        if user_profile.timezone:
            profile_info.append(f"Timezone: {user_profile.timezone}")
        
        # Add custom preferences
        if user_profile.preferences:
            for key, value in user_profile.preferences.items():
                profile_info.append(f"{key}: {value}")
        
        if profile_info:
            section = "[USER PROFILE]\n" + "\n".join(profile_info)
            return self._truncate_section(section, self.user_profile_tokens)
        
        return ""
    
    def _build_long_term_section(self, long_term_memory: LongTermMemory) -> str:
        """Build long-term memory section"""
        summary_text = long_term_memory.get_summary_text(max_length=self.long_term_tokens)
        
        if summary_text:
            section = f"[LONG-TERM CONTEXT]\n{summary_text}"
            return self._truncate_section(section, self.long_term_tokens)
        
        return ""
    
    def _build_relevant_memories_section(self, relevant_memories: List[Tuple[MemoryEntry, float]]) -> str:
        """Build relevant memories section"""
        if not relevant_memories:
            return ""
        
        memory_texts = []
        current_length = 0
        max_length = self.history_tokens
        
        for memory_entry, relevance_score in relevant_memories:
            # Only include memories above a relevance threshold
            if relevance_score < 0.3:
                continue
            
            memory_text = f"• {memory_entry.content}"
            memory_length = len(memory_text.split()) * 1.3  # Rough token estimation
            
            if current_length + memory_length > max_length:
                break
            
            memory_texts.append(memory_text)
            current_length += memory_length
        
        if memory_texts:
            section = "[RELEVANT HISTORY]\n" + "\n".join(memory_texts)
            return self._truncate_section(section, self.history_tokens)
        
        return ""
    
    def _build_conversation_section(self, conversation_memory: ConversationMemory) -> str:
        """Build recent conversation section"""
        recent_context = conversation_memory.get_recent_context(max_tokens=self.conversation_tokens)
        
        if recent_context:
            section = f"[RECENT CONVERSATION]\n{recent_context}"
            return self._truncate_section(section, self.conversation_tokens)
        
        return ""
    
    def _build_current_input_section(self, user_input: str) -> str:
        """Build current user input section"""
        return f"[CURRENT REQUEST]\nuser: {user_input}"
    
    def _truncate_section(self, section: str, max_tokens: int) -> str:
        """Truncate a section to fit within token limit"""
        words = section.split()
        estimated_tokens = len(words) * 1.3
        
        if estimated_tokens <= max_tokens:
            return section
        
        # Calculate how many words to keep
        target_words = int(max_tokens / 1.3)
        truncated_words = words[:target_words]
        
        # Add truncation indicator
        truncated_section = " ".join(truncated_words)
        if len(words) > target_words:
            truncated_section += "..."
        
        return truncated_section
    
    def _enforce_token_limit(self, prompt: str) -> str:
        """Ensure the entire prompt is within token limits"""
        words = prompt.split()
        estimated_tokens = len(words) * 1.3
        
        if estimated_tokens <= self.max_total_tokens:
            return prompt
        
        # If too long, truncate while preserving structure
        target_words = int(self.max_total_tokens / 1.3)
        
        # Try to preserve the current request section
        lines = prompt.split('\n')
        current_request_start = -1
        
        for i, line in enumerate(lines):
            if '[CURRENT REQUEST]' in line:
                current_request_start = i
                break
        
        if current_request_start != -1:
            # Keep current request and work backwards
            current_request_section = '\n'.join(lines[current_request_start:])
            available_words = target_words - len(current_request_section.split())
            
            if available_words > 50:  # Ensure some context remains
                context_lines = lines[:current_request_start]
                context_text = '\n'.join(context_lines)
                context_words = context_text.split()
                
                if len(context_words) > available_words:
                    context_words = context_words[:available_words]
                    context_text = ' '.join(context_words) + "..."
                
                return context_text + '\n\n' + current_request_section
        
        # Fallback: simple truncation
        truncated_words = words[:target_words]
        return ' '.join(truncated_words) + "..."
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for text"""
        return int(len(text.split()) * 1.3)
    
    def build_summary_prompt(self, conversation_history: str) -> str:
        """Build a prompt for conversation summarization"""
        return f"""Please provide a concise summary of the following conversation, focusing on:
1. Key topics discussed
2. Important user preferences or information revealed
3. Decisions made or actions taken
4. Any ongoing tasks or follow-ups needed

Conversation:
{conversation_history}

Summary:"""
    
    def build_fact_extraction_prompt(self, conversation_history: str) -> str:
        """Build a prompt for extracting important facts about the user"""
        return f"""Extract important facts about the user from this conversation. Focus on:
- Personal preferences (voice, language, style)
- Interests and hobbies
- Work or professional information
- Personal details they've shared
- Behavioral patterns or habits

Provide each fact as a short, clear statement.

Conversation:
{conversation_history}

Important facts:""" 