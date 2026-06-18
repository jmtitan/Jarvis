"""
Memory data structures and types
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json

class MemoryType(Enum):
    """Types of memory entries"""
    USER_PROFILE = "user_profile"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    PREFERENCE = "preference"
    TASK_HISTORY = "task_history"

@dataclass
class MemoryEntry:
    """Base class for memory entries"""
    id: str
    type: MemoryType
    content: str
    timestamp: datetime
    importance: float = 1.0  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'type': self.type.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            type=MemoryType(data['type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            importance=data.get('importance', 1.0),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None,
            metadata=data.get('metadata', {})
        )

@dataclass
class UserProfile:
    """User profile and preferences"""
    name: Optional[str] = None
    language_preference: str = "en"
    voice_preference: Optional[str] = None
    conversation_style: str = "casual"  # casual, formal, friendly, etc.
    interests: List[str] = field(default_factory=list)
    timezone: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_memory_entry(self) -> MemoryEntry:
        """Convert to memory entry"""
        content = f"User profile: {self.name or 'Unknown'}, prefers {self.language_preference} language"
        if self.voice_preference:
            content += f", voice: {self.voice_preference}"
        content += f", style: {self.conversation_style}"
        if self.interests:
            content += f", interests: {', '.join(self.interests)}"
        
        return MemoryEntry(
            id="user_profile",
            type=MemoryType.USER_PROFILE,
            content=content,
            timestamp=datetime.now(),
            importance=1.0,
            metadata={
                'name': self.name,
                'language_preference': self.language_preference,
                'voice_preference': self.voice_preference,
                'conversation_style': self.conversation_style,
                'interests': self.interests,
                'timezone': self.timezone,
                'preferences': self.preferences
            }
        )

@dataclass
class ConversationMemory:
    """Short-term conversation memory"""
    messages: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "user/assistant", "content": "..."}]
    max_messages: int = 10
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_context(self, max_tokens: int = 500) -> str:
        """Get recent conversation context within token limit"""
        context_parts = []
        current_tokens = 0
        
        # Process messages in reverse order (most recent first)
        for msg in reversed(self.messages):
            msg_text = f"{msg['role']}: {msg['content']}"
            msg_tokens = len(msg_text.split()) * 1.3  # Rough token estimation
            
            if current_tokens + msg_tokens > max_tokens:
                break
                
            context_parts.insert(0, msg_text)
            current_tokens += msg_tokens
        
        return "\n".join(context_parts)

@dataclass 
class LongTermMemory:
    """Long-term memory with summaries"""
    summaries: List[str] = field(default_factory=list)
    important_facts: List[str] = field(default_factory=list)
    user_patterns: List[str] = field(default_factory=list)
    
    def add_summary(self, summary: str):
        """Add a conversation summary"""
        self.summaries.append(summary)
        # Keep only recent summaries to avoid memory bloat
        if len(self.summaries) > 20:
            self.summaries = self.summaries[-20:]
    
    def add_important_fact(self, fact: str):
        """Add an important fact about the user"""
        if fact not in self.important_facts:
            self.important_facts.append(fact)
    
    def add_user_pattern(self, pattern: str):
        """Add a user behavior pattern"""
        if pattern not in self.user_patterns:
            self.user_patterns.append(pattern)
    
    def get_summary_text(self, max_length: int = 300) -> str:
        """Get condensed summary text"""
        all_content = []
        
        if self.important_facts:
            all_content.append("Important facts: " + "; ".join(self.important_facts[:3]))
        
        if self.user_patterns:
            all_content.append("User patterns: " + "; ".join(self.user_patterns[:2]))
        
        if self.summaries:
            all_content.append("Recent activity: " + "; ".join(self.summaries[-2:]))
        
        summary = " | ".join(all_content)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary 