"""
Vector storage and semantic retrieval for memory system
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json
import os
from pathlib import Path
import hashlib

try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS library available for vector search")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using simple cosine similarity for vector search")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    print("SentenceTransformers library available for embeddings")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available, using simple TF-IDF for embeddings")

from .memory_types import MemoryEntry

class VectorStore:
    """Vector storage for semantic memory retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_dir = Path("memory_data")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Vector storage
        self.vectors = []
        self.memory_entries = []
        self.vector_to_memory_map = {}
        
        # Initialize embedding model
        self.embedding_model = None
        self.vector_dim = 384  # Default dimension
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
                print(f"Loaded SentenceTransformer model with dimension {self.vector_dim}")
            except Exception as e:
                print(f"Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        
        # Initialize FAISS index
        self.faiss_index = None
        if FAISS_AVAILABLE and self.embedding_model:
            try:
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine similarity
                print("FAISS index initialized")
            except Exception as e:
                print(f"Failed to initialize FAISS: {e}")
        
        # Load existing data
        self.load_memory_data()
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text"""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, normalize_embeddings=True)
                return embedding
            except Exception as e:
                print(f"Error getting embedding: {e}")
        
        # Fallback: simple TF-IDF style representation
        return self._simple_text_embedding(text)
    
    def _simple_text_embedding(self, text: str) -> np.ndarray:
        """Simple fallback embedding based on character frequencies"""
        # Create a simple vector based on character frequencies and word patterns
        text_lower = text.lower()
        
        # Create a fixed-size vector based on various text features
        features = np.zeros(self.vector_dim)
        
        # Character frequency features (first 26 dimensions for a-z)
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
            if i < self.vector_dim:
                features[i] = text_lower.count(char) / max(len(text_lower), 1)
        
        # Word length distribution (next 10 dimensions)
        words = text_lower.split()
        for i, length in enumerate(range(1, 11)):
            if 26 + i < self.vector_dim:
                count = sum(1 for word in words if len(word) == length)
                features[26 + i] = count / max(len(words), 1)
        
        # Normalize the vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def add_memory(self, memory_entry: MemoryEntry) -> bool:
        """Add a memory entry to the vector store"""
        try:
            # Generate embedding for the memory content
            embedding = self._get_text_embedding(memory_entry.content)
            
            # Add to storage
            entry_id = len(self.memory_entries)
            self.memory_entries.append(memory_entry)
            self.vectors.append(embedding)
            self.vector_to_memory_map[entry_id] = len(self.memory_entries) - 1
            
            # Add to FAISS index if available
            if self.faiss_index is not None:
                self.faiss_index.add(embedding.reshape(1, -1))
            
            print(f"Added memory entry: {memory_entry.id}")
            return True
            
        except Exception as e:
            print(f"Error adding memory entry: {e}")
            return False
    
    def search_similar_memories(self, query_text: str, top_k: int = 5, 
                              memory_types: Optional[List[str]] = None) -> List[Tuple[MemoryEntry, float]]:
        """Search for memories similar to query text"""
        if not self.memory_entries:
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_text_embedding(query_text)
            
            # Search using FAISS if available
            if self.faiss_index is not None and len(self.vectors) > 0:
                scores, indices = self.faiss_index.search(query_embedding.reshape(1, -1), 
                                                        min(top_k * 2, len(self.vectors)))
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx >= 0 and idx < len(self.memory_entries):
                        memory = self.memory_entries[idx]
                        
                        # Filter by memory types if specified
                        if memory_types and memory.type.value not in memory_types:
                            continue
                        
                        results.append((memory, float(score)))
                        
                        if len(results) >= top_k:
                            break
                
                return results
            
            # Fallback: manual cosine similarity calculation
            else:
                similarities = []
                for i, vector in enumerate(self.vectors):
                    similarity = np.dot(query_embedding, vector)
                    similarities.append((similarity, i))
                
                # Sort by similarity (descending)
                similarities.sort(key=lambda x: x[0], reverse=True)
                
                results = []
                for similarity, idx in similarities[:top_k]:
                    memory = self.memory_entries[idx]
                    
                    # Filter by memory types if specified
                    if memory_types and memory.type.value not in memory_types:
                        continue
                    
                    results.append((memory, similarity))
                
                return results[:top_k]
                
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []
    
    def get_recent_memories(self, memory_types: Optional[List[str]] = None, 
                          max_entries: int = 10) -> List[MemoryEntry]:
        """Get recent memories, optionally filtered by type"""
        filtered_memories = []
        
        for memory in self.memory_entries:
            if memory_types and memory.type.value not in memory_types:
                continue
            filtered_memories.append(memory)
        
        # Sort by timestamp (most recent first)
        filtered_memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_memories[:max_entries]
    
    def update_memory_access(self, memory_id: str):
        """Update access statistics for a memory"""
        for memory in self.memory_entries:
            if memory.id == memory_id:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                break
    
    def save_memory_data(self):
        """Save memory data to disk"""
        try:
            # Save memory entries
            entries_file = self.memory_dir / "memory_entries.json"
            entries_data = [entry.to_dict() for entry in self.memory_entries]
            
            with open(entries_file, 'w', encoding='utf-8') as f:
                json.dump(entries_data, f, ensure_ascii=False, indent=2)
            
            # Save vectors
            if self.vectors:
                vectors_file = self.memory_dir / "vectors.npy"
                np.save(vectors_file, np.array(self.vectors))
            
            print(f"Saved {len(self.memory_entries)} memory entries")
            
        except Exception as e:
            print(f"Error saving memory data: {e}")
    
    def load_memory_data(self):
        """Load memory data from disk"""
        try:
            entries_file = self.memory_dir / "memory_entries.json"
            vectors_file = self.memory_dir / "vectors.npy"
            
            # Load memory entries
            if entries_file.exists():
                with open(entries_file, 'r', encoding='utf-8') as f:
                    entries_data = json.load(f)
                
                self.memory_entries = [MemoryEntry.from_dict(data) for data in entries_data]
                print(f"Loaded {len(self.memory_entries)} memory entries")
            
            # Load vectors
            if vectors_file.exists() and self.memory_entries:
                self.vectors = np.load(vectors_file).tolist()
                
                # Rebuild FAISS index if available
                if self.faiss_index is not None and self.vectors:
                    vectors_array = np.array(self.vectors)
                    self.faiss_index.add(vectors_array)
                    print("Rebuilt FAISS index")
                
                # Rebuild vector to memory map
                self.vector_to_memory_map = {i: i for i in range(len(self.vectors))}
                
        except Exception as e:
            print(f"Error loading memory data: {e}")
    
    def cleanup_old_memories(self, max_age_days: int = 30):
        """Remove old, low-importance memories"""
        try:
            cutoff_date = datetime.now() - datetime.timedelta(days=max_age_days)
            
            # Find memories to remove
            indices_to_remove = []
            for i, memory in enumerate(self.memory_entries):
                if (memory.timestamp < cutoff_date and 
                    memory.importance < 0.5 and 
                    memory.access_count < 2):
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                del self.memory_entries[i]
                if i < len(self.vectors):
                    del self.vectors[i]
            
            # Rebuild FAISS index if we removed items
            if indices_to_remove and self.faiss_index is not None:
                self.faiss_index.reset()
                if self.vectors:
                    vectors_array = np.array(self.vectors)
                    self.faiss_index.add(vectors_array)
            
            if indices_to_remove:
                print(f"Cleaned up {len(indices_to_remove)} old memories")
                
        except Exception as e:
            print(f"Error during memory cleanup: {e}") 