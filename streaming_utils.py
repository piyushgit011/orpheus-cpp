"""
Optimized streaming utilities for Orpheus TTS
Handles intelligent text chunking and efficient streaming
"""

import re
from typing import List, Tuple
import numpy as np

class TextChunker:
    """Smart text chunking for TTS with context limits"""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences using regex (handles abbreviations better)
        """
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|i\.e|e\.g)\.', r'\1<PERIOD>', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def chunk_by_length(text: str, max_chars: int = 150) -> List[str]:
        """
        Chunk text by character length while respecting sentence boundaries
        """
        sentences = TextChunker.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If single sentence exceeds max, split it
            if sentence_length > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence on commas or spaces
                sub_chunks = TextChunker._split_long_sentence(sentence, max_chars)
                chunks.extend(sub_chunks)
            
            # If adding this sentence exceeds max, save current chunk
            elif current_length + sentence_length > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            
            # Otherwise add to current chunk
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
        """Split a long sentence on commas or natural breaks"""
        # Try splitting on commas first
        parts = [p.strip() for p in sentence.split(',')]
        
        chunks = []
        current = []
        current_len = 0
        
        for part in parts:
            if current_len + len(part) > max_chars:
                if current:
                    chunks.append(', '.join(current) + ',')
                current = [part]
                current_len = len(part)
            else:
                current.append(part)
                current_len += len(part) + 2  # +2 for ", "
        
        if current:
            # Last part doesn't need trailing comma
            chunks.append(', '.join(current))
        
        return chunks

class AudioBuffer:
    """Efficient audio buffering for smooth streaming"""
    
    def __init__(self, chunk_size: int = 4096):
        self.chunk_size = chunk_size
        self.buffer = []
    
    def add(self, audio: np.ndarray):
        """Add audio to buffer"""
        self.buffer.append(audio)
    
    def get_chunks(self) -> List[np.ndarray]:
        """Get audio chunks of fixed size"""
        if not self.buffer:
            return []
        
        # Concatenate all buffered audio
        full_audio = np.concatenate(self.buffer)
        self.buffer = []
        
        # Split into chunks
        chunks = []
        for i in range(0, len(full_audio), self.chunk_size):
            chunks.append(full_audio[i:i + self.chunk_size])
        
        return chunks
    
    def has_data(self) -> bool:
        """Check if buffer has data"""
        return len(self.buffer) > 0