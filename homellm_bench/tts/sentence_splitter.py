#!/usr/bin/env python3
"""
Sentence Boundary Detector for TTS Streaming

This module provides intelligent sentence boundary detection for breaking
streaming LLM output into TTS-friendly chunks. It handles various punctuation
marks, edge cases, and provides configurable chunking strategies.
"""

import re
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


# Constants
DEFAULT_MIN_CHUNK_SIZE = 10
DEFAULT_MAX_CHUNK_SIZE = 200
CONTEXT_WINDOW_SIZE = 20
BOUNDARY_PROXIMITY_THRESHOLD = 2
CLAUSE_CONFIDENCE_MULTIPLIER = 0.7
BASE_CONFIDENCE = 0.8
CONFIDENCE_BOOST_CAPITAL = 0.1
CONFIDENCE_BOOST_STRONG_PUNCT = 0.1
CONFIDENCE_PENALTY_SHORT = 0.2


class BoundaryType(Enum):
    """Types of sentence boundaries"""
    SENTENCE_END = "sentence_end"          # . ! ?
    CLAUSE_BREAK = "clause_break"          # ; : , --
    PARAGRAPH_BREAK = "paragraph_break"    # \n\n
    FORCED_BREAK = "forced_break"          # Manual break point
    ELLIPSIS = "ellipsis"                  # ...


@dataclass
class SentenceBoundary:
    """Represents a detected sentence boundary"""
    position: int                    # Character position in text
    boundary_type: BoundaryType      # Type of boundary
    confidence: float                # Confidence score (0.0-1.0)
    punctuation: str                 # The punctuation mark(s)
    
    # Removed text_before and text_after to save memory
    # These can be computed on-demand if needed


class SentenceSplitter:
    """
    Intelligent sentence boundary detector for TTS streaming.
    
    Features:
    - Handles multiple punctuation types: . ! ? ; : , -- ...
    - Avoids false positives with abbreviations, decimals, URLs
    - Configurable minimum chunk size
    - Confidence scoring for boundary quality
    - Multiple chunking strategies
    """
    
    def __init__(self, 
                 min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
                 max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
                 prefer_sentence_ends: bool = True,
                 include_clause_breaks: bool = True):
        """
        Initialize the sentence splitter.
        
        Args:
            min_chunk_size: Minimum characters before considering a break
            max_chunk_size: Maximum characters before forcing a break
            prefer_sentence_ends: Prefer sentence endings over clause breaks
            include_clause_breaks: Allow breaking on semicolons, commas, etc.
            
        Raises:
            ValueError: If min_chunk_size >= max_chunk_size or sizes are invalid
        """
        if min_chunk_size >= max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        if min_chunk_size < 1:
            raise ValueError("min_chunk_size must be at least 1")
        if max_chunk_size < 5:
            raise ValueError("max_chunk_size must be at least 5")
            
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.prefer_sentence_ends = prefer_sentence_ends
        self.include_clause_breaks = include_clause_breaks
        
        # Common abbreviations that shouldn't trigger sentence breaks
        self.abbreviations = frozenset([
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'ave', 'blvd',
            'co', 'corp', 'inc', 'ltd', 'vs', 'etc', 'ie', 'eg', 'al', 'et',
            'ca', 'approx', 'max', 'min', 'misc', 'dept', 'govt', 'univ',
            'assoc', 'bros', 'ph', 'md', 'dds', 'phd', 'ma', 'ba', 'bs',
            'usa', 'uk', 'us', 'eu', 'un', 'nasa', 'fbi', 'cia', 'gps'
        ])
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for boundary detection"""
        
        # Strong sentence endings - improved to handle more quote types
        self.sentence_end_pattern = re.compile(
            r'[.!?]+(?:\s|$|["\'\)\]\}\u201C\u201D\u2018\u2019])',
            re.IGNORECASE
        )
        
        # Ellipsis patterns
        self.ellipsis_pattern = re.compile(
            r'\.{3,}(?:\s|$|["\'\)\]\}\u201C\u201D\u2018\u2019])',
            re.IGNORECASE
        )
        
        # Clause break patterns  
        self.clause_break_pattern = re.compile(
            r'[;:,](?:\s|$|["\'\)\]\}\u201C\u201D\u2018\u2019])|--(?:\s|$)',
            re.IGNORECASE
        )
        
        # Paragraph breaks
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # Decimal number pattern (to avoid false positives)
        self.decimal_pattern = re.compile(r'\d+\.\d+')
        
        # URL pattern (to avoid false positives)
        self.url_pattern = re.compile(
            r'https?://[^\s]+|www\.[^\s]+|\w+\.\w+/[^\s]*',
            re.IGNORECASE
        )
        
        # Create a more efficient abbreviation checking method
        self.abbrev_pattern = re.compile(
            r'\b\w+\.$',
            re.IGNORECASE
        )
    
    def detect_boundaries(self, text: str) -> List[SentenceBoundary]:
        """
        Detect all potential sentence boundaries in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected boundaries sorted by position
            
        Raises:
            TypeError: If text is not a string
            ValueError: If text is None
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if text is None:
            raise ValueError("Input text cannot be None")
        
        if not text.strip():
            return []
        
        boundaries = []
        
        # Detect paragraph breaks first (highest priority)
        boundaries.extend(self._detect_paragraph_breaks(text))
        
        # Detect ellipsis
        boundaries.extend(self._detect_ellipsis(text))
        
        # Detect sentence endings
        boundaries.extend(self._detect_sentence_endings(text))
        
        # Detect clause breaks if enabled
        if self.include_clause_breaks:
            boundaries.extend(self._detect_clause_breaks(text))
        
        # Sort by position and remove duplicates
        boundaries = sorted(boundaries, key=lambda x: x.position)
        boundaries = self._remove_duplicate_boundaries(boundaries)
        
        return boundaries
    
    def _detect_paragraph_breaks(self, text: str) -> List[SentenceBoundary]:
        """Detect paragraph breaks in text"""
        boundaries = []
        for match in self.paragraph_pattern.finditer(text):
            boundaries.append(SentenceBoundary(
                position=match.start(),
                boundary_type=BoundaryType.PARAGRAPH_BREAK,
                confidence=1.0,
                punctuation=match.group()
            ))
        return boundaries
    
    def _detect_ellipsis(self, text: str) -> List[SentenceBoundary]:
        """Detect ellipsis patterns in text"""
        boundaries = []
        for match in self.ellipsis_pattern.finditer(text):
            if not self._is_false_positive(text, match.start(), match.end()):
                boundaries.append(SentenceBoundary(
                    position=match.end(),
                    boundary_type=BoundaryType.ELLIPSIS,
                    confidence=0.9,
                    punctuation=match.group().strip()
                ))
        return boundaries
    
    def _detect_sentence_endings(self, text: str) -> List[SentenceBoundary]:
        """Detect sentence endings in text"""
        boundaries = []
        for match in self.sentence_end_pattern.finditer(text):
            if not self._is_false_positive(text, match.start(), match.end()):
                confidence = self._calculate_confidence(text, match.start(), match.end())
                boundaries.append(SentenceBoundary(
                    position=match.end(),
                    boundary_type=BoundaryType.SENTENCE_END,
                    confidence=confidence,
                    punctuation=match.group().strip()
                ))
        return boundaries
    
    def _detect_clause_breaks(self, text: str) -> List[SentenceBoundary]:
        """Detect clause breaks in text"""
        boundaries = []
        for match in self.clause_break_pattern.finditer(text):
            if not self._is_false_positive(text, match.start(), match.end()):
                confidence = self._calculate_confidence(text, match.start(), match.end())
                boundaries.append(SentenceBoundary(
                    position=match.end(),
                    boundary_type=BoundaryType.CLAUSE_BREAK,
                    confidence=confidence * CLAUSE_CONFIDENCE_MULTIPLIER,
                    punctuation=match.group().strip()
                ))
        return boundaries
    
    def _is_false_positive(self, text: str, start: int, end: int) -> bool:
        """Check if a boundary is likely a false positive"""
        
        # Check for abbreviations - more efficient approach
        if start > 0:
            # Look for word before punctuation
            before_text = text[:start].lower()
            words = before_text.split()
            if words and words[-1] in self.abbreviations:
                return True
        
        # Check for decimal numbers
        context = text[max(0, start-CONTEXT_WINDOW_SIZE//2):end+CONTEXT_WINDOW_SIZE//2]
        if self.decimal_pattern.search(context):
            return True
        
        # Check for URLs
        if self.url_pattern.search(context):
            return True
        
        # Check for initials (e.g., "J. R. R. Tolkien")
        if self._is_likely_initial(text, start, end):
            return True
        
        return False
    
    def _is_likely_initial(self, text: str, start: int, end: int) -> bool:
        """Check if this looks like an initial (J. R. R. Tolkien)"""
        if start == 0 or end >= len(text):
            return False
            
        before_char = text[start-1]
        after_chars = text[end:end+2] if end+2 <= len(text) else text[end:]
        
        # Pattern: single letter + period + space + single letter
        return (before_char.isalpha() and 
                len(after_chars) >= 2 and
                after_chars[0].isspace() and 
                after_chars[1].isupper())
    
    def _calculate_confidence(self, text: str, start: int, end: int) -> float:
        """Calculate confidence score for a boundary"""
        confidence = BASE_CONFIDENCE
        
        # Higher confidence for sentence endings
        punct = text[start:end].strip()
        if punct in ['.', '!', '?']:
            confidence += CONFIDENCE_BOOST_STRONG_PUNCT
        
        # Higher confidence if followed by capital letter
        if end < len(text) and text[end:end+1].isupper():
            confidence += CONFIDENCE_BOOST_CAPITAL
        
        # Lower confidence for very short segments
        if start < self.min_chunk_size:
            confidence -= CONFIDENCE_PENALTY_SHORT
        
        return max(0.0, min(1.0, confidence))
    
    def _remove_duplicate_boundaries(self, boundaries: List[SentenceBoundary]) -> List[SentenceBoundary]:
        """Remove boundaries that are too close together"""
        if not boundaries:
            return boundaries
        
        filtered = [boundaries[0]]
        for boundary in boundaries[1:]:
            # If boundaries are within threshold, keep the higher confidence one
            if boundary.position - filtered[-1].position <= BOUNDARY_PROXIMITY_THRESHOLD:
                if boundary.confidence > filtered[-1].confidence:
                    filtered[-1] = boundary
            else:
                filtered.append(boundary)
        
        return filtered
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into TTS-friendly chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks ready for TTS
            
        Raises:
            TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
            
        if not text.strip():
            return []
        
        boundaries = self.detect_boundaries(text)
        
        if not boundaries:
            # No boundaries found, return the whole text if it's not too long
            if len(text) <= self.max_chunk_size:
                return [text.strip()]
            else:
                # Force split at word boundaries
                return self._force_split_at_words(text)
        
        return self._create_chunks_from_boundaries(text, boundaries)
    
    def _create_chunks_from_boundaries(self, text: str, boundaries: List[SentenceBoundary]) -> List[str]:
        """Create chunks from detected boundaries"""
        chunks = []
        last_pos = 0
        
        for boundary in boundaries:
            # Check if we have enough text for a chunk
            chunk_text = text[last_pos:boundary.position].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                # Good chunk size
                chunks.append(chunk_text)
                last_pos = boundary.position
            elif len(chunk_text) > 0:
                # Chunk is too small, combine with previous if possible
                if chunks and len(chunks[-1]) + len(chunk_text) <= self.max_chunk_size:
                    chunks[-1] = chunks[-1] + " " + chunk_text
                    last_pos = boundary.position
                else:
                    # Start a new chunk
                    chunks.append(chunk_text)
                    last_pos = boundary.position
        
        # Add remaining text
        remaining = text[last_pos:].strip()
        if remaining:
            if chunks and len(chunks[-1]) + len(remaining) <= self.max_chunk_size:
                chunks[-1] = chunks[-1] + " " + remaining
            else:
                chunks.append(remaining)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _force_split_at_words(self, text: str) -> List[str]:
        """Force split long text at word boundaries"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_chunk else 0)  # +1 for space
            
            if current_length + word_length > self.max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def get_optimal_break_point(self, text: str, target_length: int) -> Optional[int]:
        """
        Find the optimal break point near a target length.
        
        Args:
            text: Input text
            target_length: Desired break point position
            
        Returns:
            Optimal break position or None if no good break found
            
        Raises:
            TypeError: If text is not a string
            ValueError: If target_length is invalid
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string")
        if target_length < 0:
            raise ValueError("target_length must be non-negative")
            
        if target_length >= len(text):
            return len(text)
        
        # Look for boundaries within a reasonable range of target
        search_range = min(50, target_length // 2)
        start_search = max(self.min_chunk_size, target_length - search_range)
        end_search = min(len(text), target_length + search_range)
        
        search_text = text[start_search:end_search]
        boundaries = self.detect_boundaries(search_text)
        
        if not boundaries:
            return None
        
        # Find the boundary closest to target with good confidence
        best_boundary = None
        best_score = -1
        
        for boundary in boundaries:
            actual_pos = start_search + boundary.position
            distance_score = 1.0 - abs(actual_pos - target_length) / search_range
            combined_score = boundary.confidence * 0.6 + distance_score * 0.4
            
            if combined_score > best_score:
                best_score = combined_score
                best_boundary = actual_pos
        
        return best_boundary


def main():
    """Test the sentence splitter with various examples"""
    
    # Test cases - reduced emoji usage
    test_texts = [
        "Hello world! This is a test. How are you doing today?",
        "The quick brown fox jumps over the lazy dog. It's a beautiful day, isn't it?",
        "Mr. Smith went to Washington D.C. He met with Dr. Johnson at 3:30 p.m.",
        "The price is $19.99 per item. Visit our website at www.example.com for more info.",
        "This is a long sentence that goes on and on; it has multiple clauses, separated by commas, and should be split appropriately.",
        "Sometimes... people use ellipsis... to create dramatic pauses.",
        "Short text.",
        "Lists include: apples, oranges, bananas; vegetables like carrots, celery, onions; and proteins such as chicken, fish, tofu."
    ]
    
    print("Sentence Splitter Test")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        print("-" * 50)
        
        # Test with default settings
        try:
            splitter = SentenceSplitter(min_chunk_size=10, max_chunk_size=80)
            chunks = splitter.split_into_chunks(text)
            
            print(f"Input length: {len(text)} characters")
            print(f"Number of chunks: {len(chunks)}")
            
            for j, chunk in enumerate(chunks, 1):
                print(f"  Chunk {j}: '{chunk}' ({len(chunk)} chars)")
            
            # Test boundary detection
            boundaries = splitter.detect_boundaries(text)
            print(f"Detected boundaries: {len(boundaries)}")
            for boundary in boundaries:
                print(f"  {boundary.boundary_type.value} at pos {boundary.position}: "
                      f"'{boundary.punctuation}' (conf: {boundary.confidence:.2f})")
                      
        except Exception as e:
            print(f"Error processing text: {e}")
    
    print("\n" + "=" * 60)
    print("Sentence Splitter tests completed!")


if __name__ == "__main__":
    main()