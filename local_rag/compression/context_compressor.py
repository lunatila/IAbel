"""
Context Compression - Redução inteligente de contexto irrelevante
Remove informações redundantes e mantém apenas o essencial para resposta precisa
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
import uuid
import logging

@dataclass
class CompressedContext:
    """
    Contexto comprimido com metadados de compressão
    """
    original_text: str
    compressed_text: str
    compression_ratio: float
    relevance_score: float
    key_sentences: List[str]
    removed_sentences: List[str]
    compression_method: str
    metadata: Dict[str, Any]

class ContextCompressor:
    """
    Sistema avançado de compressão de contexto para RAG
    """
    
    def __init__(self, 
                 embedder_model: Optional[SentenceTransformer] = None,
                 compression_ratio: float = 0.8,
                 min_sentence_length: int = 20,
                 max_context_length: int = 2000):
        """
        Initialize context compressor
        
        Args:
            embedder_model: SentenceTransformer for semantic analysis
            compression_ratio: Target compression ratio (0.0-1.0)
            min_sentence_length: Minimum sentence length to consider
            max_context_length: Maximum context length after compression
        """
        if embedder_model is None:
            print("Loading embedder for context compression...")
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedder = embedder_model
            
        self.compression_ratio = compression_ratio
        self.min_sentence_length = min_sentence_length
        self.max_context_length = max_context_length
        
        # TF-IDF for keyword importance
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Technical terms specific to reservoir engineering
        self.technical_terms = {
            'insim', 'insim-ft', 'permeabilidade', 'permeability', 'porosidade', 
            'porosity', 'saturação', 'saturation', 'simulação', 'simulation',
            'reservatório', 'reservoir', 'waterflooding', 'injeção', 'injection',
            'produção', 'production', 'eclipse', 'cmg', 'modelo', 'model',
            'numérico', 'numerical', 'equação', 'equation', 'pressão', 'pressure'
        }
        
        print(f"✅ Context Compressor initialized (target ratio: {compression_ratio})")
    
    def compress_context(self, 
                        contexts: List[Dict[str, Any]], 
                        query: str,
                        preserve_citations: bool = True) -> CompressedContext:
        """
        Compress multiple context chunks intelligently
        
        Args:
            contexts: List of context chunks with content and metadata
            query: Original user query
            preserve_citations: Whether to preserve citation information
            
        Returns:
            CompressedContext with optimized content
        """
        if not contexts:
            return CompressedContext(
                original_text="",
                compressed_text="",
                compression_ratio=0.0,
                relevance_score=0.0,
                key_sentences=[],
                removed_sentences=[],
                compression_method="empty_input",
                metadata={}
            )
        
        # Combine all contexts
        combined_text = self._combine_contexts(contexts, preserve_citations)
        
        # Split into sentences
        sentences = self._split_sentences(combined_text)
        
        if len(sentences) <= 3:  # Too few sentences to compress meaningfully
            return CompressedContext(
                original_text=combined_text,
                compressed_text=combined_text,
                compression_ratio=1.0,
                relevance_score=1.0,
                key_sentences=sentences,
                removed_sentences=[],
                compression_method="no_compression_needed",
                metadata={"reason": "too_few_sentences"}
            )
        
        # Calculate sentence importance scores
        sentence_scores = self._calculate_sentence_importance(sentences, query)
        
        # Select most important sentences
        selected_sentences, removed_sentences = self._select_sentences(
            sentences, sentence_scores, query
        )
        
        # Reorder sentences to maintain logical flow
        compressed_text = self._reorder_sentences(selected_sentences, combined_text)
        
        # Calculate metrics
        compression_ratio = len(compressed_text) / len(combined_text) if combined_text else 0
        relevance_score = np.mean([score for _, score in sentence_scores[:len(selected_sentences)]])
        
        return CompressedContext(
            original_text=combined_text,
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            relevance_score=relevance_score,
            key_sentences=selected_sentences,
            removed_sentences=removed_sentences,
            compression_method="semantic_importance",
            metadata={
                "original_length": len(combined_text),
                "compressed_length": len(compressed_text),
                "sentences_kept": len(selected_sentences),
                "sentences_removed": len(removed_sentences),
                "query": query
            }
        )
    
    def _combine_contexts(self, contexts: List[Dict[str, Any]], preserve_citations: bool) -> str:
        """Combine multiple contexts into single text"""
        combined_parts = []
        
        for i, context in enumerate(contexts):
            # Handle both dict and string inputs
            if isinstance(context, dict):
                content = context.get('content', '')
                if preserve_citations:
                    source = context.get('metadata', {}).get('source', f'Document_{i+1}')
                    page = context.get('metadata', {}).get('page', 'N/A')
                    combined_parts.append(f"[{source}, p.{page}] {content}")
                else:
                    combined_parts.append(content)
            else:
                # Handle string inputs (fallback)
                content = str(context)
                if preserve_citations:
                    combined_parts.append(f"[Document_{i+1}, p.N/A] {content}")
                else:
                    combined_parts.append(content)
        
        return " ".join(combined_parts)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved patterns"""
        # Handle common abbreviations and technical terms
        text = re.sub(r'\b(Fig|Table|Eq|Dr|Prof|et al)\.\s', r'\1§TEMP§ ', text)
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('§TEMP§', '.') for s in sentences]
        
        # Filter out very short sentences and clean
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.min_sentence_length:
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _calculate_sentence_importance(self, sentences: List[str], query: str) -> List[Tuple[str, float]]:
        """Calculate importance score for each sentence"""
        if not sentences:
            return []
        
        # 1. Semantic similarity to query
        query_embedding = self.embedder.encode([query])
        sentence_embeddings = self.embedder.encode(sentences)
        semantic_scores = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # 2. Technical term density
        technical_scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            tech_count = sum(1 for term in self.technical_terms if term in sentence_lower)
            technical_scores.append(tech_count / len(sentence.split()) if sentence.split() else 0)
        
        # 3. Position importance (earlier sentences often more important)
        position_scores = []
        for i, _ in enumerate(sentences):
            position_scores.append(1.0 - (i / len(sentences)) * 0.3)  # Decay by 30%
        
        # 4. Length normalization (prefer medium-length sentences)
        length_scores = []
        for sentence in sentences:
            length = len(sentence.split())
            if 10 <= length <= 30:  # Optimal range
                length_scores.append(1.0)
            elif length < 10:
                length_scores.append(0.7)
            else:
                length_scores.append(0.8)
        
        # 5. Information density (unique words / total words)
        density_scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            if words:
                unique_ratio = len(set(words)) / len(words)
                density_scores.append(unique_ratio)
            else:
                density_scores.append(0.0)
        
        # Combine scores with weights
        combined_scores = []
        for i, sentence in enumerate(sentences):
            score = (
                semantic_scores[i] * 0.4 +           # 40% semantic similarity
                technical_scores[i] * 0.25 +        # 25% technical relevance
                position_scores[i] * 0.15 +         # 15% position importance
                length_scores[i] * 0.1 +            # 10% length optimization
                density_scores[i] * 0.1              # 10% information density
            )
            combined_scores.append((sentence, score))
        
        # Sort by importance score (descending)
        return sorted(combined_scores, key=lambda x: x[1], reverse=True)
    
    def _select_sentences(self, sentences: List[str], sentence_scores: List[Tuple[str, float]], query: str) -> Tuple[List[str], List[str]]:
        """Select most important sentences based on compression ratio"""
        if not sentence_scores:
            return [], sentences
        
        # Calculate target number of sentences
        target_count = max(1, int(len(sentences) * self.compression_ratio))
        
        # Always include the most relevant sentence
        selected = [sentence_scores[0][0]]
        selected_scores = [sentence_scores[0][1]]
        
        # Add more sentences while staying under length limit
        current_length = len(selected[0])
        
        for sentence, score in sentence_scores[1:]:
            if len(selected) >= target_count:
                break
            
            # Check if adding this sentence would exceed length limit
            potential_length = current_length + len(sentence)
            if potential_length > self.max_context_length:
                continue
            
            # Avoid redundancy - check semantic similarity with already selected
            if not self._is_redundant(sentence, selected):
                selected.append(sentence)
                selected_scores.append(score)
                current_length = potential_length
        
        # Find removed sentences
        selected_set = set(selected)
        removed = [s for s in sentences if s not in selected_set]
        
        return selected, removed
    
    def _is_redundant(self, candidate: str, selected: List[str], threshold: float = 0.8) -> bool:
        """Check if candidate sentence is too similar to already selected ones"""
        if not selected:
            return False
        
        candidate_embedding = self.embedder.encode([candidate])
        selected_embeddings = self.embedder.encode(selected)
        
        similarities = cosine_similarity(candidate_embedding, selected_embeddings)[0]
        
        return np.max(similarities) > threshold
    
    def _reorder_sentences(self, selected_sentences: List[str], original_text: str) -> str:
        """Reorder selected sentences to maintain logical flow"""
        if not selected_sentences:
            return ""
        
        # Find original positions
        sentence_positions = []
        for sentence in selected_sentences:
            # Find approximate position in original text
            position = original_text.find(sentence[:50])  # Use first 50 chars for matching
            sentence_positions.append((position if position != -1 else 0, sentence))
        
        # Sort by original position to maintain flow
        sentence_positions.sort(key=lambda x: x[0])
        
        # Join with proper spacing
        reordered = [sentence for _, sentence in sentence_positions]
        return ". ".join(reordered) + "." if reordered else ""
    
    def get_compression_stats(self, compressed_context: CompressedContext) -> Dict[str, Any]:
        """Get detailed compression statistics"""
        return {
            "compression_ratio": compressed_context.compression_ratio,
            "relevance_score": compressed_context.relevance_score,
            "original_length": len(compressed_context.original_text),
            "compressed_length": len(compressed_context.compressed_text),
            "bytes_saved": len(compressed_context.original_text) - len(compressed_context.compressed_text),
            "sentences_kept": len(compressed_context.key_sentences),
            "sentences_removed": len(compressed_context.removed_sentences),
            "compression_method": compressed_context.compression_method,
            "efficiency": compressed_context.relevance_score / compressed_context.compression_ratio if compressed_context.compression_ratio > 0 else 0
        }


# Global context compressor instance
_global_context_compressor = None

def get_context_compressor(embedder_model: Optional[SentenceTransformer] = None) -> ContextCompressor:
    """
    Get or create global context compressor instance
    """
    global _global_context_compressor
    if _global_context_compressor is None:
        _global_context_compressor = ContextCompressor(embedder_model)
    return _global_context_compressor