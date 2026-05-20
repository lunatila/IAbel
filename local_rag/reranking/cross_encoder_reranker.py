"""
Cross-Encoder Re-ranking for Enhanced RAG
Improved relevance scoring using sentence-transformers cross-encoder models
"""

import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import os

class CrossEncoderReranker:
    """
    Advanced re-ranking using cross-encoder models for better relevance scoring
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 device: str = None,
                 cache_dir: str = None):
        """
        Initialize cross-encoder re-ranker
        
        Args:
            model_name: Cross-encoder model for re-ranking
            device: Device to run model on (auto-detect if None)
            cache_dir: Directory to cache models
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.join(str(Path.home()), ".cache", "iabel_models")
        os.makedirs(cache_dir, exist_ok=True)
        
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        print(f"🔄 Loading cross-encoder: {model_name}")
        print(f"   Device: {self.device}")
        print(f"   Cache: {cache_dir}")
        
        try:
            # Load cross-encoder model
            self.model = CrossEncoder(
                model_name,
                device=self.device,
                cache_folder=cache_dir
            )
            print(f"✅ Cross-encoder loaded successfully")
        except Exception as e:
            print(f"❌ Error loading cross-encoder: {e}")
            print("   Falling back to similarity-only ranking")
            self.model = None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict[str, Any]], 
               top_k: int = None) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder scoring
        
        Args:
            query: User query
            documents: List of document results with content and metadata
            top_k: Number of top documents to return (None = all)
        
        Returns:
            Re-ranked list of documents with updated scores
        """
        if not documents:
            return documents
        
        if self.model is None:
            # Fallback to original ranking if model failed to load
            print("⚠️ Cross-encoder not available, using original ranking")
            return documents[:top_k] if top_k else documents
        
        try:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = []
            for doc in documents:
                content = doc.get('content', '')
                # Truncate content to avoid token limits
                content = content[:512] if len(content) > 512 else content
                query_doc_pairs.append([query, content])
            
            print(f"🔄 Re-ranking {len(documents)} documents...")
            
            # Get cross-encoder scores
            with torch.no_grad():
                cross_scores = self.model.predict(query_doc_pairs)
            
            # Update documents with cross-encoder scores
            reranked_docs = []
            for i, doc in enumerate(documents):
                updated_doc = doc.copy()
                
                # Store original similarity and new cross-encoder score
                original_similarity = doc.get('similarity', 0.0)
                cross_score = float(cross_scores[i])
                
                # Normalize cross-encoder score to [0, 1] range
                normalized_cross_score = self._normalize_score(cross_score)
                
                # Combine scores (weighted average)
                combined_score = self._combine_scores(
                    original_similarity, 
                    normalized_cross_score
                )
                
                updated_doc.update({
                    'similarity': combined_score,
                    'cross_encoder_score': normalized_cross_score,
                    'original_similarity': original_similarity,
                    'reranked': True
                })
                
                reranked_docs.append(updated_doc)
            
            # Sort by combined score
            reranked_docs.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply top_k limit
            if top_k:
                reranked_docs = reranked_docs[:top_k]
            
            print(f"✅ Re-ranking complete")
            
            # Show score improvements for top 3
            for i, doc in enumerate(reranked_docs[:3], 1):
                original = doc.get('original_similarity', 0.0)
                new_score = doc.get('similarity', 0.0)
                improvement = new_score - original
                print(f"   {i}. {original:.3f} → {new_score:.3f} ({improvement:+.3f})")
            
            return reranked_docs
            
        except Exception as e:
            print(f"❌ Error in re-ranking: {e}")
            print("   Falling back to original ranking")
            return documents[:top_k] if top_k else documents
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize cross-encoder score to [0, 1] range using sigmoid
        """
        # Cross-encoder scores can be negative, use sigmoid to normalize
        normalized = 1 / (1 + np.exp(-score))
        return float(normalized)
    
    def _combine_scores(self, 
                       similarity_score: float, 
                       cross_score: float,
                       similarity_weight: float = 0.3,
                       cross_weight: float = 0.7) -> float:
        """
        Combine similarity and cross-encoder scores
        
        Args:
            similarity_score: Original embedding similarity
            cross_score: Cross-encoder relevance score
            similarity_weight: Weight for similarity score
            cross_weight: Weight for cross-encoder score
        """
        # Weighted combination
        combined = (similarity_weight * similarity_score + 
                   cross_weight * cross_score)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, combined))
    
    def batch_rerank(self, 
                    queries_and_docs: List[Tuple[str, List[Dict[str, Any]]]], 
                    top_k: int = None) -> List[List[Dict[str, Any]]]:
        """
        Re-rank multiple query-document sets in batch
        
        Args:
            queries_and_docs: List of (query, documents) tuples
            top_k: Number of top documents per query
        
        Returns:
            List of re-ranked document lists
        """
        results = []
        for query, docs in queries_and_docs:
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'cache_dir': self.cache_dir,
            'model_loaded': self.model is not None,
            'max_seq_length': getattr(self.model, 'max_length', 512) if self.model else None
        }


class MultiStageReranker:
    """
    Multi-stage re-ranking pipeline combining different ranking strategies
    """
    
    def __init__(self, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
                 device: str = None):
        """
        Initialize multi-stage re-ranker
        """
        self.cross_encoder = CrossEncoderReranker(
            model_name=cross_encoder_model,
            device=device
        )
        
        # Technical boost keywords for engineering documents
        self.technical_keywords = {
            'high_priority': [
                'INSIM', 'INSIM-FT', 'permeabilidade', 'permeability', 
                'porosidade', 'porosity', 'simulação', 'simulation',
                'reservatório', 'reservoir', 'petróleo', 'oil'
            ],
            'medium_priority': [
                'fluxo', 'flow', 'pressão', 'pressure', 'saturação', 'saturation',
                'viscosidade', 'viscosity', 'densidade', 'density'
            ]
        }
    
    def enhanced_rerank(self, 
                       query: str, 
                       documents: List[Dict[str, Any]], 
                       top_k: int = None,
                       use_technical_boost: bool = True) -> List[Dict[str, Any]]:
        """
        Enhanced re-ranking with technical domain boosting
        
        Args:
            query: User query
            documents: Document results
            top_k: Number of documents to return
            use_technical_boost: Whether to apply technical keyword boosting
        """
        # Stage 1: Technical keyword boosting
        if use_technical_boost:
            documents = self._apply_technical_boost(query, documents)
        
        # Stage 2: Cross-encoder re-ranking
        reranked_docs = self.cross_encoder.rerank(query, documents, top_k)
        
        # Stage 3: Diversity filtering (avoid too similar documents)
        if len(reranked_docs) > 3:
            reranked_docs = self._apply_diversity_filter(reranked_docs)
        
        return reranked_docs
    
    def _apply_technical_boost(self, 
                              query: str, 
                              documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply boosting based on technical keywords
        """
        query_lower = query.lower()
        boosted_docs = []
        
        for doc in documents:
            updated_doc = doc.copy()
            content_lower = doc.get('content', '').lower()
            
            # Calculate technical relevance boost
            boost_factor = 1.0
            
            # High priority keywords
            for keyword in self.technical_keywords['high_priority']:
                if keyword.lower() in query_lower and keyword.lower() in content_lower:
                    boost_factor *= 1.2  # +20% per high priority match
            
            # Medium priority keywords
            for keyword in self.technical_keywords['medium_priority']:
                if keyword.lower() in query_lower and keyword.lower() in content_lower:
                    boost_factor *= 1.1  # +10% per medium priority match
            
            # Apply boost to similarity
            if boost_factor > 1.0:
                original_similarity = doc.get('similarity', 0.0)
                boosted_similarity = min(1.0, original_similarity * boost_factor)
                updated_doc['similarity'] = boosted_similarity
                updated_doc['technical_boost'] = boost_factor
            
            boosted_docs.append(updated_doc)
        
        return boosted_docs
    
    def _apply_diversity_filter(self, 
                               documents: List[Dict[str, Any]], 
                               similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Filter out very similar documents to increase diversity
        """
        if len(documents) <= 3:
            return documents
        
        filtered_docs = [documents[0]]  # Always keep the top result
        
        for doc in documents[1:]:
            # Check if document is too similar to already selected ones
            is_similar = False
            doc_content = doc.get('content', '').lower()
            
            for selected_doc in filtered_docs:
                selected_content = selected_doc.get('content', '').lower()
                
                # Simple similarity check based on word overlap
                doc_words = set(doc_content.split())
                selected_words = set(selected_content.split())
                
                if len(doc_words) > 0 and len(selected_words) > 0:
                    overlap = len(doc_words & selected_words)
                    union = len(doc_words | selected_words)
                    jaccard_similarity = overlap / union if union > 0 else 0
                    
                    if jaccard_similarity > similarity_threshold:
                        is_similar = True
                        break
            
            if not is_similar:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the re-ranking system
        """
        return {
            'cross_encoder': self.cross_encoder.get_model_info(),
            'technical_keywords': self.technical_keywords,
            'features': {
                'cross_encoder_reranking': True,
                'technical_boosting': True,
                'diversity_filtering': True,
                'multi_stage_pipeline': True
            }
        }


# Create global reranker instance
_global_reranker = None

def get_reranker() -> MultiStageReranker:
    """
    Get or create global reranker instance
    """
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = MultiStageReranker()
    return _global_reranker