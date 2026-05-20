"""
RAG Fusion - Enhanced retrieval using multiple query variations and result fusion
Generates multiple query variations and combines results for better coverage
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

class QueryExpander:
    """
    Generates multiple variations of a query for better retrieval coverage
    """
    
    def __init__(self):
        # Expansion templates for technical queries
        self.expansion_templates = {
            'definition': [
                "o que é {term}",
                "definição de {term}",
                "conceito de {term}",
                "{term} significado",
                "explicação {term}"
            ],
            'technical': [
                "{term} engenharia reservatórios",
                "{term} petróleo",
                "{term} simulação",
                "como calcular {term}",
                "fórmula {term}"
            ],
            'process': [
                "como {term}",
                "processo {term}",
                "método {term}",
                "procedimento {term}",
                "técnica {term}"
            ],
            'application': [
                "{term} aplicação",
                "uso de {term}",
                "{term} na prática",
                "exemplo {term}",
                "caso {term}"
            ]
        }
        
        # Domain-specific synonyms and variations
        self.technical_synonyms = {
            'permeabilidade': ['k', 'perm', 'condutividade hidráulica', 'facilidade de fluxo'],
            'porosidade': ['phi', 'φ', 'espaço poroso', 'volume de poros'],
            'saturação': ['sw', 'so', 'sg', 'saturation'],
            'viscosidade': ['mu', 'μ', 'viscosity'],
            'simulação': ['modeling', 'modelo', 'simulation'],
            'reservatório': ['reservoir', 'jazida'],
            'INSIM': ['INSIM-FT', 'interwell simulation', 'simulação interpoços'],
            'pressão': ['pressure', 'p'],
            'densidade': ['density', 'rho', 'ρ'],
            'fluxo': ['flow', 'vazão', 'rate'],
            'produção': ['production', 'recovery', 'extração']
        }
    
    def expand_query(self, query: str, max_variations: int = 8) -> List[str]:
        """
        Generate multiple variations of the input query
        
        Args:
            query: Original user query
            max_variations: Maximum number of variations to generate
        
        Returns:
            List of query variations including the original
        """
        variations = [query.strip()]  # Start with original query
        query_lower = query.lower().strip()
        
        # Extract key terms from query
        key_terms = self._extract_key_terms(query)
        
        # Generate variations based on query type
        query_type = self._classify_query(query_lower)
        
        # Add template-based variations
        for term in key_terms:
            if query_type in self.expansion_templates:
                for template in self.expansion_templates[query_type][:2]:  # Limit per type
                    variation = template.format(term=term)
                    if variation not in variations:
                        variations.append(variation)
        
        # Add synonym-based variations
        synonym_variations = self._generate_synonym_variations(query_lower, key_terms)
        variations.extend(synonym_variations[:3])  # Limit synonym variations
        
        # Add contextual variations
        context_variations = self._generate_contextual_variations(query_lower)
        variations.extend(context_variations[:2])  # Limit contextual variations
        
        # Remove duplicates while preserving order
        unique_variations = []
        seen = set()
        for var in variations:
            var_clean = var.lower().strip()
            if var_clean not in seen:
                seen.add(var_clean)
                unique_variations.append(var)
        
        return unique_variations[:max_variations]
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key technical terms from query"""
        # Remove common question words
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'o', 'que', 'é', 'como', 'qual', 'onde', 'quando', 'por', 'para', 'de', 'da', 'do', 'em', 'na', 'no'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Prioritize known technical terms
        technical_terms = []
        for term in key_terms:
            if term in self.technical_synonyms or any(term in syns for syns in self.technical_synonyms.values()):
                technical_terms.append(term)
        
        return technical_terms if technical_terms else key_terms[:3]
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for appropriate expansion"""
        if any(word in query for word in ['o que é', 'definição', 'significa', 'conceito']):
            return 'definition'
        elif any(word in query for word in ['como', 'processo', 'método', 'procedimento']):
            return 'process'
        elif any(word in query for word in ['aplicação', 'uso', 'exemplo', 'prática']):
            return 'application'
        else:
            return 'technical'
    
    def _generate_synonym_variations(self, query: str, key_terms: List[str]) -> List[str]:
        """Generate variations using technical synonyms"""
        variations = []
        
        for term in key_terms:
            if term in self.technical_synonyms:
                for synonym in self.technical_synonyms[term][:2]:  # Limit per term
                    variation = query.replace(term, synonym)
                    if variation != query:
                        variations.append(variation)
        
        return variations
    
    def _generate_contextual_variations(self, query: str) -> List[str]:
        """Generate contextually enriched variations"""
        variations = []
        
        # Add domain context
        if not any(domain in query for domain in ['petróleo', 'reservatório', 'engenharia']):
            variations.append(f"{query} petróleo")
            variations.append(f"{query} engenharia reservatórios")
        
        # Add simulation context for technical terms
        if any(term in query for term in ['INSIM', 'modelo', 'simulação']):
            variations.append(f"{query} simulador")
        
        return variations


class ResultFusion:
    """
    Combines and ranks results from multiple query variations
    """
    
    def __init__(self, embedder_model: Optional[SentenceTransformer] = None):
        """
        Initialize result fusion system
        
        Args:
            embedder_model: SentenceTransformer model for semantic similarity
        """
        self.embedder = embedder_model
    
    def fuse_results(self, 
                    query_results: List[Tuple[str, List[Dict[str, Any]]]], 
                    max_results: int = 10,
                    fusion_method: str = "reciprocal_rank") -> List[Dict[str, Any]]:
        """
        Fuse results from multiple queries using various methods
        
        Args:
            query_results: List of (query, results) tuples
            max_results: Maximum number of results to return
            fusion_method: Method for fusion ('reciprocal_rank', 'score_sum', 'weighted')
        
        Returns:
            Fused and ranked results
        """
        if not query_results:
            return []
        
        if fusion_method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(query_results, max_results)
        elif fusion_method == "score_sum":
            return self._score_sum_fusion(query_results, max_results)
        elif fusion_method == "weighted":
            return self._weighted_fusion(query_results, max_results)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def _reciprocal_rank_fusion(self, 
                               query_results: List[Tuple[str, List[Dict[str, Any]]]], 
                               max_results: int) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) - effective for combining ranked lists
        """
        k = 60  # RRF parameter (standard value)
        document_scores = defaultdict(float)
        document_data = {}
        
        for query, results in query_results:
            for rank, result in enumerate(results, 1):
                # Create unique document ID
                doc_id = self._create_doc_id(result)
                
                # RRF score: 1 / (k + rank)
                rrf_score = 1.0 / (k + rank)
                document_scores[doc_id] += rrf_score
                
                # Store document data (from first occurrence)
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['fusion_queries'] = [query]
                    document_data[doc_id]['fusion_ranks'] = [rank]
                else:
                    document_data[doc_id]['fusion_queries'].append(query)
                    document_data[doc_id]['fusion_ranks'].append(rank)
        
        # Create final ranked list
        fused_results = []
        for doc_id, score in sorted(document_scores.items(), key=lambda x: x[1], reverse=True):
            result = document_data[doc_id]
            result['fusion_score'] = score
            result['fusion_method'] = 'reciprocal_rank'
            result['query_matches'] = len(result['fusion_queries'])
            fused_results.append(result)
        
        return fused_results[:max_results]
    
    def _score_sum_fusion(self, 
                         query_results: List[Tuple[str, List[Dict[str, Any]]]], 
                         max_results: int) -> List[Dict[str, Any]]:
        """
        Sum similarity scores across queries
        """
        document_scores = defaultdict(float)
        document_data = {}
        
        for query, results in query_results:
            for result in results:
                doc_id = self._create_doc_id(result)
                similarity = result.get('similarity', 0.0)
                
                document_scores[doc_id] += similarity
                
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['fusion_queries'] = [query]
                    document_data[doc_id]['fusion_scores'] = [similarity]
                else:
                    document_data[doc_id]['fusion_queries'].append(query)
                    document_data[doc_id]['fusion_scores'].append(similarity)
        
        # Normalize by number of queries
        for doc_id in document_scores:
            query_count = len(document_data[doc_id]['fusion_queries'])
            document_scores[doc_id] = document_scores[doc_id] / query_count
        
        # Create final ranked list
        fused_results = []
        for doc_id, score in sorted(document_scores.items(), key=lambda x: x[1], reverse=True):
            result = document_data[doc_id]
            result['fusion_score'] = score
            result['fusion_method'] = 'score_sum'
            result['query_matches'] = len(result['fusion_queries'])
            fused_results.append(result)
        
        return fused_results[:max_results]
    
    def _weighted_fusion(self, 
                        query_results: List[Tuple[str, List[Dict[str, Any]]]], 
                        max_results: int) -> List[Dict[str, Any]]:
        """
        Weighted fusion giving more importance to original query
        """
        document_scores = defaultdict(float)
        document_data = {}
        
        # First query (original) gets higher weight
        weights = [1.0] + [0.7] * (len(query_results) - 1)
        
        for i, (query, results) in enumerate(query_results):
            weight = weights[i] if i < len(weights) else 0.5
            
            for result in results:
                doc_id = self._create_doc_id(result)
                similarity = result.get('similarity', 0.0)
                
                document_scores[doc_id] += similarity * weight
                
                if doc_id not in document_data:
                    document_data[doc_id] = result.copy()
                    document_data[doc_id]['fusion_queries'] = [query]
                    document_data[doc_id]['fusion_weights'] = [weight]
                else:
                    document_data[doc_id]['fusion_queries'].append(query)
                    document_data[doc_id]['fusion_weights'].append(weight)
        
        # Create final ranked list
        fused_results = []
        for doc_id, score in sorted(document_scores.items(), key=lambda x: x[1], reverse=True):
            result = document_data[doc_id]
            result['fusion_score'] = score
            result['fusion_method'] = 'weighted'
            result['query_matches'] = len(result['fusion_queries'])
            fused_results.append(result)
        
        return fused_results[:max_results]
    
    def _create_doc_id(self, result: Dict[str, Any]) -> str:
        """Create unique document ID from result"""
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        source = metadata.get('source', '')
        page = metadata.get('page', '')
        
        # Create hash from content snippet + source + page
        id_string = f"{content[:100]}{source}{page}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def analyze_fusion_quality(self, fused_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of fusion results
        """
        if not fused_results:
            return {'total_results': 0}
        
        query_matches = [r.get('query_matches', 0) for r in fused_results]
        fusion_scores = [r.get('fusion_score', 0.0) for r in fused_results]
        
        return {
            'total_results': len(fused_results),
            'avg_query_matches': np.mean(query_matches),
            'max_query_matches': max(query_matches),
            'avg_fusion_score': np.mean(fusion_scores),
            'score_distribution': {
                'high': len([s for s in fusion_scores if s > 0.7]),
                'medium': len([s for s in fusion_scores if 0.3 <= s <= 0.7]),
                'low': len([s for s in fusion_scores if s < 0.3])
            }
        }


class RAGFusion:
    """
    Complete RAG Fusion system combining query expansion and result fusion
    """
    
    def __init__(self, embedder_model: Optional[SentenceTransformer] = None):
        """
        Initialize RAG Fusion system
        
        Args:
            embedder_model: SentenceTransformer model for embeddings
        """
        self.query_expander = QueryExpander()
        self.result_fusion = ResultFusion(embedder_model)
    
    async def enhanced_search(self, 
                             original_query: str,
                             search_function,
                             max_variations: int = 6,
                             fusion_method: str = "reciprocal_rank",
                             top_k: int = 10) -> Dict[str, Any]:
        """
        Perform enhanced search using RAG Fusion
        
        Args:
            original_query: User's original query
            search_function: Function to perform document search
            max_variations: Maximum query variations to generate
            fusion_method: Method for result fusion
            top_k: Number of final results to return
        
        Returns:
            Enhanced search results with fusion metadata
        """
        print(f"🔄 RAG Fusion: Expanding query '{original_query}'")
        
        # Generate query variations
        query_variations = self.query_expander.expand_query(
            original_query, 
            max_variations=max_variations
        )
        
        print(f"   Generated {len(query_variations)} query variations")
        for i, var in enumerate(query_variations[:3], 1):  # Show first 3
            print(f"   {i}. {var}")
        
        # Search with all variations (can be done in parallel)
        query_results = []
        
        if asyncio.iscoroutinefunction(search_function):
            # Async search
            tasks = []
            for query_var in query_variations:
                task = search_function(query_var, top_k=top_k)
                tasks.append((query_var, task))
            
            for query_var, task in tasks:
                try:
                    results = await task
                    if isinstance(results, dict) and 'results' in results:
                        results = results['results']
                    query_results.append((query_var, results))
                except Exception as e:
                    print(f"   ⚠️ Error searching with '{query_var}': {e}")
                    query_results.append((query_var, []))
        else:
            # Sync search
            for query_var in query_variations:
                try:
                    results = search_function(query_var, top_k=top_k)
                    if isinstance(results, dict) and 'results' in results:
                        results = results['results']
                    query_results.append((query_var, results))
                except Exception as e:
                    print(f"   ⚠️ Error searching with '{query_var}': {e}")
                    query_results.append((query_var, []))
        
        # Fuse results
        print(f"🔄 Fusing results using {fusion_method} method...")
        fused_results = self.result_fusion.fuse_results(
            query_results, 
            max_results=top_k,
            fusion_method=fusion_method
        )
        
        # Analyze fusion quality
        fusion_analysis = self.result_fusion.analyze_fusion_quality(fused_results)
        
        print(f"✅ RAG Fusion complete:")
        print(f"   Final results: {len(fused_results)}")
        print(f"   Avg query matches: {fusion_analysis.get('avg_query_matches', 0):.1f}")
        print(f"   High quality results: {fusion_analysis.get('score_distribution', {}).get('high', 0)}")
        
        return {
            'results': fused_results,
            'original_query': original_query,
            'query_variations': query_variations,
            'fusion_method': fusion_method,
            'fusion_analysis': fusion_analysis,
            'total_searches': len(query_variations)
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about RAG Fusion system
        """
        return {
            'query_expander': {
                'expansion_templates': len(self.query_expander.expansion_templates),
                'technical_synonyms': len(self.query_expander.technical_synonyms)
            },
            'result_fusion': {
                'available_methods': ['reciprocal_rank', 'score_sum', 'weighted'],
                'embedder_available': self.result_fusion.embedder is not None
            },
            'features': {
                'query_expansion': True,
                'synonym_replacement': True,
                'contextual_enrichment': True,
                'multi_method_fusion': True,
                'async_search': True,
                'quality_analysis': True
            }
        }


# Global RAG Fusion instance
_global_rag_fusion = None

def get_rag_fusion(embedder_model: Optional[SentenceTransformer] = None) -> RAGFusion:
    """
    Get or create global RAG Fusion instance
    """
    global _global_rag_fusion
    if _global_rag_fusion is None:
        _global_rag_fusion = RAGFusion(embedder_model)
    return _global_rag_fusion