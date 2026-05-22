"""
Citation Tracking - Rastreamento preciso de fontes nas respostas
Mapeia cada parte da resposta às suas fontes originais com precisão
"""

import re
import uuid
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

@dataclass
class CitationSource:
    """
    Informações detalhadas de uma fonte
    """
    source_id: str
    document_name: str
    page_number: int
    chunk_id: str
    content_excerpt: str
    confidence_score: float
    relevance_score: float
    character_start: int
    character_end: int
    metadata: Dict[str, Any]

@dataclass
class CitedSegment:
    """
    Segmento da resposta com suas citações
    """
    segment_id: str
    text: str
    start_position: int
    end_position: int
    sources: List[CitationSource]
    confidence: float
    citation_type: str  # 'direct', 'paraphrase', 'synthesis'

@dataclass
class CitationMap:
    """
    Mapa completo de citações para uma resposta
    """
    response_id: str
    response_text: str
    cited_segments: List[CitedSegment]
    uncited_segments: List[str]
    all_sources: List[CitationSource]
    citation_coverage: float  # Percentage of response that has citations
    source_diversity: int     # Number of unique sources used
    metadata: Dict[str, Any]

class CitationTracker:
    """
    Sistema avançado de rastreamento de citações para RAG
    """
    
    def __init__(self, 
                 embedder_model: Optional[SentenceTransformer] = None,
                 min_citation_confidence: float = 0.7,
                 citation_window_size: int = 100):
        """
        Initialize citation tracker
        
        Args:
            embedder_model: SentenceTransformer for semantic matching
            min_citation_confidence: Minimum confidence for valid citation
            citation_window_size: Size of text window for citation matching
        """
        if embedder_model is None:
            print("Loading embedder for citation tracking...")
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedder = embedder_model
            
        self.min_citation_confidence = min_citation_confidence
        self.citation_window_size = citation_window_size
        
        # Patterns for citation detection
        self.citation_patterns = [
            r'\[([^\]]+), p\.(\d+)\]',  # [Document, p.X]
            r'\(([^)]+), (\d+)\)',      # (Document, X)
            r'segundo ([^,]+),',        # segundo Document,
            r'conforme ([^,]+),',       # conforme Document,
            r'de acordo com ([^,]+),',  # de acordo com Document,
        ]
        
        print(f"✅ Citation Tracker initialized (min confidence: {min_citation_confidence})")
    
    def track_citations(self, 
                       response_text: str,
                       source_contexts: List[Dict[str, Any]],
                       response_id: Optional[str] = None) -> CitationMap:
        """
        Track citations throughout the response text
        
        Args:
            response_text: Generated response text
            source_contexts: List of source contexts used
            response_id: Unique identifier for this response
            
        Returns:
            CitationMap with detailed citation tracking
        """
        if response_id is None:
            response_id = str(uuid.uuid4())
        
        # Prepare source information
        source_info = self._prepare_sources(source_contexts)
        
        # Split response into sentences/segments
        segments = self._segment_response(response_text)
        
        # Find citations for each segment
        cited_segments = []
        uncited_segments = []
        
        for segment in segments:
            citations = self._find_citations_for_segment(segment, source_info)
            
            if citations:
                cited_segment = CitedSegment(
                    segment_id=str(uuid.uuid4()),
                    text=segment['text'],
                    start_position=segment['start'],
                    end_position=segment['end'],
                    sources=citations,
                    confidence=np.mean([c.confidence_score for c in citations]),
                    citation_type=self._determine_citation_type(segment['text'], citations)
                )
                cited_segments.append(cited_segment)
            else:
                uncited_segments.append(segment['text'])
        
        # Calculate coverage and diversity
        cited_length = sum(len(seg.text) for seg in cited_segments)
        citation_coverage = cited_length / len(response_text) if response_text else 0
        
        all_source_ids = set()
        for seg in cited_segments:
            for source in seg.sources:
                all_source_ids.add(source.source_id)
        source_diversity = len(all_source_ids)
        
        return CitationMap(
            response_id=response_id,
            response_text=response_text,
            cited_segments=cited_segments,
            uncited_segments=uncited_segments,
            all_sources=source_info,
            citation_coverage=citation_coverage,
            source_diversity=source_diversity,
            metadata={
                "total_segments": len(segments),
                "cited_segments": len(cited_segments),
                "uncited_segments": len(uncited_segments),
                "avg_citation_confidence": np.mean([seg.confidence for seg in cited_segments]) if cited_segments else 0
            }
        )
    
    def _prepare_sources(self, source_contexts: List[Dict[str, Any]]) -> List[CitationSource]:
        """Prepare source information for citation tracking"""
        sources = []
        
        for i, context in enumerate(source_contexts):
            metadata = context.get('metadata', {})
            content = context.get('content', '')
            
            # Extract key information
            source_id = metadata.get('chunk_id', f"source_{i}")
            document_name = metadata.get('source', f"Document_{i}")
            page_number = metadata.get('page', 1)
            
            # Create excerpt (first 200 chars)
            excerpt = content[:200] + "..." if len(content) > 200 else content
            
            source = CitationSource(
                source_id=source_id,
                document_name=document_name,
                page_number=page_number,
                chunk_id=source_id,
                content_excerpt=excerpt,
                confidence_score=context.get('score', 1.0),
                relevance_score=context.get('score', 1.0),
                character_start=0,
                character_end=len(content),
                metadata=metadata
            )
            sources.append(source)
        
        return sources
    
    def _segment_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Split response into sentences for citation matching."""
        segments = []
        # Split on sentence-ending punctuation followed by whitespace,
        # but not on abbreviations like Dr., et al., Fig., etc.
        # Strategy: split on ". " or "? " or "! " only when preceded by a
        # lowercase letter or closing bracket (not an uppercase abbreviation).
        sentence_end_re = re.compile(r'(?<=[a-z0-9\]\)])[.!?]+\s+')
        parts = sentence_end_re.split(response_text)

        current_pos = 0
        for part in parts:
            part = part.strip()
            if part:
                segments.append({
                    'text': part,
                    'start': current_pos,
                    'end': current_pos + len(part)
                })
            # Advance past the raw part length (+1 for the separator)
            current_pos += len(part) + 1

        return segments
    
    def _find_citations_for_segment(self, segment: Dict[str, Any], sources: List[CitationSource]) -> List[CitationSource]:
        """Find the best citations for a response segment"""
        segment_text = segment['text']
        
        # 1. Check for explicit citations first
        explicit_citations = self._find_explicit_citations(segment_text, sources)
        if explicit_citations:
            return explicit_citations
        
        # 2. Use semantic similarity to find best matching sources
        semantic_citations = self._find_semantic_citations(segment_text, sources)
        
        # 3. Filter by confidence threshold
        valid_citations = [
            citation for citation in semantic_citations 
            if citation.confidence_score >= self.min_citation_confidence
        ]
        
        # 4. Limit to top 3 most relevant sources per segment
        return sorted(valid_citations, key=lambda x: x.confidence_score, reverse=True)[:3]
    
    def _find_explicit_citations(self, text: str, sources: List[CitationSource]) -> List[CitationSource]:
        """Find explicitly marked citations in text"""
        found_citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                doc_name = match.group(1)
                
                # Find matching source
                for source in sources:
                    if doc_name.lower() in source.document_name.lower() or \
                       source.document_name.lower() in doc_name.lower():
                        
                        # Create copy with high confidence for explicit citation
                        explicit_source = CitationSource(
                            source_id=source.source_id,
                            document_name=source.document_name,
                            page_number=source.page_number,
                            chunk_id=source.chunk_id,
                            content_excerpt=source.content_excerpt,
                            confidence_score=0.95,  # High confidence for explicit
                            relevance_score=source.relevance_score,
                            character_start=match.start(),
                            character_end=match.end(),
                            metadata={**source.metadata, "citation_type": "explicit"}
                        )
                        found_citations.append(explicit_source)
                        break
        
        return found_citations
    
    def _find_semantic_citations(self, text: str, sources: List[CitationSource]) -> List[CitationSource]:
        """Find citations using semantic similarity"""
        if not text.strip() or not sources:
            return []
        
        # Get embeddings
        text_embedding = self.embedder.encode([text])
        source_embeddings = self.embedder.encode([source.content_excerpt for source in sources])
        
        # Calculate similarities
        similarities = cosine_similarity(text_embedding, source_embeddings)[0]
        
        # Create citations with similarity scores
        semantic_citations = []
        for i, (source, similarity) in enumerate(zip(sources, similarities)):
            if similarity > 0.3:  # Minimum semantic threshold
                semantic_source = CitationSource(
                    source_id=source.source_id,
                    document_name=source.document_name,
                    page_number=source.page_number,
                    chunk_id=source.chunk_id,
                    content_excerpt=source.content_excerpt,
                    confidence_score=float(similarity),
                    relevance_score=source.relevance_score,
                    character_start=0,
                    character_end=len(text),
                    metadata={**source.metadata, "citation_type": "semantic"}
                )
                semantic_citations.append(semantic_source)
        
        return semantic_citations
    
    def _determine_citation_type(self, text: str, citations: List[CitationSource]) -> str:
        """Determine the type of citation (direct, paraphrase, synthesis)"""
        if not citations:
            return "none"
        
        # Check for explicit citations
        has_explicit = any(c.metadata.get("citation_type") == "explicit" for c in citations)
        if has_explicit:
            return "direct"
        
        # Check confidence levels
        avg_confidence = np.mean([c.confidence_score for c in citations])
        
        if len(citations) > 1 and avg_confidence > 0.8:
            return "synthesis"  # Multiple high-confidence sources
        elif avg_confidence > 0.7:
            return "paraphrase"  # Single high-confidence source
        else:
            return "inference"  # Lower confidence, more interpretive
    
    def format_citations_for_display(self, citation_map: CitationMap, format_type: str = "academic") -> str:
        """Format citations for display in response"""
        if format_type == "academic":
            return self._format_academic_citations(citation_map)
        elif format_type == "inline":
            return self._format_inline_citations(citation_map)
        elif format_type == "footnotes":
            return self._format_footnote_citations(citation_map)
        else:
            return self._format_simple_citations(citation_map)
    
    def _format_academic_citations(self, citation_map: CitationMap) -> str:
        """Format citations in academic style"""
        response_with_citations = citation_map.response_text
        
        # Add numbered citations
        citation_counter = 1
        citation_list = []
        
        for segment in citation_map.cited_segments:
            for source in segment.sources:
                citation_mark = f"[{citation_counter}]"
                citation_list.append(
                    f"[{citation_counter}] {source.document_name}, p. {source.page_number}"
                )
                citation_counter += 1
        
        # Add bibliography at the end
        if citation_list:
            response_with_citations += "\n\n**Fontes:**\n" + "\n".join(set(citation_list))
        
        return response_with_citations
    
    def _format_inline_citations(self, citation_map: CitationMap) -> str:
        """Format citations inline with the text"""
        response_parts = []
        last_pos = 0
        
        # Sort segments by position
        sorted_segments = sorted(citation_map.cited_segments, key=lambda x: x.start_position)
        
        for segment in sorted_segments:
            # Add text before segment
            response_parts.append(citation_map.response_text[last_pos:segment.start_position])
            
            # Add segment with inline citations
            if segment.sources:
                sources_text = ", ".join([
                    f"({source.document_name}, p.{source.page_number})"
                    for source in segment.sources[:2]  # Limit to 2 sources per segment
                ])
                response_parts.append(f"{segment.text} {sources_text}")
            else:
                response_parts.append(segment.text)
            
            last_pos = segment.end_position
        
        # Add remaining text
        response_parts.append(citation_map.response_text[last_pos:])
        
        return "".join(response_parts)
    
    def _format_footnote_citations(self, citation_map: CitationMap) -> str:
        """Format citations as numbered footnotes."""
        footnote_counter = 1
        footnotes: List[str] = []
        result = citation_map.response_text

        for segment in citation_map.cited_segments:
            for source in segment.sources[:2]:
                note = f"[{footnote_counter}] {source.document_name}, p. {source.page_number}"
                footnotes.append(note)
                footnote_counter += 1

        if footnotes:
            result += "\n\n---\n" + "\n".join(footnotes)
        return result

    def _format_simple_citations(self, citation_map: CitationMap) -> str:
        """Simple citation format with source list"""
        unique_sources = {}
        for segment in citation_map.cited_segments:
            for source in segment.sources:
                unique_sources[source.source_id] = source
        
        if unique_sources:
            sources_list = [
                f"• {source.document_name} (p. {source.page_number})"
                for source in unique_sources.values()
            ]
            return citation_map.response_text + "\n\n**Baseado em:**\n" + "\n".join(sources_list)
        
        return citation_map.response_text
    
    def get_citation_stats(self, citation_map: CitationMap) -> Dict[str, Any]:
        """Get detailed citation statistics"""
        return {
            "citation_coverage": citation_map.citation_coverage,
            "source_diversity": citation_map.source_diversity,
            "total_sources": len(citation_map.all_sources),
            "cited_segments": len(citation_map.cited_segments),
            "uncited_segments": len(citation_map.uncited_segments),
            "avg_confidence": citation_map.metadata.get("avg_citation_confidence", 0),
            "citation_types": {
                "direct": sum(1 for seg in citation_map.cited_segments if seg.citation_type == "direct"),
                "paraphrase": sum(1 for seg in citation_map.cited_segments if seg.citation_type == "paraphrase"),
                "synthesis": sum(1 for seg in citation_map.cited_segments if seg.citation_type == "synthesis"),
                "inference": sum(1 for seg in citation_map.cited_segments if seg.citation_type == "inference")
            }
        }


# Global citation tracker instance
_global_citation_tracker = None

def get_citation_tracker(embedder_model: Optional[SentenceTransformer] = None) -> CitationTracker:
    """
    Get or create global citation tracker instance
    """
    global _global_citation_tracker
    if _global_citation_tracker is None:
        _global_citation_tracker = CitationTracker(embedder_model)
    return _global_citation_tracker