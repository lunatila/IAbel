"""
Semantic Chunking - Advanced document chunking based on semantic coherence
Creates meaningful chunks that preserve context and technical relationships
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy
from dataclasses import dataclass
import uuid

@dataclass
class SemanticChunk:
    """
    Enhanced chunk with semantic information
    """
    chunk_id: str
    content: str
    start_idx: int
    end_idx: int
    semantic_score: float
    section_type: str
    keywords: List[str]
    entities: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': {
                **self.metadata,
                'start_idx': self.start_idx,
                'end_idx': self.end_idx,
                'semantic_score': self.semantic_score,
                'section_type': self.section_type,
                'keywords': self.keywords,
                'entities': self.entities
            }
        }


class DocumentStructureAnalyzer:
    """
    Analyzes document structure to identify natural breaking points
    """
    
    def __init__(self):
        # Patterns for different document sections
        self.section_patterns = {
            'title': [
                r'^[A-Z][^.!?]*$',  # All caps or title case line
                r'^\d+\.\s+[A-Z].*',  # Numbered section
                r'^[IVX]+\.\s+[A-Z].*',  # Roman numeral section
            ],
            'abstract': [
                r'(?i)\babstract\b',
                r'(?i)\bresumo\b',
                r'(?i)\bsummary\b'
            ],
            'introduction': [
                r'(?i)\bintroduction\b',
                r'(?i)\bintrodução\b',
                r'(?i)\bintroducción\b'
            ],
            'methodology': [
                r'(?i)\bmethodology\b',
                r'(?i)\bmetodologia\b',
                r'(?i)\bmethods\b',
                r'(?i)\bmétodos\b'
            ],
            'results': [
                r'(?i)\bresults\b',
                r'(?i)\bresultados\b',
                r'(?i)\bfindings\b'
            ],
            'conclusion': [
                r'(?i)\bconclusion\b',
                r'(?i)\bconclusão\b',
                r'(?i)\bconclusiones\b'
            ],
            'equation': [
                r'\$.*\$',  # LaTeX equations
                r'\\begin\{equation\}',
                r'Equation\s+\d+',
                r'(?i)equação\s+\d+'
            ],
            'figure': [
                r'(?i)figure\s+\d+',
                r'(?i)figura\s+\d+',
                r'(?i)fig\.\s+\d+'
            ],
            'table': [
                r'(?i)table\s+\d+',
                r'(?i)tabela\s+\d+',
                r'(?i)tab\.\s+\d+'
            ]
        }
        
        # Technical terms that indicate important content
        self.technical_indicators = [
            'INSIM', 'INSIM-FT', 'permeabilidade', 'permeability',
            'porosidade', 'porosity', 'saturação', 'saturation',
            'viscosidade', 'viscosity', 'densidade', 'density',
            'simulação', 'simulation', 'modelo', 'model',
            'reservatório', 'reservoir', 'petróleo', 'oil',
            'equação', 'equation', 'fórmula', 'formula'
        ]
    
    def analyze_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyze document structure and identify section boundaries
        
        Args:
            text: Full document text
        
        Returns:
            List of sections with boundaries and types
        """
        lines = text.split('\n')
        sections = []
        current_section = {
            'start': 0,
            'type': 'body',
            'title': '',
            'importance': 1.0
        }
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check for section headers
            section_type = self._identify_section_type(line_stripped)
            
            if section_type and section_type != 'body':
                # End current section
                current_section['end'] = i
                current_section['line_count'] = i - current_section['start']
                sections.append(current_section)
                
                # Start new section
                current_section = {
                    'start': i,
                    'type': section_type,
                    'title': line_stripped,
                    'importance': self._calculate_section_importance(section_type)
                }
        
        # End last section
        current_section['end'] = len(lines)
        current_section['line_count'] = len(lines) - current_section['start']
        sections.append(current_section)
        
        return sections
    
    def _identify_section_type(self, line: str) -> str:
        """Identify the type of section based on line content"""
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, line):
                    return section_type
        return 'body'
    
    def _calculate_section_importance(self, section_type: str) -> float:
        """Calculate importance score for section type"""
        importance_scores = {
            'abstract': 1.5,
            'introduction': 1.2,
            'methodology': 1.3,
            'results': 1.4,
            'conclusion': 1.3,
            'equation': 1.6,
            'figure': 1.1,
            'table': 1.1,
            'title': 1.4,
            'body': 1.0
        }
        return importance_scores.get(section_type, 1.0)


class SemanticCoherenceAnalyzer:
    """
    Analyzes semantic coherence to determine optimal chunk boundaries
    """
    
    def __init__(self, embedder_model: Optional[SentenceTransformer] = None):
        """
        Initialize semantic analyzer
        
        Args:
            embedder_model: SentenceTransformer for semantic analysis
        """
        if embedder_model is None:
            print("Loading embedder for semantic chunking...")
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedder = embedder_model
        
        # Try to load spaCy model for NER
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️ No spaCy model found. Entity extraction disabled.")
                self.nlp = None
    
    def calculate_semantic_boundaries(self, 
                                    sentences: List[str], 
                                    threshold: float = 0.5) -> List[int]:
        """
        Calculate semantic boundaries using sliding window approach that preserves order
        
        Args:
            sentences: List of sentences
            threshold: Similarity threshold for boundaries
        
        Returns:
            List of boundary indices
        """
        if len(sentences) < 3:
            return []
        
        # Get sentence embeddings in batches for memory efficiency
        embeddings = []
        batch_size = 50
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.embedder.encode(batch)
            embeddings.extend(batch_embeddings)
        
        boundaries = []
        window_size = 5  # Look at 5 sentences context
        
        # Use sliding window to detect semantic shifts while preserving order
        for i in range(window_size, len(sentences) - window_size, 2):
            # Calculate coherence in window before and after position i
            before_coherence = 0
            after_coherence = 0
            before_count = 0
            after_count = 0
            
            # Analyze context window around position i
            for j in range(max(0, i - window_size), min(len(sentences) - 1, i + window_size)):
                if j >= len(embeddings) - 1:
                    continue
                    
                coherence = cosine_similarity([embeddings[j]], [embeddings[j+1]])[0][0]
                
                if j < i:
                    before_coherence += coherence
                    before_count += 1
                elif j >= i:
                    after_coherence += coherence
                    after_count += 1
            
            # Calculate average coherence
            if before_count > 0:
                before_coherence /= before_count
            if after_count > 0:
                after_coherence /= after_count
            
            # Check for semantic shift
            coherence_drop = before_coherence - after_coherence
            
            # Also check direct coherence at boundary
            if i > 0 and i < len(embeddings) - 1:
                direct_coherence = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                
                # Mark as boundary if significant coherence drop or low direct coherence
                if (coherence_drop > threshold * 0.3 or direct_coherence < threshold):
                    boundaries.append(i)
        
        # Add boundaries based on structural markers (preserving document order)
        for i, sentence in enumerate(sentences):
            sentence_clean = sentence.strip()
            
            # Detect section headers and structural elements
            section_patterns = [
                r'^\d+\.\s*[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]',  # "1. INTRODUCTION"
                r'^[IVXLCDM]+\.\s*[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ]',  # Roman numerals
                r'^[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-ZÁÀÂÃÉÊÍÓÔÕÚÇ\s]{8,}$',  # All caps titles
                r'^\s*\d+\.\d+',  # "1.1", "2.3" subsections
                r'^\s*(Abstract|Resumo|Introduction|Introdução|Methodology|Metodologia)s?\s*:?\s*$',
                r'^\s*(Results?|Resultados?|Conclusions?|Conclusões?)s?\s*:?\s*$',
                r'^\s*(Discussion|Discussão|References|Referências)s?\s*:?\s*$',
            ]
            
            for pattern in section_patterns:
                if re.match(pattern, sentence_clean, re.IGNORECASE):
                    if i not in boundaries and i > 2:  # Don't break too early
                        boundaries.append(i)
                    break
        
        # Filter boundaries: remove those too close together (preserve readability)
        filtered_boundaries = []
        min_gap = 3  # Minimum 3 sentences between boundaries
        
        for boundary in sorted(boundaries):
            if not filtered_boundaries or boundary - filtered_boundaries[-1] >= min_gap:
                filtered_boundaries.append(boundary)
        
        # Limit total number of boundaries to prevent over-fragmentation
        max_boundaries = max(2, len(sentences) // 8)  # More conservative boundary count
        return filtered_boundaries[:max_boundaries]
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract important keywords from text
        """
        # Simple keyword extraction based on frequency and technical terms
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Filter technical terms
        technical_words = [w for w in words if w in self._get_technical_vocabulary()]
        
        # Get most frequent words
        from collections import Counter
        word_freq = Counter(words)
        frequent_words = [w for w, count in word_freq.most_common(max_keywords) if count > 1]
        
        # Combine and deduplicate
        keywords = list(set(technical_words + frequent_words))
        return keywords[:max_keywords]
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text
        """
        if self.nlp is None:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            return entities
        except Exception:
            return []
    
    def _get_technical_vocabulary(self) -> set:
        """Get set of technical terms for keyword extraction"""
        return {
            'insim', 'permeabilidade', 'permeability', 'porosidade', 'porosity',
            'saturação', 'saturation', 'viscosidade', 'viscosity', 'densidade',
            'density', 'simulação', 'simulation', 'modelo', 'model',
            'reservatório', 'reservoir', 'petróleo', 'oil', 'gás', 'gas',
            'pressão', 'pressure', 'fluxo', 'flow', 'produção', 'production',
            'injeção', 'injection', 'poço', 'well', 'campo', 'field',
            'eclipse', 'cmg', 'simulador', 'simulator', 'numérico', 'numerical'
        }


class SemanticChunker:
    """
    Advanced semantic chunking system that creates meaningful, coherent chunks
    """
    
    def __init__(self, 
                 embedder_model: Optional[SentenceTransformer] = None,
                 target_chunk_size: int = 700,
                 max_chunk_size: int = 1000,
                 overlap_size: int = 120,
                 semantic_threshold: float = 0.5):
        """
        Initialize semantic chunker
        
        Args:
            embedder_model: SentenceTransformer for semantic analysis
            target_chunk_size: Target size for chunks
            max_chunk_size: Maximum allowed chunk size
            overlap_size: Overlap between chunks
            semantic_threshold: Threshold for semantic boundaries
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.semantic_threshold = semantic_threshold
        
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.coherence_analyzer = SemanticCoherenceAnalyzer(embedder_model)
    
    def chunk_document(self, 
                      text: str, 
                      metadata: Optional[Dict[str, Any]] = None) -> List[SemanticChunk]:
        """
        Create semantic chunks from document text
        
        Args:
            text: Document text
            metadata: Document metadata
        
        Returns:
            List of semantic chunks
        """
        if metadata is None:
            metadata = {}
        
        print(f"🔄 Semantic chunking: {len(text)} characters")
        
        # Analyze document structure
        sections = self.structure_analyzer.analyze_structure(text)
        print(f"   Found {len(sections)} document sections")
        
        # Split into sentences for semantic analysis
        sentences = self._split_into_sentences(text)
        print(f"   Split into {len(sentences)} sentences")
        
        # Calculate semantic boundaries
        semantic_boundaries = self.coherence_analyzer.calculate_semantic_boundaries(
            sentences, 
            threshold=self.semantic_threshold
        )
        print(f"   Found {len(semantic_boundaries)} semantic boundaries")
        
        # Create chunks based on structure and semantics
        chunks = self._create_semantic_chunks(
            text, 
            sentences, 
            sections, 
            semantic_boundaries, 
            metadata
        )
        
        print(f"✅ Created {len(chunks)} semantic chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple approaches"""
        # Simple sentence splitting with improved patterns
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Filter very short sentences
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def _create_semantic_chunks(self, 
                               text: str,
                               sentences: List[str],
                               sections: List[Dict[str, Any]],
                               boundaries: List[int],
                               metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """
        Create chunks based on semantic and structural analysis
        """
        chunks = []
        current_chunk_text = ""
        current_chunk_start = 0
        sentence_idx = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk_text + " " + sentence if current_chunk_text else sentence
            
            # Check if we should create a chunk
            should_chunk = (
                len(potential_chunk) > self.max_chunk_size or
                (len(potential_chunk) > self.target_chunk_size and sentence_idx in boundaries) or
                self._is_section_boundary(sentence, sections)
            )
            
            if should_chunk and current_chunk_text:
                # Create chunk
                chunk = self._create_chunk(
                    current_chunk_text,
                    current_chunk_start,
                    current_chunk_start + len(current_chunk_text),
                    sections,
                    metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text, self.overlap_size)
                current_chunk_text = overlap_text + " " + sentence if overlap_text else sentence
                current_chunk_start += len(current_chunk_text) - len(overlap_text) - len(sentence)
            else:
                # Add to current chunk
                current_chunk_text = potential_chunk
            
            sentence_idx += 1
        
        # Create final chunk
        if current_chunk_text:
            chunk = self._create_chunk(
                current_chunk_text,
                current_chunk_start,
                current_chunk_start + len(current_chunk_text),
                sections,
                metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, 
                     text: str,
                     start_idx: int,
                     end_idx: int,
                     sections: List[Dict[str, Any]],
                     metadata: Dict[str, Any]) -> SemanticChunk:
        """Create a semantic chunk with all metadata"""
        
        # Determine section type and importance
        section_type, importance = self._get_section_info(start_idx, sections)
        
        # Extract keywords and entities
        keywords = self.coherence_analyzer.extract_keywords(text)
        entities = self.coherence_analyzer.extract_entities(text)
        
        # Calculate semantic score
        semantic_score = self._calculate_semantic_score(text, section_type, keywords)
        
        # Create enhanced metadata
        chunk_metadata = {
            **metadata,
            'section_type': section_type,
            'importance': importance,
            'chunk_length': len(text),
            'word_count': len(text.split()),
            'has_technical_terms': any(kw in text.lower() for kw in keywords),
            'priority_section': 'abstract' if section_type == 'abstract' else 
                              'definition' if any(term in text.lower() for term in ['definição', 'definition', 'conceito']) else
                              'equation' if section_type == 'equation' else 'regular'
        }
        
        return SemanticChunk(
            chunk_id=str(uuid.uuid4()),
            content=text,
            start_idx=start_idx,
            end_idx=end_idx,
            semantic_score=semantic_score,
            section_type=section_type,
            keywords=keywords,
            entities=entities,
            metadata=chunk_metadata
        )
    
    def _get_section_info(self, 
                         start_idx: int, 
                         sections: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Get section type and importance for chunk position"""
        for section in sections:
            if section['start'] <= start_idx < section['end']:
                return section['type'], section['importance']
        return 'body', 1.0
    
    def _calculate_semantic_score(self, 
                                 text: str, 
                                 section_type: str, 
                                 keywords: List[str]) -> float:
        """Calculate semantic importance score for chunk"""
        base_score = 0.5
        
        # Section type bonus
        section_bonus = {
            'abstract': 0.3,
            'introduction': 0.1,
            'methodology': 0.2,
            'results': 0.25,
            'conclusion': 0.2,
            'equation': 0.35,
            'title': 0.25
        }.get(section_type, 0.0)
        
        # Technical keywords bonus
        technical_bonus = min(0.2, len(keywords) * 0.02)
        
        # Length penalty for very short or very long chunks
        length_penalty = 0.0
        if len(text) < 100:
            length_penalty = -0.1
        elif len(text) > 600:
            length_penalty = -0.05
        
        return max(0.0, min(1.0, base_score + section_bonus + technical_bonus + length_penalty))
    
    def _is_section_boundary(self, 
                           sentence: str, 
                           sections: List[Dict[str, Any]]) -> bool:
        """Check if sentence represents a section boundary"""
        return any(sentence.strip() == section.get('title', '') for section in sections)
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of chunk"""
        if len(text) <= overlap_size:
            return text
        
        # Try to find sentence boundary near the overlap point
        target_start = len(text) - overlap_size
        sentence_end = text.rfind('.', target_start)
        
        if sentence_end != -1 and sentence_end > target_start - 50:
            return text[sentence_end + 1:].strip()
        else:
            return text[-overlap_size:].strip()
    
    def get_chunking_stats(self, chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Get statistics about the chunking results"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        semantic_scores = [chunk.semantic_score for chunk in chunks]
        section_types = [chunk.section_type for chunk in chunks]
        
        from collections import Counter
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean(chunk_sizes),
            'median_chunk_size': np.median(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_semantic_score': np.mean(semantic_scores),
            'section_distribution': dict(Counter(section_types)),
            'high_quality_chunks': len([s for s in semantic_scores if s > 0.7]),
            'technical_chunks': len([c for c in chunks if c.metadata.get('has_technical_terms', False)])
        }


# Global semantic chunker instance
_global_semantic_chunker = None

def get_semantic_chunker(embedder_model: Optional[SentenceTransformer] = None) -> SemanticChunker:
    """
    Get or create global semantic chunker instance
    """
    global _global_semantic_chunker
    if _global_semantic_chunker is None:
        _global_semantic_chunker = SemanticChunker(embedder_model)
    return _global_semantic_chunker