"""
Academic Citation Formatter
Formata citações no estilo acadêmico/científico padrão
"""

import re
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AcademicReference:
    """Representa uma referência acadêmica formatada"""
    authors: List[str]
    title: str
    journal: Optional[str] = None
    volume: Optional[str] = None
    number: Optional[str] = None
    pages: Optional[str] = None
    year: Optional[int] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    type: str = "article"  # article, book, thesis, conference
    
    def format_citation(self, citation_number: int) -> str:
        """Formata a citação no estilo acadêmico"""
        
        # Format authors
        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]} e {self.authors[1]}"
        elif len(self.authors) <= 3:
            author_str = ", ".join(self.authors[:-1]) + f", e {self.authors[-1]}"
        else:
            author_str = f"{self.authors[0]} et al."
        
        # Base citation with number
        citation = f"[{citation_number}]: {author_str}. \"{self.title}\""
        
        # Add publication details based on type
        if self.type == "article" and self.journal:
            citation += f". {self.journal}"
            if self.volume:
                citation += f", vol. {self.volume}"
            if self.number:
                citation += f", no. {self.number}"
            if self.year:
                citation += f", {self.year}"
            if self.pages:
                citation += f", pp. {self.pages}"
        
        elif self.type == "thesis":
            if self.publisher:
                citation += f". {self.publisher}"
            if self.year:
                citation += f", {self.year}"
        
        elif self.type == "book":
            if self.publisher:
                citation += f". {self.publisher}"
            if self.year:
                citation += f", {self.year}"
        
        elif self.type == "conference":
            if self.journal:  # Conference proceedings
                citation += f". In {self.journal}"
            if self.year:
                citation += f", {self.year}"
        
        citation += "."
        
        if self.doi:
            citation += f" DOI: {self.doi}"
        
        return citation

class AcademicCitationExtractor:
    """Extrai informações acadêmicas de documentos PDF"""
    
    def __init__(self):
        # Patterns for extracting academic information
        self.author_patterns = [
            r'(?:by|author[s]?:?\s*)\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+,?\s+[A-Z]\.(?:\s+[A-Z]\.)*)',
            r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)',
        ]
        
        self.title_patterns = [
            r'(?:title:?\s*)?([A-Z][^.!?]*[.!?])',
            r'^([A-Z][^.!?]{10,100}[.!?])',
        ]
        
        self.journal_patterns = [
            r'(?:published\s+in|journal\s+of|proceedings\s+of)\s+([A-Za-z\s&]+)',
            r'(Journal\s+of\s+[A-Za-z\s&]+)',
            r'(Proceedings\s+of\s+[A-Za-z\s&]+)',
            r'([A-Za-z\s&]+\s+Journal)',
        ]
        
        self.year_patterns = [
            r'(\d{4})',
            r'(?:year|published|copyright)[\s:]*(\d{4})',
        ]
        
        # Common academic terms in Portuguese and English
        self.academic_keywords = {
            'thesis': ['tese', 'thesis', 'dissertação', 'dissertation'],
            'article': ['artigo', 'article', 'paper', 'estudo', 'study'],
            'book': ['livro', 'book', 'manual', 'handbook'],
            'conference': ['conferência', 'conference', 'simpósio', 'symposium', 'congresso']
        }
    
    def extract_from_pdf_metadata(self, pdf_path: str) -> Optional[AcademicReference]:
        """Extrai informações dos metadados do PDF"""
        import os
        if not pdf_path or not pdf_path.strip() or not os.path.exists(pdf_path):
            return None

        # Try PyMuPDF first (reads structured Info dict)
        try:
            import fitz
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            title = (meta.get('title') or '').strip()
            author = (meta.get('author') or '').strip()
            doc.close()
            if title and author:
                authors = self._parse_authors(author)
                doc_type = self._detect_document_type(title, '')
                return AcademicReference(authors=authors, title=title, type=doc_type)
        except Exception:
            pass

        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    metadata = pdf.metadata
                    title = metadata.get('Title', '')
                    author = metadata.get('Author', '')
                    subject = metadata.get('Subject', '')
                    if title and author:
                        authors = self._parse_authors(author)
                        doc_type = self._detect_document_type(title, subject)
                        return AcademicReference(authors=authors, title=title, type=doc_type)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("PDF metadata extraction failed for %s: %s", pdf_path, e)

        return None
    
    def extract_from_first_page(self, first_page_text: str, filename: str) -> AcademicReference:
        """Extrai informações da primeira página do documento"""
        
        # Clean and prepare text
        text = self._clean_text(first_page_text)
        lines = text.split('\n')
        
        # Extract information
        authors = self._extract_authors(text, lines)
        title = self._extract_title(text, lines, filename)
        journal = self._extract_journal(text)
        year = self._extract_year(text)
        doc_type = self._detect_document_type(text, filename)
        
        # If no authors found, try to infer from filename
        if not authors:
            authors = self._infer_author_from_filename(filename)
        
        # If no title found, use cleaned filename
        if not title:
            title = self._clean_filename_as_title(filename)
        
        return AcademicReference(
            authors=authors,
            title=title,
            journal=journal,
            year=year,
            type=doc_type
        )
    
    def _clean_text(self, text: str) -> str:
        """Limpa o texto para análise"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def _extract_authors(self, text: str, lines: List[str]) -> List[str]:
        """Extrai autores do texto"""
        authors = []
        
        # Try different patterns
        for pattern in self.author_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                authors.extend([self._clean_author_name(match) for match in matches])
        
        # Look in first few lines for author-like patterns
        for line in lines[:10]:
            if len(line.strip()) > 5 and len(line.strip()) < 100:
                # Check if line looks like author names
                if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z]\.)*\s+[A-Z][a-z]+', line.strip()):
                    authors.append(self._clean_author_name(line.strip()))
        
        # Remove duplicates and return first 5
        unique_authors = []
        for author in authors:
            if author not in unique_authors and len(author) > 3:
                unique_authors.append(author)
        
        return unique_authors[:5]  # Limit to 5 authors
    
    def _extract_title(self, text: str, lines: List[str], filename: str) -> str:
        """Extrai o título do documento"""
        
        # Look for title in first few lines
        for line in lines[:5]:
            line = line.strip()
            if (len(line) > 10 and len(line) < 200 and 
                not line.lower().startswith(('abstract', 'resumo', 'page', 'página')) and
                not re.match(r'^\d+', line)):
                
                # Clean potential title
                title = re.sub(r'\s+', ' ', line).strip()
                if title:
                    return title
        
        # Fallback to filename-based title
        return self._clean_filename_as_title(filename)
    
    def _extract_journal(self, text: str) -> Optional[str]:
        """Extrai nome do journal/periódico"""
        for pattern in self.journal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extrai ano de publicação"""
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the most recent reasonable year
                years = [int(y) for y in matches if 1900 <= int(y) <= datetime.now().year]
                if years:
                    return max(years)
        return None
    
    def _detect_document_type(self, text: str, filename: str) -> str:
        """Detecta o tipo de documento"""
        text_lower = (text + " " + filename).lower()
        
        for doc_type, keywords in self.academic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return doc_type
        
        return "article"  # Default
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string from metadata"""
        if not author_string:
            return []

        authors = re.split(r'[;&]|\s+and\s+|\s+e\s+', author_string)
        cleaned = [self._clean_author_name(a.strip()) for a in authors if a.strip()]
        return cleaned[:5]
    
    def _clean_author_name(self, name: str) -> str:
        """Limpa nome do autor"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Capitalize properly
        words = name.split()
        cleaned_words = []
        
        for word in words:
            if len(word) == 1 or (len(word) == 2 and word.endswith('.')):
                # Initial
                cleaned_words.append(word.upper())
            else:
                # Full name
                cleaned_words.append(word.capitalize())
        
        return ' '.join(cleaned_words)
    
    def _infer_author_from_filename(self, filename: str) -> List[str]:
        """Tenta inferir autor do nome do arquivo"""
        # Remove extension and common prefixes
        base_name = os.path.splitext(filename)[0]
        base_name = re.sub(r'_compressed|_final|_v\d+', '', base_name)
        
        # Look for name patterns
        name_match = re.search(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', base_name)
        if name_match:
            return [self._clean_author_name(name_match.group(1))]
        
        return ["Autor não identificado"]
    
    def _clean_filename_as_title(self, filename: str) -> str:
        """Usa o nome do arquivo como título limpo"""
        # Remove extension and common suffixes
        title = os.path.splitext(filename)[0]
        title = re.sub(r'_compressed|_final|_v\d+', '', title)
        title = re.sub(r'[-_]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Capitalize
        return title.title()

class AcademicCitationFormatter:
    """Formata citações no texto e lista de referências"""
    
    def __init__(self):
        self.extractor = AcademicCitationExtractor()
        self.citation_counter = 0
        self.references = {}  # source_id -> AcademicReference
        self.source_to_citation = {}  # source_id -> citation_number
    
    def add_source(self, source_id: str, source_data: Dict[str, Any]) -> int:
        """Adiciona uma fonte e retorna o número da citação"""
        
        if source_id in self.source_to_citation:
            return self.source_to_citation[source_id]
        
        # Extract academic information
        filename = source_data.get('filename', source_data.get('source', ''))
        first_page_text = source_data.get('preview', source_data.get('content', ''))
        
        # Try to extract from metadata first, then from content
        reference = None
        file_path = source_data.get('file_path', '')
        if file_path and file_path.strip():
            reference = self.extractor.extract_from_pdf_metadata(file_path)
        
        if not reference:
            reference = self.extractor.extract_from_first_page(first_page_text, filename)
        
        # Assign citation number
        self.citation_counter += 1
        citation_number = self.citation_counter
        
        self.references[source_id] = reference
        self.source_to_citation[source_id] = citation_number
        
        return citation_number
    
    def format_reference_list(self) -> str:
        """Formata a lista de referências"""
        if not self.references:
            return ""
        
        references_text = "\n## Referências:\n\n"
        
        # Sort by citation number
        sorted_refs = sorted(
            [(self.source_to_citation[source_id], ref) for source_id, ref in self.references.items()],
            key=lambda x: x[0]
        )
        
        for citation_num, reference in sorted_refs:
            references_text += reference.format_citation(citation_num) + "\n\n"
        
        return references_text
    
    def get_citation_number(self, source_id: str) -> Optional[int]:
        """Retorna o número da citação para uma fonte"""
        return self.source_to_citation.get(source_id)
    
    def clear(self):
        """Limpa todas as citações"""
        self.citation_counter = 0
        self.references.clear()
        self.source_to_citation.clear()