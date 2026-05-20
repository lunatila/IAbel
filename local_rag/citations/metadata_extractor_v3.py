"""
Enhanced PDF Metadata Extractor for RAG v3
Extracts author, year, and title information from PDFs for academic-style citations
"""

import re
import os
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import pdfplumber


@dataclass
class DocumentMetadata:
    """Metadata for academic citations"""
    authors: List[str]
    year: Optional[int]
    title: str
    filename: str

    def format_citation(self) -> str:
        """Format as (Author, Year) or (Author et al., Year)"""
        if not self.authors:
            # Fallback to filename without extension
            author = os.path.splitext(self.filename)[0]
            year_str = str(self.year) if self.year else "n.d."
            return f"({author}, {year_str})"

        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]} and {self.authors[1]}"
        else:
            # More than 2 authors: use et al.
            author_str = f"{self.authors[0]} et al."

        year_str = str(self.year) if self.year else "n.d."
        return f"({author_str}, {year_str})"

    def format_inline_citation(self) -> str:
        """Format for inline use in text"""
        return self.format_citation()


class MetadataExtractorV3:
    """Enhanced metadata extractor for RAG v3"""

    def __init__(self):
        # Patterns for extracting author names
        self.author_patterns = [
            # "by John Smith" or "Author: John Smith"
            r'(?:by|author[s]?:?\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            # "Smith, J." or "Smith, J. D."
            r'([A-Z][a-z]+,\s+[A-Z]\.(?:\s+[A-Z]\.)*)',
            # "John Smith" at start of line (first few lines)
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]

        # Patterns for extracting years
        self.year_patterns = [
            r'\b(20\d{2})\b',  # 2000-2099
            r'\b(19\d{2})\b',  # 1900-1999
            r'(?:year|published|copyright)[\s:]*(\d{4})',
        ]

        # Common Portuguese/English first names to validate authors
        self.common_names = {
            'joão', 'maria', 'josé', 'ana', 'paulo', 'carlos', 'pedro', 'lucas',
            'julia', 'mariana', 'fernando', 'ricardo', 'diego', 'gabriel',
            'john', 'mary', 'james', 'robert', 'michael', 'william', 'david',
            'richard', 'joseph', 'thomas', 'charles', 'daniel', 'matthew',
            'emilio', 'dimary', 'malu', 'malú', 'onur'
        }

    def extract_metadata(self, pdf_path: str) -> DocumentMetadata:
        """
        Extract metadata from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            DocumentMetadata object
        """
        filename = os.path.basename(pdf_path)

        # Try multiple extraction methods
        authors = []
        year = None
        title = None

        # 1. Extract from filename first (often most reliable)
        filename_authors, filename_year = self._extract_from_filename(filename)
        if filename_authors:
            authors = filename_authors
        if filename_year:
            year = filename_year

        # 2. Try PDF metadata
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    metadata = pdf.metadata

                    # Extract from metadata
                    if not authors and metadata.get('Author'):
                        metadata_authors = self._parse_author_string(metadata.get('Author', ''))
                        if metadata_authors:
                            authors = metadata_authors

                    if not year and metadata.get('CreationDate'):
                        # CreationDate format: D:20240726...
                        creation_date = metadata.get('CreationDate', '')
                        year_match = re.search(r'D:(\d{4})', creation_date)
                        if year_match:
                            year = int(year_match.group(1))

                    if not title and metadata.get('Title'):
                        title = metadata.get('Title')

                # 3. Extract from first page
                if (not authors or not year or not title) and pdf.pages:
                    first_page_text = pdf.pages[0].extract_text()
                    if first_page_text:
                        if not authors:
                            page_authors = self._extract_authors_from_text(first_page_text)
                            if page_authors:
                                authors = page_authors

                        if not year:
                            page_year = self._extract_year_from_text(first_page_text)
                            if page_year:
                                year = page_year

                        if not title:
                            page_title = self._extract_title_from_text(first_page_text)
                            if page_title:
                                title = page_title

        except Exception as e:
            print(f"   ⚠️ Error extracting PDF metadata: {e}")

        # Fallbacks
        if not authors:
            # Use filename as author
            authors = [self._clean_filename_as_author(filename)]

        if not title:
            # Use filename as title
            title = self._clean_filename_as_title(filename)

        # Ensure year is valid
        if year and (year < 1900 or year > 2100):
            year = None

        return DocumentMetadata(
            authors=authors,
            year=year,
            title=title,
            filename=filename
        )

    def _extract_from_filename(self, filename: str) -> Tuple[Optional[List[str]], Optional[int]]:
        """Extract author and year from filename"""
        authors = []
        year = None

        # Remove extension
        name_without_ext = os.path.splitext(filename)[0]

        # Common patterns:
        # "Dimary 2024.pdf" -> Dimary, 2024
        # "Emilio Coutinho - PhD Dissertation_compressed.pdf" -> Emilio Coutinho
        # "Qualificação Dimary.pdf" -> Dimary

        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', name_without_ext)
        if year_match:
            year = int(year_match.group(1))

        # Extract author names (before " - " or before year or before keywords)
        # Remove common keywords
        cleaned = re.sub(r'(?i)(qualificação|dissertation|thesis|manual|compressed|_compressed)', '', name_without_ext)
        cleaned = re.sub(r'[-_]+', ' ', cleaned)

        # Look for capitalized words (likely names)
        words = cleaned.split()
        potential_authors = []

        for i, word in enumerate(words):
            # Stop at year
            if word.isdigit() and len(word) == 4:
                break

            # Check if word starts with capital and is likely a name
            if word and word[0].isupper() and len(word) > 1:
                # Check if it's a known name or follows name pattern
                if (word.lower() in self.common_names or
                    len(word) >= 3 and word.isalpha()):
                    potential_authors.append(word)

        if potential_authors:
            # Join consecutive names (e.g., ["Emilio", "Coutinho"] -> "Emilio Coutinho")
            authors = [' '.join(potential_authors)]

        return (authors if authors else None, year)

    def _parse_author_string(self, author_str: str) -> List[str]:
        """Parse author string from PDF metadata"""
        if not author_str or not author_str.strip():
            return []

        # Common separators: ",", "and", "&", ";"
        authors = re.split(r'[,;&]|\band\b', author_str)
        authors = [a.strip() for a in authors if a.strip()]

        # Clean each author name
        cleaned_authors = []
        for author in authors:
            # Remove titles (Dr., Prof., etc.)
            author = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms)\.?\s+', '', author, flags=re.IGNORECASE)
            author = author.strip()
            if author:
                cleaned_authors.append(author)

        return cleaned_authors[:3]  # Maximum 3 authors

    def _extract_authors_from_text(self, text: str) -> Optional[List[str]]:
        """Extract author names from first page text"""
        lines = text.split('\n')[:15]  # Check first 15 lines

        authors = []
        for line in lines:
            line = line.strip()
            if not line or len(line) > 100:  # Skip empty or too long
                continue

            # Look for author patterns
            for pattern in self.author_patterns:
                matches = re.findall(pattern, line, re.MULTILINE)
                for match in matches:
                    author = match.strip()
                    if self._is_likely_author(author):
                        authors.append(author)

            # Also check for simple capitalized names at line start
            if line and line[0].isupper():
                words = line.split()
                if len(words) >= 2 and len(words) <= 4:
                    # Check if all words are capitalized
                    if all(w[0].isupper() for w in words if w):
                        potential_author = ' '.join(words)
                        if self._is_likely_author(potential_author):
                            authors.append(potential_author)

        return authors[:3] if authors else None  # Max 3 authors

    def _is_likely_author(self, name: str) -> bool:
        """Check if string is likely an author name"""
        if not name or len(name) < 3:
            return False

        # Should contain at least 2 words
        words = name.split()
        if len(words) < 1:
            return False

        # Check for common patterns
        # 1. "Firstname Lastname"
        # 2. "Lastname, F."
        # 3. Known names from our list

        first_word = words[0].lower().rstrip(',')
        if first_word in self.common_names:
            return True

        # Check if it looks like a name (starts with capital, only letters/spaces/commas)
        if re.match(r'^[A-Z][a-zA-Z\s,.]+$', name):
            return True

        return False

    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract publication year from text"""
        # Look for years in first 500 characters
        text_start = text[:500]

        for pattern in self.year_patterns:
            matches = re.findall(pattern, text_start)
            for match in matches:
                try:
                    year = int(match)
                    if 1900 <= year <= 2100:
                        return year
                except ValueError:
                    continue

        return None

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from first page"""
        lines = text.split('\n')

        # Look for title in first 10 lines
        for line in lines[:10]:
            line = line.strip()

            # Skip short lines, page numbers, headers
            if len(line) < 10 or len(line) > 200:
                continue

            # Skip lines that look like metadata
            if re.match(r'^\d+$', line) or line.lower().startswith(('page', 'abstract', 'keywords')):
                continue

            # Title is usually in title case or all caps, and substantial
            if (len(line) > 15 and
                (line.isupper() or line.istitle() or any(c.isupper() for c in line[:20]))):
                # Clean and return
                title = re.sub(r'\s+', ' ', line).strip()
                return title

        return None

    def _clean_filename_as_author(self, filename: str) -> str:
        """Use cleaned filename as author fallback"""
        name = os.path.splitext(filename)[0]
        # Remove common suffixes
        name = re.sub(r'(?i)(_compressed|compressed|\.pdf)', '', name)
        # Remove underscores and hyphens
        name = re.sub(r'[-_]+', ' ', name)
        # Take first few words
        words = name.split()[:3]
        return ' '.join(words)

    def _clean_filename_as_title(self, filename: str) -> str:
        """Use cleaned filename as title fallback"""
        title = os.path.splitext(filename)[0]
        title = re.sub(r'(?i)(_compressed|compressed)', '', title)
        title = re.sub(r'[-_]+', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        return title
