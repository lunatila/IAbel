"""
Enhanced PDF Metadata Extractor for RAG v3
Uses PyMuPDF Info dict as primary source, text heuristics as fallback.
"""

import re
import os
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False
    logger.warning("PyMuPDF not installed — falling back to pdfplumber for metadata extraction.")

# Valid year range for publications
_YEAR_MIN = 1900
_YEAR_MAX = 2026

_YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')
_DOI_RE = re.compile(r'10\.\d{4,}/\S+')
_CREATION_DATE_RE = re.compile(r'D:(\d{4})')

# Words that look like names but are not
_NON_NAME_WORDS = frozenset({
    'abstract', 'introduction', 'conclusion', 'results', 'discussion',
    'references', 'acknowledgements', 'university', 'institute', 'department',
    'school', 'faculty', 'college', 'laboratory', 'center', 'centre',
    'journal', 'proceedings', 'conference', 'symposium', 'workshop',
    'volume', 'number', 'edition', 'chapter', 'section', 'appendix',
    'figure', 'table', 'equation', 'theorem', 'proof', 'example',
    'january', 'february', 'march', 'april', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
    'janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho',
    'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro',
})


@dataclass
class DocumentMetadata:
    """Metadata for academic citations"""
    authors: List[str]
    year: Optional[int]
    title: str
    filename: str
    doi: Optional[str] = None
    confidence: float = 0.0  # fraction of fields sourced from PDF Info dict

    def format_citation(self) -> str:
        """Format as (Author, Year) or (Author et al., Year)"""
        if not self.authors:
            author = os.path.splitext(self.filename)[0]
            year_str = str(self.year) if self.year else "n.d."
            return f"({author}, {year_str})"

        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]} and {self.authors[1]}"
        else:
            author_str = f"{self.authors[0]} et al."

        year_str = str(self.year) if self.year else "n.d."
        return f"({author_str}, {year_str})"

    def format_inline_citation(self) -> str:
        return self.format_citation()


def _is_valid_year(value: int) -> bool:
    return _YEAR_MIN <= value <= _YEAR_MAX


def _is_valid_field(value: Optional[str]) -> bool:
    """Return True if a metadata string field contains genuinely useful content."""
    if not value or not value.strip():
        return False
    v = value.strip()
    if len(v) < 3:
        return False
    if v.lower() in ('unknown', 'none', 'n/a', 'untitled', 'author'):
        return False
    # Reject UUID-like strings
    if re.match(r'^[0-9a-f\-]{32,}$', v, re.IGNORECASE):
        return False
    return True


def _looks_like_name_token(token: str) -> bool:
    """Return True if token could be a component of a human name."""
    if len(token) < 3:
        return False
    if not token[0].isupper():
        return False
    if not re.match(r"^[A-Za-záàâãäéèêëíìîïóòôõöúùûüçñA-Z'\-]+$", token):
        return False
    if token.lower() in _NON_NAME_WORDS:
        return False
    return True


def _validate_author(name: str) -> bool:
    """Return True if name looks like a real person name."""
    name = name.strip()
    if len(name) < 5:
        return False
    if any(c.isdigit() for c in name):
        return False
    tokens = name.replace(',', ' ').split()
    if len(tokens) < 2:
        return False
    return any(_looks_like_name_token(t) for t in tokens if len(t) >= 3)


def _parse_year_from_date_string(date_str: str) -> Optional[int]:
    """Extract year from PDF CreationDate format 'D:YYYYMMDD...'."""
    m = _CREATION_DATE_RE.search(date_str)
    if m:
        y = int(m.group(1))
        return y if _is_valid_year(y) else None
    return None


def _extract_years_from_text(text: str) -> List[int]:
    return [int(m) for m in _YEAR_RE.findall(text) if _is_valid_year(int(m))]


def _extract_doi_from_text(text: str) -> Optional[str]:
    m = _DOI_RE.search(text[:3000])
    if m:
        return m.group(0).rstrip('.,;)')
    return None


def _split_author_string(raw: str) -> List[str]:
    """Split a raw author metadata field into individual name strings."""
    parts = re.split(r'\s*;\s*|\s+and\s+|\s*&\s*', raw, flags=re.IGNORECASE)
    result = []
    for part in parts:
        part = part.strip()
        part = re.sub(r'\b(Dr|Prof|Mr|Mrs|Ms)\.?\s+', '', part, flags=re.IGNORECASE).strip()
        if part:
            result.append(part)
    return result


class MetadataExtractorV3:
    """
    Extracts author, year, title, and DOI from PDF files.
    Primary source: PyMuPDF Info dict (structured PDF metadata).
    Fallback: text heuristics on the first 3 pages.
    """

    def extract_metadata(self, pdf_path: str) -> DocumentMetadata:
        filename = os.path.basename(pdf_path)
        authors: List[str] = []
        year: Optional[int] = None
        title: Optional[str] = None
        doi: Optional[str] = None
        structured_fields = 0
        total_fields = 3  # author, year, title

        if _FITZ_AVAILABLE:
            authors, year, title, doi, structured_fields = self._extract_with_fitz(pdf_path)
        else:
            authors, year, title = self._extract_with_pdfplumber(pdf_path, [], None, None)

        # Fill any missing fields with text heuristics
        if not authors or not year or not title or not doi:
            authors, year, title, doi = self._extract_with_heuristics(
                pdf_path, authors, year, title, doi
            )

        # Filename fallback
        if not authors or not year:
            fn_authors, fn_year = self._extract_from_filename(filename)
            if not authors and fn_authors:
                authors = fn_authors
            if not year and fn_year:
                year = fn_year

        if not authors:
            authors = [self._clean_filename_as_author(filename)]
        if not title:
            title = self._clean_filename_as_title(filename)
        if year and not _is_valid_year(year):
            year = None

        confidence = structured_fields / total_fields if _FITZ_AVAILABLE else 0.0

        return DocumentMetadata(
            authors=authors,
            year=year,
            title=title,
            filename=filename,
            doi=doi,
            confidence=confidence,
        )

    def _extract_with_fitz(
        self, pdf_path: str
    ) -> Tuple[List[str], Optional[int], Optional[str], Optional[str], int]:
        """Read PDF Info dict with PyMuPDF."""
        authors: List[str] = []
        year: Optional[int] = None
        title: Optional[str] = None
        doi: Optional[str] = None
        structured_count = 0

        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata

            raw_title = meta.get('title', '')
            if _is_valid_field(raw_title):
                title = raw_title.strip()
                structured_count += 1

            raw_author = meta.get('author', '')
            if _is_valid_field(raw_author):
                parsed = _split_author_string(raw_author)
                valid = [a for a in parsed if _validate_author(a)]
                if valid:
                    authors = valid[:5]
                    structured_count += 1

            for date_key in ('creationDate', 'modDate'):
                raw_date = meta.get(date_key, '')
                if raw_date:
                    y = _parse_year_from_date_string(raw_date)
                    if y:
                        year = y
                        structured_count += 1
                        break

            if doc.page_count > 0:
                first_page_text = doc[0].get_text()
                doi = _extract_doi_from_text(first_page_text)

            doc.close()
        except Exception as e:
            logger.warning("PyMuPDF failed to read '%s': %s", pdf_path, e)

        return authors, year, title, doi, structured_count

    def _extract_with_pdfplumber(
        self,
        pdf_path: str,
        existing_authors: List[str],
        existing_year: Optional[int],
        existing_title: Optional[str],
    ) -> Tuple[List[str], Optional[int], Optional[str]]:
        """Fallback when PyMuPDF is unavailable."""
        authors = list(existing_authors)
        year = existing_year
        title = existing_title

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                meta = pdf.metadata or {}

                if not authors and _is_valid_field(meta.get('Author')):
                    parsed = _split_author_string(meta['Author'])
                    valid = [a for a in parsed if _validate_author(a)]
                    if valid:
                        authors = valid[:5]

                if not year and _is_valid_field(meta.get('CreationDate')):
                    y = _parse_year_from_date_string(meta['CreationDate'])
                    if y:
                        year = y

                if not title and _is_valid_field(meta.get('Title')):
                    title = meta['Title'].strip()
        except Exception as e:
            logger.warning("pdfplumber failed to read '%s': %s", pdf_path, e)

        return authors, year, title

    def _extract_with_heuristics(
        self,
        pdf_path: str,
        existing_authors: List[str],
        existing_year: Optional[int],
        existing_title: Optional[str],
        existing_doi: Optional[str],
    ) -> Tuple[List[str], Optional[int], Optional[str], Optional[str]]:
        """Text-based heuristics on the first 3 pages."""
        authors = list(existing_authors)
        year = existing_year
        title = existing_title
        doi = existing_doi

        try:
            if _FITZ_AVAILABLE:
                doc = fitz.open(pdf_path)
                pages_text = [doc[i].get_text() for i in range(min(3, doc.page_count))]
                doc.close()
            else:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    pages_text = [p.extract_text() or '' for p in pdf.pages[:3]]
        except Exception as e:
            logger.warning("Heuristic text extraction failed for '%s': %s", pdf_path, e)
            return authors, year, title, doi

        combined = '\n'.join(pages_text)

        if not doi:
            doi = _extract_doi_from_text(combined)

        if not year:
            years = _extract_years_from_text(combined[:1000])
            year = max(years) if years else None

        if not title and pages_text:
            title = self._extract_title_from_text(pages_text[0])

        if not authors and pages_text:
            authors = self._extract_authors_from_text(pages_text[0])

        return authors, year, title, doi

    def _extract_authors_from_text(self, text: str) -> List[str]:
        """Heuristic author extraction from page text."""
        lines = text.split('\n')[:20]
        found: List[str] = []
        # Match lines that start with a capital-letter name pattern
        name_re = re.compile(
            r'^([A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+'
            r'(?:[\s\-][A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç]+)+)'
        )
        for line in lines:
            line = line.strip()
            if not line or len(line) > 120:
                continue
            m = name_re.match(line)
            if m:
                candidate = m.group(1)
                if _validate_author(candidate) and candidate not in found:
                    found.append(candidate)
            if len(found) >= 5:
                break
        return found

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Heuristic title extraction from first page."""
        for line in text.split('\n')[:15]:
            line = line.strip()
            if 20 <= len(line) <= 200:
                if re.match(r'^\d+$', line):
                    continue
                if line.lower().startswith(('abstract', 'resumo', 'keywords', 'page ')):
                    continue
                return re.sub(r'\s+', ' ', line)
        return None

    def _extract_from_filename(
        self, filename: str
    ) -> Tuple[Optional[List[str]], Optional[int]]:
        """Extract author and year hints from the filename."""
        base = os.path.splitext(filename)[0]
        year_m = _YEAR_RE.search(base)
        year = int(year_m.group(1)) if year_m and _is_valid_year(int(year_m.group(1))) else None

        cleaned = re.sub(
            r'(?i)(qualifica[cç][aã]o|dissertation|thesis|manual|compressed)', '', base
        )
        cleaned = re.sub(r'[-_]+', ' ', cleaned).strip()

        name_tokens = [
            t for t in cleaned.split()
            if t and not t.isdigit() and len(t) > 2 and _looks_like_name_token(t)
        ]
        authors = [' '.join(name_tokens)] if len(name_tokens) >= 2 else None
        return authors, year

    def _clean_filename_as_author(self, filename: str) -> str:
        name = os.path.splitext(filename)[0]
        name = re.sub(r'(?i)(_compressed|compressed)', '', name)
        name = re.sub(r'[-_]+', ' ', name)
        return ' '.join(name.split()[:3])

    def _clean_filename_as_title(self, filename: str) -> str:
        title = os.path.splitext(filename)[0]
        title = re.sub(r'(?i)(_compressed|compressed)', '', title)
        title = re.sub(r'[-_]+', ' ', title)
        return re.sub(r'\s+', ' ', title).strip()
