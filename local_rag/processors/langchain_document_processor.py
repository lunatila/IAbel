"""
Document Processor using LangChain loaders
Supports PDFs, DOCs, TXT, and other document formats with robust extraction
"""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import (
    PyPDFLoader, 
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

@dataclass
class LangChainDocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    page_number: Optional[int]
    document_path: str

class LangChainDocumentProcessor:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Technical terms dictionary for text cleaning
        self.technical_terms = {
            # Common reservoir engineering terms that get stuck together
            'volume': ['volumes', 'volumetria', 'volumétrico'],
            'poro': ['poros', 'porosidade', 'poroso'],
            'permeabilidade': ['permeável', 'permeabilidades'],
            'pressão': ['pressões', 'pressurização'],
            'saturação': ['saturações', 'saturado'],
            'reservatório': ['reservatórios'],
            'poço': ['poços', 'wellbore'],
            'fluido': ['fluidos', 'fluidodinâmica'],
            'simulação': ['simulador', 'simuladores'],
            'modelo': ['modelos', 'modelagem'],
            'equação': ['equações'],
            'parâmetro': ['parâmetros'],
            'entre': ['entretanto', 'entrepoços'],
            'características': ['caracterizam', 'caracterização'],
            'propriedades': ['propriedade'],
            'análise': ['analisar', 'analítico'],
            'método': ['métodos', 'metodologia'],
            'resultado': ['resultados'],
            'conclusão': ['conclusões'],
            'discussão': ['discussões']
        }
        
        # Build regex patterns for common word combinations
        self.word_break_patterns = [
            # Portuguese prepositions and articles that get stuck
            (r'([a-z])de([A-Z])', r'\1 de \2'),
            (r'([a-z])da([A-Z])', r'\1 da \2'),
            (r'([a-z])do([A-Z])', r'\1 do \2'),
            (r'([a-z])para([A-Z])', r'\1 para \2'),
            (r'([a-z])com([A-Z])', r'\1 com \2'),
            (r'([a-z])que([A-Z])', r'\1 que \2'),
            (r'([a-z])entre([A-Z])', r'\1 entre \2'),
            (r'([a-z])são([A-Z])', r'\1 são \2'),
            (r'([a-z])este([A-Z])', r'\1 este \2'),
            (r'([a-z])essa([A-Z])', r'\1 essa \2'),
            (r'([a-z])esse([A-Z])', r'\1 esse \2'),
            (r'([a-z])uma([A-Z])', r'\1 uma \2'),
            # Technical terms that commonly get stuck
            (r'([a-z])volume([A-Z])', r'\1 volume \2'),
            (r'([a-z])poro([A-Z])', r'\1 poro \2'),
            (r'([a-z])pressão([A-Z])', r'\1 pressão \2'),
            (r'([a-z])modelo([A-Z])', r'\1 modelo \2'),
            (r'([a-z])simulação([A-Z])', r'\1 simulação \2'),
            (r'([a-z])método([A-Z])', r'\1 método \2'),
            (r'([a-z])análise([A-Z])', r'\1 análise \2'),
            (r'([a-z])resultado([A-Z])', r'\1 resultado \2'),
        ]
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Better separation hierarchy
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document using appropriate LangChain loader based on file extension
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        print(f"📄 Loading document: {os.path.basename(file_path)} ({file_extension})")
        
        try:
            if file_extension == '.pdf':
                # Try PyPDFLoader first (faster), fallback to UnstructuredPDFLoader
                try:
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    print(f"   ✅ Loaded with PyPDFLoader: {len(documents)} pages")
                except Exception as e:
                    print(f"   ⚠️ PyPDFLoader failed, trying UnstructuredPDFLoader: {e}")
                    loader = UnstructuredPDFLoader(file_path)
                    documents = loader.load()
                    print(f"   ✅ Loaded with UnstructuredPDFLoader: {len(documents)} pages")
            
            elif file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css']:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                print(f"   ✅ Loaded text file: {len(documents)} documents")
            
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
                documents = loader.load()
                print(f"   ✅ Loaded CSV file: {len(documents)} rows")
            
            elif file_extension == '.json':
                loader = JSONLoader(file_path, jq_schema='.')
                documents = loader.load()
                print(f"   ✅ Loaded JSON file: {len(documents)} objects")
            
            else:
                # Try as text file for unknown extensions
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                print(f"   ✅ Loaded as text file: {len(documents)} documents")
            
            return documents
        
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {e}")
            return []

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text with focus on technical documents
        """
        if not text:
            return ""
        
        # Step 1: Normalize line breaks and remove excessive whitespace
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Step 2: Fix hyphenated words across lines (preserve technical terms)
        text = re.sub(r'([a-záàâãéêíóôõúç])-\n([a-záàâãéêíóôõúç])', r'\1\2', text)
        
        # Step 3: Join words split across lines without hyphens
        text = re.sub(r'([a-záàâãéêíóôõúç])\n([a-záàâãéêíóôõúç])', r'\1 \2', text)
        
        # Step 4: Apply technical word break patterns
        for pattern, replacement in self.word_break_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Step 5: Fix common PDF extraction issues
        # Add space after punctuation if missing
        text = re.sub(r'([.!?:;,])([A-Za-záàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ])', r'\1 \2', text)
        
        # Add space between lowercase and uppercase letters (but be careful with acronyms)
        text = re.sub(r'([a-záàâãéêíóôõúç])([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç])', r'\1 \2', text)
        
        # Add space between letters and numbers
        text = re.sub(r'([a-zA-Záàâãéêíóôõúç])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Záàâãéêíóôõúç])', r'\1 \2', text)
        
        # Step 6: Specific technical terms correction
        technical_corrections = [
            (r'quecaracterizam', 'que caracterizam'),
            (r'essesvolumes', 'esses volumes'),
            (r'deporos', 'de poros'),
            (r'entrepoços', 'entre poços'),
            (r'sãoparâmetros', 'são parâmetros'),
            (r'demodelo', 'de modelo'),
            (r'dosimulador', 'do simulador'),
            (r'parareservatório', 'para reservatório'),
            (r'dopoço', 'do poço'),
            (r'nafase', 'na fase'),
            (r'dopetróleo', 'do petróleo'),
            (r'narocha', 'na rocha'),
            (r'dapressão', 'da pressão'),
            (r'datemperatura', 'da temperatura'),
            (r'dapermeabilidade', 'da permeabilidade'),
            (r'daporosidade', 'da porosidade'),
            (r'dosaturation', 'do saturation'),
            (r'thepressure', 'the pressure'),
            (r'oilwater', 'oil water'),
            (r'gaswater', 'gas water'),
        ]
        
        for pattern, replacement in technical_corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Step 7: Clean up mathematical expressions and preserve them
        text = re.sub(r'([=<>≤≥])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([=<>≤≥])', r'\1 \2', text)
        
        # Step 8: Final cleanup
        text = re.sub(r' +', ' ', text)
        
        # Clean up unwanted characters but preserve technical symbols
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\/\%\$\#\@\&\*\+\=\<\>\|\~\`\^\"\'\u00C0-\u017F≤≥≠∞∂∇±×÷°]', '', text)
        
        # Ensure proper spacing around parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text.strip()

    def extract_title_from_metadata(self, document: Document) -> str:
        """
        Extract document title from LangChain document metadata
        """
        metadata = document.metadata
        
        # Try different metadata fields for title
        title = (metadata.get('title') or 
                metadata.get('Title') or 
                metadata.get('subject') or 
                metadata.get('Subject'))
        
        if title:
            return title.strip()
        
        # Extract from source filename if no title in metadata
        source = metadata.get('source', '')
        if source:
            return os.path.splitext(os.path.basename(source))[0]
        
        return "Documento sem título"

    def detect_priority_section(self, content: str, page_num: Optional[int] = None) -> str:
        """
        Detect if chunk is in a priority section with expanded analysis
        """
        content_lower = content.lower()
        
        # Detect abstract/summary sections (can appear anywhere)
        abstract_keywords = [
            'abstract', 'resumo', 'summary', 'síntese',
            'introduction', 'introdução', 'overview', 'visão geral',
            'executive summary', 'sumário executivo'
        ]
        if any(keyword in content_lower for keyword in abstract_keywords):
            return 'abstract'
        
        # Detect methodology sections
        methodology_keywords = [
            'methodology', 'metodologia', 'method', 'método', 'approach', 'abordagem',
            'procedure', 'procedimento', 'technique', 'técnica',
            'experimental', 'experimental setup', 'setup experimental'
        ]
        if any(keyword in content_lower for keyword in methodology_keywords):
            return 'methodology'
        
        # Detect results sections
        results_keywords = [
            'results', 'resultados', 'findings', 'descobertas',
            'analysis', 'análise', 'discussion', 'discussão',
            'conclusions', 'conclusões', 'outcome', 'desfecho'
        ]
        if any(keyword in content_lower for keyword in results_keywords):
            return 'results'
        
        # Enhanced definition detection
        definition_indicators = [
            'define', 'defined', 'definition', 'means', 'refers to', 'is a', 'stands for',
            'definir', 'definido', 'definição', 'significa', 'refere-se', 'é um', 'representa',
            'acronym', 'abbreviation', 'acrônimo', 'abreviação', 'sigla',
            'terminology', 'terminologia', 'glossary', 'glossário'
        ]
        
        # Expanded technical terms for reservoir engineering
        technical_terms = [
            'insim', 'ft', 'simulator', 'simulador', 'eclipse', 'cmg', 'petrel',
            'reservoir', 'reservatório', 'porosity', 'porosidade',
            'permeability', 'permeabilidade', 'saturation', 'saturação',
            'pressure', 'pressão', 'well', 'poço', 'production', 'produção',
            'injection', 'injeção', 'oil', 'óleo', 'gas', 'gás', 'water', 'água',
            'pvt', 'eos', 'black oil', 'compositional', 'thermal',
            'streamline', 'front tracking', 'finite difference', 'finite element'
        ]
        
        has_definition = any(indicator in content_lower for indicator in definition_indicators)
        has_technical_term = any(term in content_lower for term in technical_terms)
        
        if has_definition and has_technical_term:
            return 'definition'
        
        # Detect equations and mathematical content
        equation_indicators = ['=', '∂', '∇', '∫', '∑', 'equation', 'equação', 'formula', 'fórmula']
        if any(pattern in content_lower or pattern in content for pattern in equation_indicators):
            return 'equation'
        
        # Detect figures and tables references
        figure_indicators = [
            'figure', 'figura', 'fig.', 'table', 'tabela', 'tab.',
            'chart', 'gráfico', 'plot', 'diagram', 'diagrama'
        ]
        if any(indicator in content_lower for indicator in figure_indicators):
            return 'figure_table'
        
        # Detect conclusions and key findings
        conclusion_indicators = [
            'conclusion', 'conclusão', 'summary', 'resumo',
            'key findings', 'principais descobertas', 'main results', 'resultados principais',
            'implications', 'implicações', 'recommendations', 'recomendações'
        ]
        if any(indicator in content_lower for indicator in conclusion_indicators):
            return 'conclusion'
        
        # High priority for early pages (likely contains key information)
        if page_num and page_num <= 3:
            return 'early_page'
        
        return 'regular'

    def create_chunks(self, documents: List[Document]) -> List[LangChainDocumentChunk]:
        """
        Create chunks from LangChain documents using enhanced text splitting
        """
        if not documents:
            return []
        
        print(f"🔨 Creating chunks from {len(documents)} documents...")
        
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            # Clean the document content
            cleaned_content = self.clean_text(document.page_content)
            if not cleaned_content.strip():
                continue
            
            # Extract document info
            doc_title = self.extract_title_from_metadata(document)
            source_path = document.metadata.get('source', '')
            page_num = document.metadata.get('page', doc_idx + 1)
            
            # Split into chunks using LangChain's text splitter
            chunks = self.text_splitter.split_text(cleaned_content)
            
            for chunk_idx, chunk_content in enumerate(chunks):
                if not chunk_content.strip():
                    continue
                
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(chunk_content, source_path, chunk_idx)
                
                # Detect section priority
                priority_section = self.detect_priority_section(chunk_content, page_num)
                
                # Create enhanced metadata
                chunk_metadata = {
                    'source': doc_title,
                    'title': doc_title,
                    'filename': os.path.basename(source_path) if source_path else doc_title,
                    'page': page_num if isinstance(page_num, (int, str)) else 'N/A',
                    'chunk_index': chunk_idx,
                    'document_type': 'technical_document',
                    'chunk_type': 'langchain_recursive',
                    'priority_section': priority_section,
                    'document_path': source_path,
                    'original_metadata': document.metadata
                }
                
                chunk = LangChainDocumentChunk(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id,
                    page_number=page_num if isinstance(page_num, int) else None,
                    document_path=source_path
                )
                
                all_chunks.append(chunk)
        
        print(f"🎯 Created {len(all_chunks)} chunks total")
        return all_chunks

    def process_document(self, file_path: str) -> List[LangChainDocumentChunk]:
        """
        Process a single document: load + clean + chunk
        """
        print(f"🔄 Processing: {os.path.basename(file_path)}")
        
        # Load document
        documents = self.load_document(file_path)
        if not documents:
            return []
        
        # Create chunks
        chunks = self.create_chunks(documents)
        
        print(f"✅ Created {len(chunks)} chunks from {os.path.basename(file_path)}")
        return chunks

    def process_directory(self, directory_path: str, supported_extensions: List[str] = None) -> List[LangChainDocumentChunk]:
        """
        Process all supported documents in a directory
        """
        if supported_extensions is None:
            supported_extensions = ['.pdf', '.txt', '.md', '.csv', '.json', '.py', '.js', '.html']
        
        if not os.path.exists(directory_path):
            print(f"❌ Directory not found: {directory_path}")
            return []
        
        print(f"📁 Processing directory: {directory_path}")
        
        all_chunks = []
        files_processed = 0
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Skip directories and unsupported files
            if os.path.isdir(file_path):
                continue
            
            file_extension = Path(filename).suffix.lower()
            if file_extension not in supported_extensions:
                continue
            
            try:
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                files_processed += 1
                
            except Exception as e:
                print(f"   ❌ Error processing {filename}: {e}")
                continue
        
        print(f"🎯 Processed {files_processed} files, created {len(all_chunks)} chunks total")
        return all_chunks

    def _generate_chunk_id(self, content: str, doc_path: str, chunk_index: int) -> str:
        """
        Generate unique ID for chunk
        """
        import time
        filename = os.path.basename(doc_path) if doc_path else "unknown"
        timestamp = str(int(time.time() * 1000000))  # microseconds
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        source = f"{filename}_{chunk_index}_{timestamp}_{content_hash}"
        return hashlib.md5(source.encode()).hexdigest()