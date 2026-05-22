"""
Processador de PDFs para extração e chunking de texto
Especializado em documentos de engenharia de reservatórios
"""

import os
import pdfplumber
import re
from typing import List, Dict, Any
from dataclasses import dataclass
import hashlib

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    page_number: int
    document_path: str

class PDFProcessor:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120, batch_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size  # Process pages in batches to save memory
        
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
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extrai texto de PDF página por página usando pdfplumber
        Otimizado para PDFs grandes
        """
        print(f"📄 Processando PDF: {os.path.basename(pdf_path)}")
        
        # Check file size first
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"   Tamanho do arquivo: {file_size_mb:.1f} MB")
        
        pages = []
        document_title = None
        total_pages = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"   Total de páginas: {total_pages}")
            
            # Try to extract document title from metadata or first page
            try:
                # First try PDF metadata
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    document_title = pdf.metadata.get('Title', None)
                
                # If no metadata title, try to extract from first page
                if not document_title and pdf.pages:
                    first_page_text = pdf.pages[0].extract_text()
                    if first_page_text:
                        document_title = self._extract_title_from_text(first_page_text)
            except Exception as e:
                print(f"   ⚠️ Erro ao extrair título: {e}")
            
            # Fallback to filename without extension
            if not document_title:
                document_title = os.path.splitext(os.path.basename(pdf_path))[0]
            
            print(f"   Título detectado: {document_title}")
            
            # Process pages in batches to manage memory
            if total_pages > self.batch_size:
                return self._extract_text_from_large_pdf(pdf, pdf_path, document_title, total_pages)
            else:
                return self._extract_text_from_small_pdf(pdf, pdf_path, document_title)
    
    def _extract_text_from_small_pdf(self, pdf, pdf_path: str, document_title: str) -> List[Dict[str, Any]]:
        """Process small PDFs normally"""
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            if text:
                text = self._clean_text(text)
                if text.strip():
                    pages.append({
                        'page_number': page_num + 1,
                        'content': text,
                        'document_path': pdf_path,
                        'document_title': document_title
                    })
        return pages
    
    def _extract_text_from_large_pdf(self, pdf, pdf_path: str, document_title: str, total_pages: int) -> List[Dict[str, Any]]:
        """Process large PDFs in batches to manage memory"""
        pages = []
        
        for batch_start in range(0, total_pages, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pages)
            print(f"   📦 Processando páginas {batch_start + 1}-{batch_end} de {total_pages}")
            
            batch_pages = []
            for page_num in range(batch_start, batch_end):
                try:
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    
                    if text:
                        text = self._clean_text(text)
                        if text.strip():
                            batch_pages.append({
                                'page_number': page_num + 1,
                                'content': text,
                                'document_path': pdf_path,
                                'document_title': document_title
                            })
                    
                    # Force garbage collection periodically
                    if page_num % 10 == 0:
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    print(f"   ⚠️ Erro na página {page_num + 1}: {e}")
                    continue
            
            pages.extend(batch_pages)
            print(f"   ✅ Lote processado: {len(batch_pages)} páginas válidas")
        
        print(f"   🎯 Total processado: {len(pages)} páginas de {total_pages}")
        return pages
    
    def _fix_mojibake(self, text: str) -> str:
        """Repair UTF-8 text that was mis-decoded as Latin-1 (Ã© → é)."""
        try:
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    def _clean_text(self, text: str) -> str:
        """
        Limpa e normaliza o texto extraído com foco em documentos técnicos
        """
        if not text:
            return ""

        # Step 0: Fix mojibake before any other processing
        text = self._fix_mojibake(text)

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
        # Fix common stuck technical terms
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
            (r'dosaturation', 'do saturation'),  # English terms
            (r'thepressure', 'the pressure'),
            (r'oilwater', 'oil water'),
            (r'gaswater', 'gas water'),
            # Add more as needed
        ]
        
        for pattern, replacement in technical_corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Step 7: Clean up mathematical expressions and preserve them
        # Preserve mathematical notation
        text = re.sub(r'([=<>≤≥])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([=<>≤≥])', r'\1 \2', text)
        
        # Step 8: Final cleanup
        # Remove excessive spaces but preserve single spaces
        text = re.sub(r' +', ' ', text)
        
        # Clean up unwanted characters but preserve technical symbols
        # Allow more characters including accented characters and mathematical symbols
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\/\%\$\#\@\&\*\+\=\<\>\|\~\`\^\"\'\u00C0-\u017F≤≥≠∞∂∇±×÷°]', '', text)
        
        # Step 9: Post-processing cleanup
        # Remove standalone single characters that might be artifacts
        text = re.sub(r'\b[a-zA-Z]\s+(?=[A-Z])', '', text)  # Remove single letters before capitals
        
        # Ensure proper spacing around parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        return text.strip()
    
    def _extract_title_from_text(self, text: str) -> str:
        """
        Tenta extrair o título do documento do texto da primeira página
        """
        if not text:
            return None
            
        lines = text.split('\n')
        
        # Look for common title patterns in first few lines
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
                
            # Skip headers, page numbers, etc.
            if re.match(r'^\d+$', line) or len(line) < 10:
                continue
                
            # Look for title-like patterns
            if (len(line) > 10 and len(line) < 200 and 
                not line.lower().startswith(('abstract', 'resumo', 'keywords', 'página')) and
                not re.match(r'^\d+\.', line)):  # Not a numbered section
                
                # Clean the potential title
                title = re.sub(r'\s+', ' ', line).strip()
                if title:
                    return title
        
        return None
    
    def create_chunks(self, pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """
        Divide o texto em chunks com overlapping
        """
        print(f"🔨 Criando chunks para {len(pages)} páginas...")
        chunks = []
        
        # Process pages in smaller batches to save memory
        page_batch_size = 20  # Process 20 pages at a time
        
        for batch_start in range(0, len(pages), page_batch_size):
            batch_end = min(batch_start + page_batch_size, len(pages))
            batch_pages = pages[batch_start:batch_end]
            
            print(f"   📦 Processando páginas {batch_start + 1}-{batch_end} ({len(batch_pages)} páginas)")
            batch_chunks = self._create_chunks_for_batch(batch_pages)
            chunks.extend(batch_chunks)
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
            
            print(f"   ✅ Batch processado: {len(batch_chunks)} chunks criados")
        
        print(f"🎯 Total de chunks criados: {len(chunks)}")
        return chunks
    
    def _create_chunks_for_batch(self, pages: List[Dict[str, Any]]) -> List[DocumentChunk]:
        """Create chunks for a batch of pages"""
        chunks = []
        
        for page in pages:
            content = page['content']
            page_num = page['page_number']
            doc_path = page['document_path']
            doc_title = page.get('document_title', os.path.splitext(os.path.basename(doc_path))[0])
            
            # Divide por sentenças E parágrafos para chunks mais semânticos
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            chunk_count = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Se parágrafo é muito grande, divide por sentenças
                if len(paragraph) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk.strip():
                                chunk_id = self._generate_chunk_id(current_chunk, doc_path, chunk_count)
                                chunks.append(DocumentChunk(
                                    content=current_chunk.strip(),
                                    metadata={
                                        'source': doc_title,
                                        'title': doc_title,
                                        'filename': os.path.basename(doc_path),
                                        'page': page_num,
                                        'chunk_index': chunk_count,
                                        'document_type': 'reservoir_engineering',
                                        'chunk_type': 'sentence_split',
                                        'priority_section': self._detect_priority_section(current_chunk, page_num),
                                        'document_path': doc_path
                                    },
                                    chunk_id=chunk_id,
                                    page_number=page_num,
                                    document_path=doc_path
                                ))
                                
                                # Overlap: mantém últimas palavras do chunk anterior
                                overlap_words = current_chunk.split()[-20:]  # 20 palavras de overlap
                                current_chunk = " ".join(overlap_words) + " " + sentence + " "
                                chunk_count += 1
                            else:
                                current_chunk = sentence + " "
                else:
                    # Parágrafo pequeno - tenta adicionar ao chunk atual
                    if len(current_chunk) + len(paragraph) <= self.chunk_size:
                        current_chunk += paragraph + "\n\n"
                    else:
                        # Salva chunk atual se não vazio
                        if current_chunk.strip():
                            chunk_id = self._generate_chunk_id(current_chunk, doc_path, chunk_count)
                            chunks.append(DocumentChunk(
                                content=current_chunk.strip(),
                                metadata={
                                    'source': doc_title,
                                    'title': doc_title,
                                    'filename': os.path.basename(doc_path),
                                    'page': page_num,
                                    'chunk_index': chunk_count,
                                    'document_type': 'reservoir_engineering',
                                    'chunk_type': 'paragraph_based',
                                    'priority_section': self._detect_priority_section(current_chunk, page_num),
                                    'document_path': doc_path
                                },
                                chunk_id=chunk_id,
                                page_number=page_num,
                                document_path=doc_path
                            ))
                            chunk_count += 1
                        
                        # Inicia novo chunk com parágrafo atual
                        current_chunk = paragraph + "\n\n"
            
            # Adiciona último chunk se houver conteúdo
            if current_chunk.strip():
                chunk_id = self._generate_chunk_id(current_chunk, doc_path, chunk_count)
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    metadata={
                        'source': doc_title,
                        'title': doc_title,
                        'filename': os.path.basename(doc_path),
                        'page': page_num,
                        'chunk_index': chunk_count,
                        'document_type': 'reservoir_engineering',
                        'priority_section': self._detect_priority_section(current_chunk, page_num),
                        'document_path': doc_path
                    },
                    chunk_id=chunk_id,
                    page_number=page_num,
                    document_path=doc_path
                ))
        
        return chunks
    
    def _detect_priority_section(self, content: str, page_num: int) -> str:
        """
        Detecta se o chunk está em uma seção prioritária com análise expandida
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
        
        # Detect methodology sections (very important for technical docs)
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
        
        # Enhanced definition detection (works throughout document)
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
        
        # Detect equations and mathematical content (high priority)
        equation_indicators = ['=', '∂', '∇', '∫', '∑', 'equation', 'equação', 'formula', 'fórmula']
        if any(pattern in content_lower or pattern in content for pattern in equation_indicators):
            return 'equation'
        
        # Detect figures and tables references (important context)
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
        
        # Detect literature review sections
        literature_indicators = [
            'literature review', 'revisão da literatura', 'state of the art', 'estado da arte',
            'previous work', 'trabalhos anteriores', 'related work', 'trabalhos relacionados',
            'background', 'contexto', 'references', 'referências'
        ]
        if any(indicator in content_lower for indicator in literature_indicators):
            return 'literature'
        
        # High priority for early pages (likely contains key information)
        if page_num <= 3:
            return 'early_page'
        
        return 'regular'
    
    def _generate_chunk_id(self, content: str, doc_path: str, chunk_index: int) -> str:
        """
        Gera ID único para o chunk
        """
        import time
        import os
        # Inclui nome do arquivo, índice, timestamp e hash do conteúdo completo para garantir unicidade
        filename = os.path.basename(doc_path)
        timestamp = str(int(time.time() * 1000000))  # microsegundos
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        source = f"{filename}_{chunk_index}_{timestamp}_{content_hash}"
        return hashlib.md5(source.encode()).hexdigest()
    
    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Processa um PDF completo: extração + chunking
        """
        print(f"Processando: {pdf_path}")
        pages = self.extract_text_from_pdf(pdf_path)
        chunks = self.create_chunks(pages)
        print(f"Criados {len(chunks)} chunks do documento")
        return chunks
    
    def process_directory(self, directory_path: str) -> List[DocumentChunk]:
        """
        Processa todos os PDFs de um diretório
        """
        all_chunks = []
        
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
        
        return all_chunks

# Termos específicos de engenharia de reservatórios para melhoria da busca
RESERVOIR_TERMS = [
    'permeabilidade', 'porosidade', 'saturação', 'pressão', 'temperatura',
    'viscosidade', 'densidade', 'compressibilidade', 'fator volume formação',
    'razão gás óleo', 'simulador', 'modelo', 'grid', 'células', 'timestep',
    'well', 'poço', 'injeção', 'produção', 'água', 'óleo', 'gás',
    'reservatório', 'rocha', 'fluido', 'pvt', 'equações', 'darcy',
    'buckley leverett', 'material balance', 'decline curve', 'recovery factor'
]