"""
LoRA Fine-tuning Data Processor for IAbel
Prepares domain-specific reservoir engineering texts for PEFT training
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Defensive imports for PDF extractors
PDF_EXTRACTORS = []
PyMuPDF = None
pdfplumber = None

try:
    import PyMuPDF
    PDF_EXTRACTORS.append("PyMuPDF")
    logger.info("✅ PyMuPDF available for fast PDF extraction")
except ImportError:
    logger.warning("⚠️ PyMuPDF not available. Install with: pip install PyMuPDF==1.23.26")

try:
    import pdfplumber  
    PDF_EXTRACTORS.append("pdfplumber")
    logger.info("✅ pdfplumber available for PDF extraction")
except ImportError:
    logger.warning("⚠️ pdfplumber not available. Install with: pip install pdfplumber==0.10.3")

if not PDF_EXTRACTORS:
    raise ImportError(
        "No PDF extractors available! Please install at least one:\n"
        "  pip install PyMuPDF==1.23.26 pdfplumber==0.10.3"
    )

logger.info(f"Available PDF extractors: {PDF_EXTRACTORS}")

from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer

@dataclass
class DocumentChunk:
    """Enhanced document chunk for fine-tuning"""
    text: str
    source: str
    page: int
    chunk_type: str  # 'abstract', 'definition', 'equation', 'example', 'discussion'
    technical_density: float  # 0-1 score based on technical term frequency
    metadata: Dict

class AcademicDataProcessor:
    """Processes large academic PDFs for LoRA fine-tuning"""
    
    def __init__(self, 
                 chunk_size: int = 1024,  # Larger chunks for fine-tuning
                 chunk_overlap: int = 128,
                 min_technical_density: float = 0.3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_technical_density = min_technical_density
        
        # Enhanced technical vocabulary for reservoir engineering
        self.technical_terms = {
            # Simulation & Modeling
            'insim', 'eclipse', 'cmg', 'intersect', 'tnavigator', 'petrel',
            'streamlines', 'simulation', 'modeling', 'numerical', 'finite',
            'discretization', 'grid', 'timestep', 'convergence', 'iteration',
            
            # Reservoir Properties
            'permeability', 'porosity', 'saturation', 'compressibility',
            'viscosity', 'density', 'pressure', 'temperature', 'pvt',
            'relative', 'capillary', 'wettability', 'heterogeneity',
            
            # Production & Injection
            'production', 'injection', 'waterflooding', 'breakthrough',
            'sweep', 'efficiency', 'recovery', 'displacement', 'mobility',
            'fractional', 'flow', 'cut', 'gor', 'wor', 'bhp', 'thp',
            
            # Well & Completion
            'wellbore', 'completion', 'perforation', 'skin', 'productivity',
            'injectivity', 'drawdown', 'buildup', 'interference', 'connectivity',
            
            # Enhanced Recovery
            'eor', 'ior', 'wag', 'surfactant', 'polymer', 'thermal',
            'steam', 'sagd', 'css', 'chemical', 'miscible', 'immiscible',
            
            # Reservoir Types
            'carbonate', 'sandstone', 'shale', 'tight', 'unconventional',
            'naturally', 'fractured', 'dual', 'porosity', 'vuggy',
            
            # Portuguese Terms
            'reservatório', 'simulação', 'permeabilidade', 'porosidade',
            'saturação', 'produção', 'injeção', 'recuperação', 'varrido'
        }
        
        # Question-answer patterns for instruction tuning
        self.qa_patterns = [
            "O que é {}?",
            "Como funciona {}?", 
            "Qual a importância de {}?",
            "Explique o conceito de {}",
            "Descreva o processo de {}",
            "What is {}?",
            "How does {} work?",
            "Explain the concept of {}",
            "Describe the process of {}"
        ]
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Enhanced PDF extraction with adaptive extractor selection"""
        chunks = []
        
        # Try PyMuPDF first if available (faster)
        if PyMuPDF:
            try:
                logger.info(f"Using PyMuPDF for {Path(pdf_path).name}")
                chunks.extend(self._extract_with_pymupdf(pdf_path))
                return chunks
            except Exception as e:
                logger.warning(f"PyMuPDF failed for {pdf_path}: {e}")
        
        # Fallback to pdfplumber
        if pdfplumber:
            try:
                logger.info(f"Using pdfplumber for {Path(pdf_path).name}")
                chunks.extend(self._extract_with_pdfplumber(pdf_path))
                return chunks
            except Exception as e:
                logger.error(f"pdfplumber failed for {pdf_path}: {e}")
        
        # No extractors worked
        logger.error(f"All PDF extractors failed for {pdf_path}")
        return []
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Extract using PyMuPDF with academic structure detection"""
        if not PyMuPDF:
            raise ImportError("PyMuPDF not available")
        
        chunks = []
        doc = PyMuPDF.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                continue
                
            # Detect academic sections
            chunk_type = self._detect_section_type(text)
            technical_density = self._calculate_technical_density(text)
            
            # Split into chunks
            page_chunks = self._smart_chunking(
                text, 
                source=Path(pdf_path).name,
                page=page_num + 1,
                chunk_type=chunk_type,
                technical_density=technical_density
            )
            
            chunks.extend(page_chunks)
        
        doc.close()
        return chunks
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[DocumentChunk]:
        """Fallback extraction with pdfplumber"""
        if not pdfplumber:
            raise ImportError("pdfplumber not available")
        
        chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                
                if not text:
                    continue
                
                chunk_type = self._detect_section_type(text)
                technical_density = self._calculate_technical_density(text)
                
                page_chunks = self._smart_chunking(
                    text,
                    source=Path(pdf_path).name,
                    page=page_num + 1,
                    chunk_type=chunk_type,
                    technical_density=technical_density
                )
                
                chunks.extend(page_chunks)
        
        return chunks
    
    def _detect_section_type(self, text: str) -> str:
        """Detect academic section type for prioritization"""
        text_lower = text.lower()
        
        # Abstract detection
        if any(keyword in text_lower for keyword in 
               ['abstract', 'resumo', 'summary', 'síntese']):
            return 'abstract'
        
        # Definition detection
        if any(pattern in text_lower for pattern in 
               ['define', 'definition', 'definição', 'conceito', 'é definido']):
            return 'definition'
        
        # Equation detection
        if any(char in text for char in ['=', '∂', '∆', '∫', '∑']) or \
           any(keyword in text_lower for keyword in ['equation', 'equação', 'formula']):
            return 'equation'
        
        # Example detection
        if any(keyword in text_lower for keyword in 
               ['example', 'exemplo', 'case study', 'estudo de caso']):
            return 'example'
        
        # Discussion/Analysis
        if any(keyword in text_lower for keyword in 
               ['discussion', 'discussão', 'analysis', 'análise', 'results', 'resultados']):
            return 'discussion'
        
        return 'general'
    
    def _calculate_technical_density(self, text: str) -> float:
        """Calculate density of technical terms (0-1 scale)"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0
        
        technical_count = sum(1 for word in words if word in self.technical_terms)
        return min(technical_count / len(words) * 10, 1.0)  # Scale up and cap at 1.0
    
    def _smart_chunking(self, text: str, source: str, page: int, 
                       chunk_type: str, technical_density: float) -> List[DocumentChunk]:
        """Intelligent chunking with overlap for fine-tuning"""
        if technical_density < self.min_technical_density:
            return []  # Skip low-density chunks
        
        chunks = []
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(DocumentChunk(
                        text=current_chunk.strip(),
                        source=source,
                        page=page,
                        chunk_type=chunk_type,
                        technical_density=technical_density,
                        metadata={
                            'length': len(current_chunk),
                            'sentence_count': len(re.split(r'[.!?]+', current_chunk))
                        }
                    ))
                
                # Start new chunk with overlap
                if len(sentence) < self.chunk_size:
                    # Include overlap from previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if current_chunk else ""
                    current_chunk = overlap_text + " " + sentence
                else:
                    # Sentence too long, split it
                    current_chunk = sentence[:self.chunk_size]
            else:
                current_chunk += " " + sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                text=current_chunk.strip(),
                source=source,
                page=page,
                chunk_type=chunk_type,
                technical_density=technical_density,
                metadata={
                    'length': len(current_chunk),
                    'sentence_count': len(re.split(r'[.!?]+', current_chunk))
                }
            ))
        
        return chunks
    
    def create_instruction_dataset(self, chunks: List[DocumentChunk]) -> Dataset:
        """Create instruction-following dataset for LoRA fine-tuning"""
        instructions = []
        
        for chunk in chunks:
            # Skip low-quality chunks
            if len(chunk.text) < 100 or chunk.technical_density < self.min_technical_density:
                continue
            
            # Extract key concepts from chunk
            concepts = self._extract_key_concepts(chunk.text)
            
            for concept in concepts[:2]:  # Limit to top 2 concepts per chunk
                # Generate multiple question types
                for pattern in self.qa_patterns[:4]:  # Use top 4 patterns
                    try:
                        question = pattern.format(concept)
                        
                        # Create instruction-following format
                        instruction = {
                            "instruction": question,
                            "input": f"Contexto: {chunk.text[:500]}...",  # Truncate context
                            "output": self._generate_answer(concept, chunk.text),
                            "source": chunk.source,
                            "chunk_type": chunk.chunk_type,
                            "technical_density": chunk.technical_density
                        }
                        
                        instructions.append(instruction)
                    except:
                        continue  # Skip malformed instructions
        
        # Convert to Dataset
        df = pd.DataFrame(instructions)
        return Dataset.from_pandas(df)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key technical concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        # Find technical terms in context
        for term in self.technical_terms:
            if term in text_lower:
                # Get surrounding context
                words = text_lower.split()
                for i, word in enumerate(words):
                    if term in word:
                        # Get 2-3 word phrases around the term
                        start = max(0, i-1)
                        end = min(len(words), i+3)
                        phrase = " ".join(words[start:end])
                        
                        # Clean and add if meaningful
                        phrase = re.sub(r'[^\w\s]', '', phrase).strip()
                        if len(phrase.split()) >= 2 and phrase not in concepts:
                            concepts.append(phrase)
        
        return concepts[:5]  # Return top 5 concepts
    
    def _generate_answer(self, concept: str, context: str) -> str:
        """Generate contextual answer based on the text"""
        # Find sentences containing the concept
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Return the most relevant sentence(s)
            return ". ".join(relevant_sentences[:2])
        else:
            # Fallback: return first few sentences
            return ". ".join(sentences[:2])
    
    def process_pdf_directory(self, pdf_dir: str, output_path: str = None) -> Dataset:
        """Process all PDFs in directory and create training dataset"""
        pdf_dir = Path(pdf_dir)
        all_chunks = []
        
        logger.info(f"Processing PDFs in {pdf_dir}")
        
        for pdf_file in pdf_dir.glob("*.pdf"):
            logger.info(f"Processing {pdf_file.name}...")
            
            try:
                chunks = self.extract_text_from_pdf(str(pdf_file))
                all_chunks.extend(chunks)
                logger.info(f"Extracted {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
        
        logger.info(f"Total chunks: {len(all_chunks)}")
        
        # Filter high-quality chunks
        quality_chunks = [c for c in all_chunks if c.technical_density >= self.min_technical_density]
        logger.info(f"High-quality chunks: {len(quality_chunks)}")
        
        # Create instruction dataset
        dataset = self.create_instruction_dataset(quality_chunks)
        
        # Save if output path provided
        if output_path:
            dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
        
        return dataset

def main():
    """Example usage"""
    processor = AcademicDataProcessor(
        chunk_size=1024,
        chunk_overlap=128,
        min_technical_density=0.3
    )
    
    # Process PDFs
    pdf_directory = "/home/lacucaratila/Projetos/IAbel/backend/data/pdfs"
    output_directory = "/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/datasets"
    
    dataset = processor.process_pdf_directory(pdf_directory, output_directory)
    
    print(f"Created dataset with {len(dataset)} instruction examples")
    print("\nSample instruction:")
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input'][:200]}...")
        print(f"Output: {sample['output'][:200]}...")

if __name__ == "__main__":
    main()