"""
RAG v3 System - Portuguese responses with (Author, Year) citations
Uses Google Gemini API and enhanced metadata extraction
"""

import os
import sys
import hashlib
import unicodedata
from dataclasses import replace
from typing import AsyncIterator, List, Dict, Any, Optional
from pathlib import Path

# Import from existing modules
sys.path.insert(0, str(Path(__file__).parent))

from processors.pdf_processor import PDFProcessor
from vectorstore.chroma_store import LocalVectorStore
from embeddings.local_embedder import LocalEmbedder
from models.gemini_client import GeminiClient
from citations.metadata_extractor_v3 import MetadataExtractorV3, DocumentMetadata


class RAGv3System:
    """
    RAG v3 System with Portuguese responses and academic citations
    """

    def __init__(self,
                 vectorstore_path: str = "data/vectorstore_v3",
                 embedder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 llm_model: str = None,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50):
        """
        Initialize RAG v3 System

        Args:
            vectorstore_path: Path for v3 vector database
            embedder_model: Multilingual embedding model
            llm_model: Gemini model to use
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        print("🚀 Initializing RAG v3 System...")

        # PDF Processor
        self.pdf_processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Metadata Extractor for author/year
        self.metadata_extractor = MetadataExtractorV3()

        # Embedder
        print(f"📊 Loading embedder: {embedder_model}")
        self.embedder = LocalEmbedder(model_name=embedder_model)

        # Vector Store (v3)
        print(f"💾 Initializing vector store: {vectorstore_path}")
        self.vector_store = LocalVectorStore(
            persist_directory=vectorstore_path,
            collection_name="iabel_v3"
        )

        # LLM (Gemini)
        llm_model = llm_model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        print(f"🤖 Initializing Gemini LLM: {llm_model}")
        self.llm_client = GeminiClient(model_name=llm_model)

        # Document metadata cache (filename -> DocumentMetadata)
        self.document_metadata_cache: Dict[str, DocumentMetadata] = {}

        print("✅ RAG v3 System initialized")

    @staticmethod
    def _drop_repeated_boilerplate(chunks: list, max_len: int = 200) -> list:
        """
        Remove chunks curtos cujo texto se repete em várias páginas do mesmo documento
        (cabeçalho de slide, folha de título). Sem conteúdo próprio, eles casavam com
        perguntas genéricas e ocupavam as vagas de fonte na busca.
        """
        pages_by_text: Dict[str, set] = {}
        for chunk in chunks:
            key = ' '.join(chunk.content.split())
            pages_by_text.setdefault(key, set()).add(chunk.page_number)
        kept = [
            c for c in chunks
            if not (len(c.content) <= max_len and len(pages_by_text[' '.join(c.content.split())]) > 1)
        ]
        if len(kept) != len(chunks):
            print(f"   🧹 Dropped {len(chunks) - len(kept)} repeated header/boilerplate chunks")
        return kept

    def _fit_to_token_window(self, chunks: list) -> list:
        """
        O modelo de embeddings só lê max_seq_length tokens (128 no mpnet multilíngue) e
        descarta o resto em silêncio: a parte final do trecho ficava invisível para a busca.
        O tamanho em caracteres não garante o limite (frases sem pontuação, fórmulas e
        siglas geram muitos tokens), então os trechos longos são divididos por palavras
        medindo com o tokenizador do próprio modelo.
        """
        model = self.embedder.model
        tokenizer = model.tokenizer
        limit = model.max_seq_length - 2  # tokens especiais de início e fim

        def n_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)['input_ids'])

        fitted, split_count = [], 0
        for chunk in chunks:
            if n_tokens(chunk.content) <= limit:
                fitted.append(chunk)
                continue
            split_count += 1
            parts, current = [], []
            for word in chunk.content.split():
                if current and n_tokens(' '.join(current + [word])) > limit:
                    parts.append(' '.join(current))
                    current = []
                current.append(word)
            if current:
                parts.append(' '.join(current))
            for i, part in enumerate(parts):
                fitted.append(replace(
                    chunk,
                    content=part,
                    chunk_id=hashlib.md5(f"{chunk.chunk_id}_part{i}".encode()).hexdigest(),
                    metadata={**chunk.metadata, 'chunk_part': i},
                ))
        if split_count:
            print(f"   ✂️ Split {split_count} chunks longer than {limit} tokens")
        return fitted

    def index_pdf_directory(self, pdf_directory: str, force_reindex: bool = False) -> bool:
        """
        Index PDF directory with enhanced metadata extraction

        Args:
            pdf_directory: Directory containing PDFs
            force_reindex: Whether to force reindexing

        Returns:
            Success status
        """
        try:
            print(f"\n📁 Indexing PDF directory: {pdf_directory}")

            if not os.path.exists(pdf_directory):
                print(f"❌ Directory not found: {pdf_directory}")
                return False

            # Get all PDF files
            pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
            print(f"📄 Found {len(pdf_files)} PDF files")

            if not pdf_files:
                print("⚠️ No PDF files found")
                return False

            # Check if already indexed
            if not force_reindex:
                stats = self.get_system_status()
                if stats.get('vector_store', {}).get('total_documents', 0) > 0:
                    print(f"✅ Vector store already has {stats['vector_store']['total_documents']} documents")
                    print("   Use force_reindex=True to reindex")
                    return True
            else:
                # Sem limpar, cada reindexação acrescentava uma cópia inteira do acervo
                print("🧹 Clearing existing index before reindexing...")
                if not self.vector_store.clear_collection():
                    return False

            catalog_path = Path(pdf_directory).parent / "catalogo_bibliografico.json"
            self.metadata_extractor.catalog = MetadataExtractorV3.load_catalog(str(catalog_path))
            print(f"📚 Bibliographic catalog: {len(self.metadata_extractor.catalog)} entries ({catalog_path})")

            # Process each PDF
            all_chunks = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                print(f"\n📄 Processing: {pdf_file}")

                # Extract metadata
                print("   📝 Extracting metadata...")
                metadata = self.metadata_extractor.extract_metadata(pdf_path)
                self.document_metadata_cache[pdf_file] = metadata

                print(f"   👤 Author(s): {', '.join(metadata.authors)}")
                print(f"   📅 Year: {metadata.year or 'Unknown'}")
                print(f"   📖 Title: {metadata.title}")
                if unicodedata.normalize('NFC', pdf_file) not in self.metadata_extractor.catalog:
                    print("   ⚠️ Not in catalogo_bibliografico.json — citation guessed automatically")

                # Process PDF into chunks
                print("   🔨 Creating chunks...")
                chunks = self._drop_repeated_boilerplate(self.pdf_processor.process_pdf(pdf_path))
                chunks = self._fit_to_token_window(chunks)

                # Enhance chunk metadata with author/year
                for chunk in chunks:
                    chunk.metadata['author'] = ', '.join(metadata.authors)
                    chunk.metadata['year'] = str(metadata.year) if metadata.year is not None else 'n.d.'
                    chunk.metadata['document_title'] = metadata.title
                    chunk.metadata['citation'] = metadata.format_citation()

                all_chunks.extend(chunks)
                print(f"   ✅ Created {len(chunks)} chunks")

            # Generate embeddings
            print(f"\n🔢 Generating embeddings for {len(all_chunks)} chunks...")
            texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedder.embed_batch(texts, batch_size=32)

            # Prepare data for vector store
            documents = [chunk.content for chunk in all_chunks]
            metadatas = [chunk.metadata for chunk in all_chunks]
            ids = [chunk.chunk_id for chunk in all_chunks]

            # Add to vector store
            print("💾 Adding to vector store...")
            success = self.vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            if success:
                print(f"✅ Successfully indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
                return True
            else:
                print("❌ Failed to add documents to vector store")
                return False

        except Exception as e:
            print(f"❌ Error indexing PDFs: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search_documents(self,
                        query: str,
                        top_k: int = 8,
                        similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Search documents with enhanced metadata

        Args:
            query: Search query
            top_k: Number of results
            similarity_threshold: Minimum similarity

        Returns:
            List of search results with metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        return results

    def ask_question(self,
                    question: str,
                    top_k: int = 8,
                    include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer question with Portuguese response and (Author, Year) citations.
        """
        search_results = self.search_documents(question, top_k=top_k)

        if not search_results:
            return {
                'answer': "Não encontrei informações relevantes nos documentos indexados para responder à sua pergunta.",
                'confidence': 0.0,
                'total_sources': 0,
                'sources': []
            }

        _, system_prompt = self._build_context_and_prompt(search_results)

        answer = self.llm_client.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3
        )

        confidences = [r.get('similarity', 0) for r in search_results]
        avg_confidence = sum(confidences[:3]) / min(3, len(confidences)) if confidences else 0

        sources = self._build_sources(search_results) if include_sources else []

        return {
            'answer': answer,
            'confidence': avg_confidence,
            'total_sources': len(search_results),
            'sources': sources,
            'enhanced': True,
            'generation_mode': 'rag_v3'
        }

    def _build_context_and_prompt(self, search_results: List[Dict[str, Any]]) -> tuple[str, str]:
        """Return (context_text, system_prompt) from search results."""
        context_parts = []
        for result in search_results:
            citation = result['metadata'].get('citation', '(Unknown, n.d.)')
            context_parts.append(f"{citation}:\n{result['content']}")
        context_text = "\n\n---\n\n".join(context_parts)

        system_prompt = f"""You are an expert assistant in Reservoir Engineering and Petroleum Engineering.

REFERENCE DOCUMENTS:
{context_text}

REASONING PROCESS:
Before writing your answer, reason through the question step by step:
1. Identify what the question is really asking (core concept, comparison, limitation, mechanism, etc.)
2. Scan the reference documents for directly relevant passages
3. Identify any gaps or partial coverage
4. Synthesize across documents if multiple sources address the same point
5. Only then compose the final answer

INSTRUCTIONS:
- Answer in Brazilian Portuguese with technical precision, even when the reference documents are in English
- Keep established English technical terms and model names as they are (e.g. CRM, INSIM-FT, front tracking)
- Synthesize and summarize information from the reference documents to answer the question fully
- Use inline citations in (Author, Year) format when referencing specific claims, e.g. (Silva et al., 2022)
- You may draw technical inferences from the provided content where they are clearly supported
- If the documents only partially cover the question, provide what is available and note any gaps
- Do not fabricate facts — ground every claim in the retrieved content above
- Use appropriate reservoir engineering terminology"""
        return context_text, system_prompt

    def _build_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for result in search_results:
            metadata = result['metadata']
            sources.append({
                'content': result['content'][:300],
                'source': metadata.get('document_title', metadata.get('source', 'Unknown')),
                'page': metadata.get('page', 'N/A'),
                'similarity': result.get('similarity', 0),
                'author': metadata.get('author', 'Unknown'),
                'year': metadata.get('year', 'N/A'),
                'citation': metadata.get('citation', '(Unknown, n.d.)')
            })
        return sources

    async def ask_question_stream(self,
                                  question: str,
                                  top_k: int = 8,
                                  include_sources: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream answer with Portuguese response and (Author, Year) citations.
        Yields frontend-compatible SSE chunks: search_complete, generation_start, token, complete.
        """
        search_results = self.search_documents(question, top_k=top_k)

        if not search_results:
            yield {
                'type': 'complete',
                'answer': "Não encontrei informações relevantes nos documentos indexados para responder à sua pergunta.",
                'confidence': 0.0,
                'total_sources': 0,
                'sources': [],
                'enhanced': True,
                'generation_mode': 'rag_v3'
            }
            return

        yield {'type': 'search_complete', 'sources_found': len(search_results)}

        _, system_prompt = self._build_context_and_prompt(search_results)

        yield {'type': 'generation_start'}

        full_answer = ""
        async for token in self.llm_client.generate_response_stream(
            prompt=question,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.3
        ):
            full_answer += token
            yield {'type': 'token', 'content': token}

        confidences = [r.get('similarity', 0) for r in search_results]
        avg_confidence = sum(confidences[:3]) / min(3, len(confidences)) if confidences else 0

        sources = self._build_sources(search_results) if include_sources else []

        yield {
            'type': 'complete',
            'answer': full_answer,
            'confidence': avg_confidence,
            'total_sources': len(search_results),
            'sources': sources,
            'enhanced': True,
            'generation_mode': 'rag_v3'
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            # Get vector store stats
            total_docs = len(self.vector_store.collection.get()['ids']) if hasattr(self.vector_store, 'collection') else 0

            return {
                'version': 'v3',
                'vector_store': {
                    'path': self.vectorstore_path,
                    'total_documents': total_docs,
                    'collection': 'iabel_v3'
                },
                'embedder': {
                    'model': self.embedder.model_name,
                    'dimension': self.embedder.get_embedding_dimension()
                },
                'llm': {
                    'provider': 'gemini',
                    'model': self.llm_client.model_name
                },
                'features': {
                    'language': 'Portuguese',
                    'citation_format': '(Author, Year)',
                    'metadata_extraction': True
                }
            }
        except Exception as e:
            return {
                'version': 'v3',
                'status': 'error',
                'error': str(e)
            }
