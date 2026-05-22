"""
Enhanced RAG System - Integrates all improvements for superior performance
Combines semantic chunking, RAG fusion, re-ranking, and context enhancement
"""

import os
import sys

# Disable ChromaDB telemetry to avoid posthog errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime
from pathlib import Path
import asyncio
import uuid

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import enhanced components
from rag_system import LocalRAGSystem
from chunking.semantic_chunker import get_semantic_chunker, SemanticChunk
from fusion.rag_fusion import get_rag_fusion
from reranking.cross_encoder_reranker import get_reranker
from context_enhancer import ContextEnhancer
from query_optimizer import QueryOptimizer

# Import new advanced features
from compression.context_compressor import get_context_compressor, CompressedContext
from citations.citation_tracker import get_citation_tracker, CitationMap
from citations.academic_formatter import AcademicCitationFormatter
from learning.feedback_system import get_feedback_system, FeedbackLearningSystem
from memory.conversation_memory import get_conversation_memory, ConversationMemory

# Import base components
from processors.pdf_processor import PDFProcessor
from processors.langchain_document_processor import LangChainDocumentProcessor
from embeddings.local_embedder import LocalEmbedder
from vectorstore.chroma_store import LocalVectorStore
from models.ollama_client import OllamaClient


class EnhancedRAGSystem(LocalRAGSystem):
    """
    Enhanced RAG system with all performance improvements integrated
    """
    
    def __init__(self,
                 vectorstore_path: str = "./vectorstore_enhanced",
                 embedder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 llm_model: str = "llama3.2:3b",
                 llm_provider: str = "ollama",
                 chunk_size: int = 700,
                 chunk_overlap: int = 120,
                 enable_semantic_chunking: bool = True,
                 enable_rag_fusion: bool = True,
                 enable_reranking: bool = False,
                 enable_context_compression: bool = True,
                 enable_citation_tracking: bool = True,
                 enable_feedback_learning: bool = True,
                 enable_conversation_memory: bool = True,
                 use_langchain_loader: bool = True):
        """
        Initialize enhanced RAG system with all improvements

        Args:
            vectorstore_path: Path for vector database storage
            embedder_model: Multilingual embedding model
            llm_model: LLM model to use
            llm_provider: LLM provider ("ollama", "gemini")
            chunk_size: Target chunk size for semantic chunking
            chunk_overlap: Overlap between chunks
            enable_semantic_chunking: Use semantic chunking instead of simple chunking
            enable_rag_fusion: Use RAG fusion for better retrieval
            enable_reranking: Use cross-encoder re-ranking
            enable_context_compression: Use intelligent context compression
            enable_citation_tracking: Enable precise citation tracking
            enable_feedback_learning: Enable learning from user feedback
            enable_conversation_memory: Enable conversation memory for context
            use_langchain_loader: Use LangChain document loaders for better extraction
        """
        print("🚀 Initializing Enhanced RAG System for IAbel...")
        
        # Store configuration
        self.enable_semantic_chunking = enable_semantic_chunking
        self.enable_rag_fusion = enable_rag_fusion
        self.enable_reranking = enable_reranking
        self.enable_context_compression = enable_context_compression
        self.enable_citation_tracking = enable_citation_tracking
        self.enable_feedback_learning = enable_feedback_learning
        self.enable_conversation_memory = enable_conversation_memory
        self.use_langchain_loader = use_langchain_loader
        
        # Initialize base components
        if use_langchain_loader:
            print("   📚 Using LangChain document loaders")
            self.document_processor = LangChainDocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.pdf_processor = None
        else:
            print("   📄 Using legacy PDF processor")
            self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.document_processor = None
        
        if enable_semantic_chunking and not use_langchain_loader:
            print("   🧠 Using semantic chunking")
            self.semantic_chunker = get_semantic_chunker()
        else:
            self.semantic_chunker = None
        
        self.embedder = LocalEmbedder(model_name=embedder_model)
        self.vector_store = LocalVectorStore(persist_directory=vectorstore_path)

        # Initialize LLM client based on provider
        if llm_provider == "gemini":
            from models.gemini_client import GeminiClient
            self.llm_client = GeminiClient(model_name=llm_model)
            print(f"   🤖 Using Google Gemini: {llm_model}")
        else:
            self.llm_client = OllamaClient(model_name=llm_model)
            print(f"   🤖 Using Ollama: {llm_model}")

        self.query_optimizer = QueryOptimizer()
        self.context_enhancer = ContextEnhancer()
        
        # Initialize academic citation formatter
        self.academic_formatter = AcademicCitationFormatter()
        
        # Initialize enhanced components
        if enable_rag_fusion:
            print("   🔀 Enabling RAG Fusion")
            self.rag_fusion = get_rag_fusion(self.embedder.model)
        else:
            self.rag_fusion = None
        
        if enable_reranking:
            print("   🎯 Enabling Cross-Encoder Re-ranking")
            self.reranker = get_reranker()
        else:
            self.reranker = None
        
        # Initialize new advanced features
        if enable_context_compression:
            print("   🗜️ Enabling Context Compression")
            self.context_compressor = get_context_compressor(self.embedder.model)
        else:
            self.context_compressor = None
        
        if enable_citation_tracking:
            print("   📚 Enabling Citation Tracking")
            self.citation_tracker = get_citation_tracker(self.embedder.model)
        else:
            self.citation_tracker = None
        
        if enable_feedback_learning:
            print("   🧠 Enabling Feedback Learning")
            self.feedback_system = get_feedback_system()
        else:
            self.feedback_system = None
        
        if enable_conversation_memory:
            print("   💭 Enabling Conversation Memory")
            self.conversation_memory = get_conversation_memory()
        else:
            self.conversation_memory = None
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        print("✅ Enhanced RAG System initialized!")
        self._print_capabilities()
    
    def _print_capabilities(self):
        """Print enabled capabilities"""
        capabilities = []
        if self.use_langchain_loader:
            capabilities.append("LangChain Document Loaders")
        if self.enable_semantic_chunking and not self.use_langchain_loader:
            capabilities.append("Semantic Chunking")
        if self.enable_rag_fusion:
            capabilities.append("RAG Fusion")
        if self.enable_reranking:
            capabilities.append("Cross-Encoder Re-ranking")
        if self.enable_context_compression:
            capabilities.append("Context Compression")
        if self.enable_citation_tracking:
            capabilities.append("Citation Tracking")
        if self.enable_feedback_learning:
            capabilities.append("Feedback Learning")
        
        print(f"   💪 Enhanced capabilities: {', '.join(capabilities)}")
        print(f"   📊 Embedder: {self.embedder.model_name}")
        print(f"   🤖 LLM: {self.llm_client.model_name}")
        print(f"   💾 Vector Store: {self.vector_store.persist_directory}")
        
        if self.use_langchain_loader:
            print(f"   📚 Document Types: PDF, TXT, MD, CSV, JSON, HTML, CSS, JS, PY")
    
    def index_pdf_directory(self, pdf_directory: str, force_reindex: bool = False) -> bool:
        """
        Enhanced PDF indexing with semantic chunking
        
        Args:
            pdf_directory: Directory containing PDFs
            force_reindex: Whether to force reindexing
        """
        if not os.path.exists(pdf_directory):
            print(f"❌ Directory not found: {pdf_directory}")
            return False
        
        # Check existing documents
        stats = self.vector_store.get_collection_stats()
        if stats['total_documents'] > 0 and not force_reindex:
            print(f"ℹ️  Already have {stats['total_documents']} documents indexed")
            print("   Use force_reindex=True to reindex")
            return True
        
        if force_reindex:
            print("🧹 Clearing existing index...")
            self.vector_store.clear_collection()
        
        print(f"📁 Processing documents in: {pdf_directory}")
        
        # Process documents with enhanced processing
        if self.use_langchain_loader:
            return self._process_documents_with_langchain(pdf_directory, force_reindex)
        elif self.enable_semantic_chunking:
            all_chunks = self._process_pdfs_semantic(pdf_directory)
        else:
            all_chunks = self.pdf_processor.process_directory(pdf_directory)
        
        if not all_chunks:
            print("❌ No chunks extracted from PDFs")
            return False
        
        print(f"📄 Total chunks created: {len(all_chunks)}")
        
        # Show chunking statistics if using semantic chunking
        if self.enable_semantic_chunking and hasattr(all_chunks[0], 'semantic_score'):
            stats = self.semantic_chunker.get_chunking_stats(all_chunks)
            print(f"   📊 Avg chunk size: {stats.get('avg_chunk_size', 0):.0f} chars")
            print(f"   🧠 High quality chunks: {stats.get('high_quality_chunks', 0)}")
            print(f"   🔬 Technical chunks: {stats.get('technical_chunks', 0)}")
        
        # Generate embeddings in batches
        print("🔄 Generating embeddings...")
        texts = [
            getattr(chunk, 'content', None) or getattr(chunk, 'page_content', '')
            for chunk in all_chunks
        ]
        embeddings = self.embedder.embed_batch(texts, batch_size=32)
        
        # Prepare data for vector store
        def clean_metadata(metadata):
            """Clean metadata to only include primitive types for ChromaDB"""
            cleaned = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    cleaned[key] = value
                elif isinstance(value, dict):
                    # Skip nested dicts or convert to string if needed
                    cleaned[f"{key}_str"] = str(value)[:100]  # Truncate if too long
                elif isinstance(value, (list, tuple)):
                    # Convert list to string representation
                    cleaned[f"{key}_str"] = str(value)[:100]
                else:
                    # Convert other types to string
                    cleaned[f"{key}_str"] = str(value)[:100]
            return cleaned

        documents = texts
        metadatas = [clean_metadata(getattr(chunk, 'metadata', {})) for chunk in all_chunks]
        ids = [getattr(chunk, 'chunk_id', str(i)) for i, chunk in enumerate(all_chunks)]
        
        # Add to vector store
        print("💾 Adding to enhanced vector store...")
        success = self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        if success:
            final_stats = self.vector_store.get_collection_stats()
            print("✅ Enhanced indexing completed!")
            print(f"   📚 Total documents: {final_stats['total_documents']}")
            return True
        else:
            print("❌ Error in enhanced indexing")
            return False
    
    def _process_pdfs_semantic(self, pdf_directory: str) -> List[SemanticChunk]:
        """Process PDFs using semantic chunking"""
        all_chunks = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"   🔄 Processing: {pdf_file}")
            
            try:
                # Extract text using optimized processor
                pages = self.pdf_processor.extract_text_from_pdf(pdf_path)
                
                if not pages:
                    print(f"   ⚠️ No pages extracted from {pdf_file}")
                    continue
                
                # For large PDFs, process in smaller chunks to avoid memory issues
                if len(pages) > 100:  # Large PDF
                    print(f"   📚 Large PDF detected ({len(pages)} pages), processing in batches...")
                    chunks = self._process_large_pdf_semantic(pdf_file, pdf_path, pages)
                else:
                    # Process normally for smaller PDFs
                    text = '\n\n'.join([page['content'] for page in pages])
                    if not text.strip():
                        print(f"   ⚠️ No text content in {pdf_file}")
                        continue
                    
                    metadata = {
                        'source': pages[0].get('document_title', pdf_file),
                        'filename': pdf_file,
                        'file_path': pdf_path,
                        'document_path': pdf_path,
                        'processing_date': datetime.now().isoformat()
                    }
                    
                    chunks = self.semantic_chunker.chunk_document(text, metadata)
                all_chunks.extend(chunks)
                
                print(f"   ✅ Created {len(chunks)} semantic chunks from {pdf_file}")
                
            except Exception as e:
                print(f"   ❌ Error processing {pdf_file}: {e}")
        
        return all_chunks
    
    def _process_large_pdf_semantic(self, pdf_file: str, pdf_path: str, pages: List[Dict]) -> List[SemanticChunk]:
        """Process large PDFs in batches to manage memory"""
        all_chunks = []
        batch_size = 50  # Process 50 pages at a time
        
        for batch_start in range(0, len(pages), batch_size):
            batch_end = min(batch_start + batch_size, len(pages))
            batch_pages = pages[batch_start:batch_end]
            
            print(f"     📦 Processing batch {batch_start + 1}-{batch_end} of {len(pages)} pages")
            
            # Combine text from batch
            batch_text = '\n\n'.join([page['content'] for page in batch_pages])
            
            if batch_text.strip():
                metadata = {
                    'source': batch_pages[0].get('document_title', pdf_file),
                    'filename': pdf_file,
                    'file_path': pdf_path,
                    'document_path': pdf_path,
                    'processing_date': datetime.now().isoformat(),
                    'batch_pages': f"{batch_start + 1}-{batch_end}"
                }
                
                batch_chunks = self.semantic_chunker.chunk_document(batch_text, metadata)
                all_chunks.extend(batch_chunks)
                
                print(f"     ✅ Batch created {len(batch_chunks)} chunks")
            
            # Force garbage collection after each batch
            import gc
            gc.collect()
        
        return all_chunks
    
    def _process_documents_with_langchain(self, directory_path: str, force_reindex: bool = False) -> bool:
        """
        Process documents using LangChain loaders with better extraction
        """
        print("🚀 Processing documents with LangChain loaders...")
        
        try:
            # Process all supported documents in directory
            all_chunks = self.document_processor.process_directory(
                directory_path,
                supported_extensions=['.pdf', '.txt', '.md', '.csv', '.json', '.py', '.js', '.html', '.css']
            )
            
            if not all_chunks:
                print("⚠️ No chunks created from documents")
                return False
            
            print(f"📊 Total chunks to embed: {len(all_chunks)}")
            
            # Convert LangChain chunks to format expected by vector store
            texts = []
            metadatas = []
            
            for chunk in all_chunks:
                texts.append(chunk.content)
                metadatas.append(clean_metadata(chunk.metadata))
            
            # Generate embeddings and store
            print("🔢 Generating embeddings...")
            embeddings = self.embedder.embed_batch(texts, batch_size=32)
            
            print("💾 Storing in vector database...")
            # Generate unique IDs
            ids = [f"langchain_{i}_{int(datetime.now().timestamp())}" for i in range(len(texts))]
            
            self.vector_store.add_documents(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            # Get final stats
            stats = self.vector_store.get_collection_stats()
            print(f"✅ LangChain processing complete!")
            print(f"   📊 Total documents in store: {stats['total_documents']}")
            print(f"   💾 Collection: {stats['collection_name']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in LangChain processing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def enhanced_search_documents(self, 
                                       query: str, 
                                       top_k: int = 8,
                                       similarity_threshold: float = 0.5,
                                       use_fusion: bool = None,
                                       use_reranking: bool = None) -> List[Dict[str, Any]]:
        """
        Enhanced document search with all improvements
        
        Args:
            query: Search query
            top_k: Number of documents to return
            similarity_threshold: Minimum similarity threshold
            use_fusion: Override fusion setting
            use_reranking: Override reranking setting
        """
        print(f"🔍 Enhanced search: '{query}'")
        
        # Determine which enhancements to use
        use_fusion = use_fusion if use_fusion is not None else self.enable_rag_fusion
        use_reranking = use_reranking if use_reranking is not None else self.enable_reranking
        
        if use_fusion and self.rag_fusion:
            # Use RAG Fusion for enhanced retrieval
            print("   🔀 Using RAG Fusion...")
            
            # Create search function for RAG Fusion
            def search_function(q, top_k=top_k):
                return self.search_documents(q, top_k, similarity_threshold)
            
            fusion_result = await self.rag_fusion.enhanced_search(
                original_query=query,
                search_function=search_function,
                max_variations=6,
                fusion_method="reciprocal_rank",
                top_k=top_k * 2  # Get more results for re-ranking
            )
            
            search_results = fusion_result['results']
            
        else:
            # Use standard search
            search_results = self.search_documents(query, top_k * 2, similarity_threshold)
        
        # Apply re-ranking if enabled
        if use_reranking and self.reranker and search_results:
            print("   🎯 Applying cross-encoder re-ranking...")
            search_results = self.reranker.enhanced_rerank(
                query=query,
                documents=search_results,
                top_k=top_k,
                use_technical_boost=True
            )
        else:
            # Limit results if no re-ranking
            search_results = search_results[:top_k]
        
        print(f"✅ Enhanced search complete: {len(search_results)} results")
        return search_results
    
    async def enhanced_ask_question(self, 
                                   question: str,
                                   top_k: int = 4,
                                   include_sources: bool = True,
                                   use_all_enhancements: bool = True,
                                   conversation_id: str = None) -> Dict[str, Any]:
        """
        Enhanced question answering with all improvements
        
        Args:
            question: User question
            top_k: Number of context documents
            include_sources: Whether to include sources
            use_all_enhancements: Whether to use all enhancements
            conversation_id: ID for conversation memory (optional)
        """
        print(f"❓ Enhanced Q&A: {question}")
        
        try:
            # Generate unique response ID for tracking
            response_id = str(uuid.uuid4())
            
            # Enhance question with conversation memory if enabled
            original_question = question
            context_summary = ""
            
            if self.conversation_memory and use_all_enhancements and conversation_id:
                enhanced_question, context_summary = self.conversation_memory.enhance_question_with_context(
                    conversation_id, question
                )
                if enhanced_question != question:
                    print(f"💭 Enhanced with conversation context")
                    question = enhanced_question
            
            # Get feedback-based adjustments if learning is enabled
            feedback_adjustments = {}
            if self.feedback_system and use_all_enhancements:
                feedback_adjustments = self.feedback_system.get_adjustment_recommendations(
                    question, []
                )
                if feedback_adjustments.get('confidence', 0) > 0.5:
                    print(f"   🧠 Applying learned adjustments (confidence: {feedback_adjustments['confidence']:.2f})")
            
            # Enhanced document search
            search_results = await self.enhanced_search_documents(
                query=question,
                top_k=top_k,
                use_fusion=use_all_enhancements,
                use_reranking=use_all_enhancements
            )
            
            if not search_results:
                return {
                    'answer': "Não encontrei informações relevantes nos documentos indexados para responder sua pergunta.",
                    'sources': [],
                    'confidence': 0.0,
                    'enhanced': use_all_enhancements
                }
            
            # Filter high-quality results
            high_quality_results = [r for r in search_results if r.get('similarity', 0) >= 0.4]
            if not high_quality_results:
                high_quality_results = search_results[:3]
            
            # Build enhanced context
            context_docs = [result['content'] for result in high_quality_results]
            enhanced_context = self.context_enhancer.build_enhanced_context(question, context_docs)
            
            # Apply context compression if enabled
            compressed_context = None
            if self.context_compressor and use_all_enhancements:
                print("   🗜️ Compressing context...")
                # Transform search results to expected format for compression
                context_chunks = []
                for result in high_quality_results:
                    chunk = {
                        'content': result.get('content', ''),
                        'metadata': {
                            'source': result.get('metadata', {}).get('source', 'unknown'),
                            'page': result.get('metadata', {}).get('page', 'N/A')
                        }
                    }
                    context_chunks.append(chunk)
                
                compressed_context = self.context_compressor.compress_context(
                    context_chunks, question, preserve_citations=True
                )
                print(f"      Compression: {compressed_context.compression_ratio:.1%} "
                      f"(relevance: {compressed_context.relevance_score:.2f})")
                
                # Use compressed context if it's high quality
                if compressed_context.relevance_score > 0.7:
                    enhanced_context = compressed_context.compressed_text
                else:
                    print("      Using original context due to low compression quality")
            
            # Detect definition gaps
            definition_gaps = self.context_enhancer.detect_definition_gaps(question, high_quality_results)
            if definition_gaps:
                print(f"   ⚠️ Definition gaps: {'; '.join(definition_gaps)}")
            
            # Generate answer with enhanced context and academic citations
            print("   🤖 Generating enhanced answer...")
            
            # Prepare citation mapping for the LLM
            citation_mapping = {}
            if include_sources:
                self.academic_formatter.clear()
                for i, result in enumerate(high_quality_results):
                    source_data = {
                        'filename': result.get('metadata', {}).get('filename', result.get('metadata', {}).get('source', 'Unknown')),
                        'source': result.get('metadata', {}).get('source', 'Unknown'),
                        'page': result.get('metadata', {}).get('page', 'N/A'),
                        'preview': result['content'][:300] if len(result['content']) > 300 else result['content'],
                        'content': result['content'],
                        'file_path': result.get('metadata', {}).get('document_path', '')
                    }
                    
                    source_id = f"{source_data['filename']}_{source_data['page']}"
                    citation_number = self.academic_formatter.add_source(source_id, source_data)
                    citation_mapping[source_id] = citation_number
            
            # Build system prompt with citation instructions
            citation_instructions = ""
            if citation_mapping:
                citation_list = [f"Documento {i+1} → [fonte {citation_mapping[source_id]}]" 
                               for i, source_id in enumerate(citation_mapping.keys())]
                citation_instructions = f"""
INSTRUÇÕES DE CITAÇÃO:
- Use citações numeradas [1], [2], [3] etc. ao referenciar informações específicas
- Mapeamento das fontes: {'; '.join(citation_list)}
- Cite sempre que usar informações específicas de um documento"""
            
            system_prompt = f"""Você é um assistente especialista em Engenharia de Reservatórios.

CONTEXTO DOS DOCUMENTOS:
{enhanced_context}

{citation_instructions}

FORMATO DE RESPOSTA:
- Responda em português brasileiro de forma natural e direta, sem se apresentar
- Seja preciso e técnico quando apropriado
- Use citações numeradas [1], [2], [3] para referenciar informações específicas dos documentos
- NÃO inclua uma seção de referências no final
- Se não souber, diga que não encontrou informações suficientes nos documentos"""
            
            answer = self.llm_client.generate_response(
                prompt=question,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Track citations if enabled
            citation_map = None
            if self.citation_tracker and use_all_enhancements:
                print("   📚 Tracking citations...")
                citation_map = self.citation_tracker.track_citations(
                    response_text=answer,
                    source_contexts=high_quality_results,
                    response_id=response_id
                )
                print(f"      Citation coverage: {citation_map.citation_coverage:.1%} "
                      f"(sources: {citation_map.source_diversity})")
                
                # Format answer with citations
                formatted_answer = self.citation_tracker.format_citations_for_display(
                    citation_map, format_type="simple"
                )
                answer = formatted_answer
            
            # Calculate enhanced confidence
            confidence = self._calculate_enhanced_confidence(high_quality_results, use_all_enhancements)
            
            # Prepare sources with academic citations
            sources = []
            academic_references = ""
            
            if include_sources:
                # Clear previous citations for new response
                self.academic_formatter.clear()
                
                for result in high_quality_results:
                    # Create source data for academic formatter
                    source_data = {
                        'filename': result.get('metadata', {}).get('filename', result.get('metadata', {}).get('source', 'Unknown')),
                        'source': result.get('metadata', {}).get('source', 'Unknown'),
                        'page': result.get('metadata', {}).get('page', 'N/A'),
                        'preview': result['content'][:300] if len(result['content']) > 300 else result['content'],
                        'content': result['content'],
                        'file_path': result.get('metadata', {}).get('document_path', '')
                    }
                    
                    # Add source to academic formatter and get citation number
                    source_id = f"{source_data['filename']}_{source_data['page']}"
                    citation_number = self.academic_formatter.add_source(source_id, source_data)
                    
                    source_info = {
                        'source': source_data['source'],
                        'page': source_data['page'],
                        'similarity': round(result.get('similarity', 0), 3),
                        'preview': source_data['preview'],
                        'citation_number': citation_number
                    }
                    
                    # Add enhancement information
                    if 'fusion_score' in result:
                        source_info['fusion_score'] = round(result['fusion_score'], 3)
                    if 'cross_encoder_score' in result:
                        source_info['rerank_score'] = round(result['cross_encoder_score'], 3)
                    if 'section_type' in result.get('metadata', {}):
                        source_info['section_type'] = result['metadata']['section_type']
                    
                    sources.append(source_info)
                
                # Generate academic reference list
                academic_references = self.academic_formatter.format_reference_list()
                
                # Append academic references to the answer
                if academic_references:
                    answer += "\n\n" + academic_references
            
            # Prepare response with all enhancement information
            response = {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'total_sources': len(high_quality_results),
                'enhanced': use_all_enhancements,
                'response_id': response_id,
                'enhancements_used': {
                    'semantic_chunking': self.enable_semantic_chunking,
                    'rag_fusion': use_all_enhancements and self.enable_rag_fusion,
                    'reranking': use_all_enhancements and self.enable_reranking,
                    'context_enhancement': True,
                    'context_compression': use_all_enhancements and self.enable_context_compression,
                    'citation_tracking': use_all_enhancements and self.enable_citation_tracking,
                    'feedback_learning': use_all_enhancements and self.enable_feedback_learning
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Add compression statistics if available
            if compressed_context:
                response['compression_stats'] = {
                    'compression_ratio': compressed_context.compression_ratio,
                    'relevance_score': compressed_context.relevance_score,
                    'bytes_saved': len(compressed_context.original_text) - len(compressed_context.compressed_text)
                }
            
            # Add citation statistics if available
            if citation_map:
                response['citation_stats'] = {
                    'coverage': citation_map.citation_coverage,
                    'source_diversity': citation_map.source_diversity,
                    'cited_segments': len(citation_map.cited_segments),
                    'uncited_segments': len(citation_map.uncited_segments)
                }
            
            # Add feedback adjustments if applied
            if feedback_adjustments and feedback_adjustments.get('confidence', 0) > 0.5:
                response['feedback_adjustments'] = {
                    'applied': True,
                    'confidence': feedback_adjustments['confidence'],
                    'adjustments_count': len(feedback_adjustments.get('response_adjustments', {}))
                }
            
            # Store interaction in conversation memory if enabled
            if self.conversation_memory and conversation_id:
                try:
                    self.conversation_memory.add_interaction(
                        conversation_id=conversation_id,
                        user_question=original_question,
                        assistant_response=response.get('answer', ''),
                        response_metadata={
                            'sources': response.get('sources', []),
                            'confidence': response.get('confidence', 0),
                            'context_summary': context_summary,
                            'response_id': response_id
                        }
                    )
                except Exception as mem_error:
                    print(f"⚠️ Warning: Failed to store conversation memory: {mem_error}")
            
            return response
            
        except Exception as e:
            print(f"❌ Error in enhanced Q&A: {e}")
            return {
                'answer': f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}",
                'confidence': 0.0,
                'enhanced': use_all_enhancements,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def enhanced_ask_question_stream(self, 
                                         question: str,
                                         top_k: int = 4,
                                         include_sources: bool = True,
                                         use_all_enhancements: bool = True,
                                         conversation_id: str = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Enhanced question answering with streaming response
        
        Args:
            question: User question
            top_k: Number of context documents
            include_sources: Whether to include sources
            use_all_enhancements: Whether to use all enhancements
            conversation_id: ID for conversation memory (optional)
            
        Yields:
            Dict: Stream chunks with type, content, and metadata
        """
        print(f"❓ Enhanced Q&A Stream: {question}")
        
        try:
            # Generate unique response ID for tracking
            response_id = str(uuid.uuid4())
            
            # Start streaming
            yield {
                'type': 'start',
                'message': 'Iniciando busca avançada...',
                'conversation_id': conversation_id,
                'response_id': response_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Enhance question with conversation memory if enabled
            original_question = question
            context_summary = ""
            
            if self.conversation_memory and use_all_enhancements and conversation_id:
                enhanced_question, context_summary = self.conversation_memory.enhance_question_with_context(
                    conversation_id, question
                )
                if enhanced_question != question:
                    yield {
                        'type': 'context',
                        'message': 'Contexto da conversa aplicado',
                        'enhanced': True
                    }
                    question = enhanced_question
            
            # Enhanced document search
            yield {
                'type': 'search',
                'message': 'Buscando documentos relevantes...',
            }
            
            search_results = await self.enhanced_search_documents(
                query=question,
                top_k=top_k,
                similarity_threshold=0.5,
                use_fusion=self.enable_rag_fusion and use_all_enhancements,
                use_reranking=self.enable_reranking and use_all_enhancements
            )
            
            # Debug removed - was causing issues
            
            yield {
                'type': 'search_complete',
                'message': f'Encontrados {len(search_results)} documentos',
                'sources_count': len(search_results)
            }
            
            if not search_results:
                yield {
                    'type': 'error',
                    'message': 'Nenhum documento relevante encontrado'
                }
                return
            
            # Context compression if enabled
            context_text = "\n\n".join([
                doc.get('content', '') if isinstance(doc, dict) else str(doc) 
                for doc in search_results
            ])
            compression_stats = None
            
            if self.context_compressor and use_all_enhancements:
                yield {
                    'type': 'compression',
                    'message': 'Comprimindo contexto...',
                }
                
                # Filter high-quality results
                high_quality_results = [r for r in search_results if r.get('similarity', 0) >= 0.4]
                if not high_quality_results:
                    high_quality_results = search_results[:3]
                
                # Transform search results to expected format for compression
                context_chunks = []
                for result in high_quality_results:
                    chunk = {
                        'content': result.get('content', ''),
                        'metadata': {
                            'source': result.get('metadata', {}).get('source', 'unknown'),
                            'page': result.get('metadata', {}).get('page', 'N/A')
                        }
                    }
                    context_chunks.append(chunk)
                
                compressed = self.context_compressor.compress_context(
                    context_chunks, question, preserve_citations=True
                )
                
                # Use compressed context if it's high quality
                if compressed.relevance_score > 0.7:
                    context_text = compressed.compressed_text
                else:
                    context_text = "\n\n".join([
                        doc.get('content', '') if isinstance(doc, dict) else str(doc) 
                        for doc in search_results
                    ])
                
                compression_stats = {
                    'compression_ratio': compressed.compression_ratio,
                    'relevance_score': compressed.relevance_score,
                    'original_length': len(compressed.original_text),
                    'compressed_length': len(compressed.compressed_text)
                }
            
            # Generate streaming response
            yield {
                'type': 'generation_start',
                'message': 'Gerando resposta...',
                'compression_stats': compression_stats
            }
            
            # Prepare academic citations for streaming
            self.academic_formatter.clear()
            citation_mapping = {}
            
            # Build system prompt with academic citation instructions
            citation_instructions = ""
            if include_sources:
                citation_instructions = """
INSTRUÇÕES DE CITAÇÃO:
- Use citações numeradas [1], [2], [3] etc. ao referenciar informações específicas
- Cite sempre que usar informações específicas de um documento"""
            
            system_prompt = f"""Você é um assistente especialista em Engenharia de Reservatórios.

CONTEXTO DOS DOCUMENTOS:
{context_text}

{citation_instructions}

FORMATO DE RESPOSTA:
- Responda em português brasileiro de forma natural e direta, sem se apresentar
- Seja preciso e técnico quando apropriado
- Use citações numeradas [1], [2], [3] para referenciar informações específicas dos documentos
- NÃO inclua uma seção de referências no final
- Se não souber, diga que não encontrou informações suficientes nos documentos"""
            
            if include_sources:
                for i, result in enumerate(search_results[:top_k]):
                    source_data = {
                        'filename': result.get('metadata', {}).get('filename', result.get('metadata', {}).get('source', 'Unknown')),
                        'source': result.get('metadata', {}).get('source', 'Unknown'),
                        'page': result.get('metadata', {}).get('page', 'N/A'),
                        'preview': result['content'][:300] if len(result['content']) > 300 else result['content'],
                        'content': result['content'],
                        'file_path': result.get('metadata', {}).get('document_path', '')
                    }
                    
                    source_id = f"{source_data['filename']}_{source_data['page']}"
                    citation_number = self.academic_formatter.add_source(source_id, source_data)
                    citation_mapping[source_id] = citation_number
            
            # Stream tokens from LLM
            full_answer = ""
            async for token in self.llm_client.generate_response_stream(
                prompt=question,
                system_prompt=system_prompt,
                max_tokens=1000,
                temperature=0.7
            ):
                full_answer += token
                yield {
                    'type': 'token',
                    'content': token,
                    'response_id': response_id
                }
                # Small delay to make streaming more visible
                await asyncio.sleep(0.02)
            
            # Generate and stream academic references
            academic_references = self.academic_formatter.format_reference_list()
            if academic_references:
                # Stream the academic references
                for char in academic_references:
                    yield {
                        'type': 'token',
                        'content': char,
                        'response_id': response_id
                    }
                    await asyncio.sleep(0.01)
                
                full_answer += academic_references
            
            # Calculate confidence and prepare sources
            confidence = min(0.95, len(search_results) * 0.2 + 0.3)
            sources = []
            
            if include_sources:
                for i, doc in enumerate(search_results[:3]):
                    if isinstance(doc, dict):
                        sources.append({
                            'id': f"source_{i+1}",
                            'title': doc.get('metadata', {}).get('source', f'Documento {i+1}'),
                            'content': doc.get('content', '')[:200] + '...',
                            'similarity': doc.get('similarity', 0.0),
                            'page': doc.get('metadata', {}).get('page'),
                            'chunk_id': doc.get('id')
                        })
                    else:
                        sources.append({
                            'id': f"source_{i+1}",
                            'title': f'Documento {i+1}',
                            'content': str(doc)[:200] + '...',
                            'similarity': 0.0,
                            'page': None,
                            'chunk_id': f"chunk_{i+1}"
                        })
            
            # Final response metadata
            yield {
                'type': 'complete',
                'answer': full_answer,
                'confidence': confidence,
                'sources': sources,
                'total_sources': len(search_results),
                'enhanced': use_all_enhancements,
                'compression_stats': compression_stats,
                'response_id': response_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store interaction in conversation memory if enabled
            if self.conversation_memory and conversation_id:
                try:
                    self.conversation_memory.add_interaction(
                        conversation_id=conversation_id,
                        user_question=original_question,
                        assistant_response=full_answer,
                        response_metadata={
                            'sources': sources,
                            'confidence': confidence,
                            'context_summary': context_summary,
                            'response_id': response_id
                        }
                    )
                except Exception as mem_error:
                    print(f"⚠️ Warning: Failed to store conversation memory: {mem_error}")
            
        except Exception as e:
            print(f"❌ Error in enhanced streaming Q&A: {e}")
            yield {
                'type': 'error',
                'message': f"Erro ao processar pergunta: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_enhanced_confidence(self, 
                                     results: List[Dict[str, Any]], 
                                     enhancements_used: bool) -> float:
        """Calculate confidence score considering enhancements"""
        if not results:
            return 0.0
        
        # Base confidence from similarity scores
        similarities = [r.get('similarity', 0.0) for r in results[:3]]
        base_confidence = sum(s ** 2 for s in similarities) / len(similarities)
        
        # Enhancement bonuses
        enhancement_bonus = 0.0
        
        if enhancements_used:
            # Fusion bonus
            if any('fusion_score' in r for r in results):
                enhancement_bonus += 0.1
            
            # Re-ranking bonus
            if any('cross_encoder_score' in r for r in results):
                enhancement_bonus += 0.15
            
            # Semantic chunking bonus
            if any(r.get('metadata', {}).get('section_type') in ['abstract', 'definition', 'equation'] 
                   for r in results):
                enhancement_bonus += 0.1
        
        # Final confidence
        final_confidence = min(1.0, base_confidence + enhancement_bonus)
        return round(final_confidence, 3)
    
    async def stream_enhanced_response(self, 
                                      question: str,
                                      conversation_id: Optional[str] = None,
                                      top_k: int = 6) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream enhanced response with progress updates
        """
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        try:
            # Start streaming
            yield {
                'type': 'start',
                'message': 'Iniciando busca avançada...',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # RAG Fusion progress
            if self.enable_rag_fusion:
                yield {
                    'type': 'progress',
                    'message': 'Expandindo query com RAG Fusion...',
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.1)
            
            # Search progress
            yield {
                'type': 'progress',
                'message': 'Buscando documentos relevantes...',
                'timestamp': datetime.now().isoformat()
            }
            
            # Perform enhanced search
            search_results = await self.enhanced_search_documents(question, top_k * 2)
            
            # Re-ranking progress
            if self.enable_reranking:
                yield {
                    'type': 'progress',
                    'message': 'Aplicando re-ranking inteligente...',
                    'timestamp': datetime.now().isoformat()
                }
                await asyncio.sleep(0.1)
            
            # Context enhancement progress
            yield {
                'type': 'progress',
                'message': 'Enriquecendo contexto...',
                'timestamp': datetime.now().isoformat()
            }
            
            # Get enhanced answer
            result = await self.enhanced_ask_question(question, top_k, True, True)
            
            # Stream context info
            yield {
                'type': 'context',
                'message': f'Encontrados {result["total_sources"]} documentos relevantes',
                'confidence': result['confidence'],
                'enhancements': result['enhancements_used'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Stream answer
            answer_words = result['answer'].split()
            current_answer = ""
            
            for i, word in enumerate(answer_words):
                current_answer += word + " "
                
                if i % 5 == 0 or i == len(answer_words) - 1:
                    yield {
                        'type': 'answer_chunk',
                        'content': current_answer.strip(),
                        'is_complete': i == len(answer_words) - 1,
                        'timestamp': datetime.now().isoformat()
                    }
                    await asyncio.sleep(0.05)
            
            # Send enhanced sources
            if result.get('sources'):
                yield {
                    'type': 'sources',
                    'sources': result['sources'][:3],
                    'timestamp': datetime.now().isoformat()
                }
            
            # Final completion
            yield {
                'type': 'complete',
                'final_answer': result['answer'],
                'confidence': result['confidence'],
                'conversation_id': conversation_id,
                'enhanced': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e),
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive status of enhanced system"""
        base_status = self.get_system_status()
        
        enhanced_info = {
            'enhanced_rag': {
                'version': '2.0',
                'semantic_chunking': self.enable_semantic_chunking,
                'rag_fusion': self.enable_rag_fusion,
                'cross_encoder_reranking': self.enable_reranking,
                'context_enhancement': True
            },
            'performance_features': {
                'parallel_query_processing': True,
                'semantic_chunk_analysis': self.enable_semantic_chunking,
                'multi_stage_reranking': self.enable_reranking,
                'technical_domain_boosting': True,
                'definition_gap_detection': True
            }
        }
        
        # Add component-specific information
        if self.rag_fusion:
            enhanced_info['rag_fusion_info'] = self.rag_fusion.get_system_info()
        
        if self.reranker:
            enhanced_info['reranker_info'] = self.reranker.get_system_info()
        
        return {**base_status, **enhanced_info}
    
    def record_user_feedback(self, 
                            response_id: str,
                            user_id: str,
                            question: str,
                            response_text: str,
                            rating: int,
                            feedback_type: str = "rating",
                            feedback_text: Optional[str] = None,
                            aspects: Optional[Dict[str, int]] = None) -> str:
        """
        Record user feedback for continuous learning
        
        Args:
            response_id: ID of the response being rated
            user_id: ID of the user providing feedback
            question: Original question
            response_text: Generated response
            rating: 1-5 rating scale
            feedback_type: Type of feedback
            feedback_text: Optional text feedback
            aspects: Ratings for different aspects (accuracy, relevance, etc.)
            
        Returns:
            feedback_id: Unique ID for this feedback
        """
        if not self.feedback_system:
            print("⚠️ Feedback learning system not enabled")
            return ""
        
        try:
            feedback_id = self.feedback_system.record_feedback(
                response_id=response_id,
                user_id=user_id,
                question=question,
                response_text=response_text,
                rating=rating,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                aspects=aspects
            )
            
            print(f"✅ Feedback recorded (ID: {feedback_id[:8]}...)")
            return feedback_id
            
        except Exception as e:
            print(f"❌ Error recording feedback: {e}")
            return ""
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get comprehensive feedback and learning statistics"""
        if not self.feedback_system:
            return {"feedback_enabled": False}
        
        try:
            stats = self.feedback_system.get_feedback_stats()
            stats["feedback_enabled"] = True
            return stats
        except Exception as e:
            return {"feedback_enabled": False, "error": str(e)}


# Global enhanced RAG instance
_global_enhanced_rag = None

def get_enhanced_rag_system() -> EnhancedRAGSystem:
    """
    Get or create global enhanced RAG system instance
    """
    global _global_enhanced_rag
    if _global_enhanced_rag is None:
        _global_enhanced_rag = EnhancedRAGSystem()
    return _global_enhanced_rag


# Setup function for quick initialization
def setup_enhanced_isabel_rag(pdf_directory: str, 
                             vectorstore_path: str = "./vectorstore_enhanced") -> EnhancedRAGSystem:
    """
    Quick setup of enhanced RAG system
    
    Args:
        pdf_directory: Directory with engineering PDFs
        vectorstore_path: Path for enhanced vector store
    """
    print("🔧 Setting up Enhanced IAbel RAG System...")
    
    # Create enhanced system
    rag = EnhancedRAGSystem(vectorstore_path=vectorstore_path)
    
    # Index PDFs with enhancements
    success = rag.index_pdf_directory(pdf_directory)
    
    if success:
        # Display enhanced status
        status = rag.get_enhanced_status()
        print("\n📊 Enhanced System Status:")
        print(f"   📚 Documents: {status['vector_store']['total_documents']}")
        print(f"   🤖 LLM: {status['llm']['model']} ({status['llm']['status']})")
        print(f"   🧠 Semantic Chunking: {'✅' if status['enhanced_rag']['semantic_chunking'] else '❌'}")
        print(f"   🔀 RAG Fusion: {'✅' if status['enhanced_rag']['rag_fusion'] else '❌'}")
        print(f"   🎯 Re-ranking: {'✅' if status['enhanced_rag']['cross_encoder_reranking'] else '❌'}")
    
    return rag


if __name__ == "__main__":
    # Example usage
    pdf_dir = "../backend/data/pdfs"
    
    if os.path.exists(pdf_dir):
        isabel = setup_enhanced_isabel_rag(pdf_dir)
        
        # Demo enhanced capabilities
        print("\n" + "="*60)
        print("🚀 Enhanced IAbel RAG System Ready!")
        print("   Try asking technical questions about reservoir engineering")
        print("="*60)
        
        # Start chat session (async wrapper needed for enhanced features)
        isabel.chat_session()
    else:
        print(f"❌ PDF directory not found: {pdf_dir}")
        print("   Please add PDFs to enable enhanced functionality")