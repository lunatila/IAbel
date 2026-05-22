"""
Enhanced RAG Service - Integrates the improved local RAG system with FastAPI
Provides high-performance, multilingual document search with context enhancement
"""

import sys
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
import asyncio
import uuid
import json
from datetime import datetime

# Add local_rag to path - fix for correct project structure
# From backend/app/services/rag_service.py -> go up to project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))  # Add project root to path

# Import enhanced RAG components
try:
    from local_rag.enhanced_rag_system import EnhancedRAGSystem
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced RAG System available with all improvements")
except ImportError as e:
    print(f"⚠️ Enhanced RAG System not available: {e}")
    try:
        from local_rag.rag_system import LocalRAGSystem
        ENHANCED_AVAILABLE = False
        print("📄 Using Basic RAG System as fallback")
    except ImportError as e2:
        print(f"❌ No RAG system available: {e2}")
        ENHANCED_AVAILABLE = False

# Import RAG v3 components
try:
    from local_rag.rag_v3_system import RAGv3System
    RAG_V3_AVAILABLE = True
    print("✅ RAG v3 System available (English + Academic Citations)")
except ImportError as e:
    print(f"⚠️ RAG v3 System not available: {e}")
    RAG_V3_AVAILABLE = False

# Fix relative imports
import sys
services_dir = Path(__file__).parent
app_dir = services_dir.parent
sys.path.insert(0, str(services_dir))
sys.path.insert(0, str(app_dir))

from cache_service import get_cache_service
from utils.logging_config import LoggerMixin, log_performance, get_error_tracker


class EnhancedRAGService(LoggerMixin):
    """
    Enhanced RAG service that integrates the improved local RAG system
    with FastAPI for high-performance web responses
    """
    
    def __init__(self,
                 pdf_directory: str = "data/pdfs",
                 vectorstore_path: str = "data/vectorstore",
                 embedder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 llm_model: str = None,
                 llm_provider: str = None):
        """
        Initialize enhanced RAG service

        Args:
            pdf_directory: Directory containing PDF documents
            vectorstore_path: Path for vector database storage
            embedder_model: Multilingual embedding model
            llm_model: LLM model to use (default from env)
            llm_provider: LLM provider: "ollama", "gemini" (default from env)
        """
        self.pdf_directory = pdf_directory
        self.vectorstore_path = vectorstore_path

        # Get LLM configuration from environment variables
        llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "ollama")

        # Get model based on provider
        if llm_provider == "gemini":
            llm_model = llm_model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        else:
            llm_model = llm_model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        print(f"🤖 Configurando LLM: Provider={llm_provider}, Model={llm_model}")

        # Initialize enhanced RAG system if available, otherwise fallback to basic
        if ENHANCED_AVAILABLE:
            print("🚀 Using Enhanced RAG System with all improvements")
            self.rag_system = EnhancedRAGSystem(
                vectorstore_path=vectorstore_path,
                embedder_model=embedder_model,
                llm_model=llm_model,
                llm_provider=llm_provider,
                chunk_size=500,
                chunk_overlap=80,
                enable_semantic_chunking=True,
                enable_rag_fusion=True,
                enable_reranking=False,
                enable_context_compression=True,
                enable_citation_tracking=True,
                enable_feedback_learning=True,
                enable_conversation_memory=True
            )
            self.enhanced_mode = True
        else:
            print("⚠️ Using Basic RAG System (Enhanced components not available)")
            self.rag_system = LocalRAGSystem(
                vectorstore_path=vectorstore_path,
                embedder_model=embedder_model,
                llm_model=llm_model,
                llm_provider=llm_provider,
                chunk_size=500,
                chunk_overlap=80
            )
            self.enhanced_mode = False

        # Initialize RAG v3 system if available
        self.rag_v3_system = None
        if RAG_V3_AVAILABLE:
            try:
                print("🚀 Initializing RAG v3 System...")
                self.rag_v3_system = RAGv3System(
                    vectorstore_path="data/vectorstore_v3",
                    embedder_model=embedder_model,
                    llm_model=llm_model if llm_provider == "gemini" else os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
                    chunk_size=500,
                    chunk_overlap=80
                )
                print("✅ RAG v3 System initialized successfully")
            except Exception as e:
                print(f"⚠️ Failed to initialize RAG v3 System: {e}")
                self.rag_v3_system = None

        # Conversation storage — capped at 100 sessions (LRU eviction)
        from collections import OrderedDict
        self.conversations: OrderedDict = OrderedDict()
        self._MAX_CONVERSATIONS = 100
        
        # Initialize cache service
        self.cache = get_cache_service()
        
        # Initialize error tracker
        self.error_tracker = get_error_tracker()
        
        # Initialize if PDF directory exists
        self.log_operation("RAG Service Initialization")
        try:
            success = self._initialize_system()
            if success:
                self.log_success("RAG Service Initialization")
            else:
                self.log_warning("RAG Service partially initialized", status="partial")
        except Exception as e:
            self.log_error("RAG Service Initialization", e)
            self.error_tracker.track_error(e, {"component": "service_init"})
    
    def _ensure_conversation_slot(self, conversation_id: str) -> None:
        """Create conversation slot, evicting the oldest session if at capacity."""
        if conversation_id not in self.conversations:
            if len(self.conversations) >= self._MAX_CONVERSATIONS:
                self.conversations.popitem(last=False)
            self.conversations[conversation_id] = []
        self.conversations.move_to_end(conversation_id)

    def _initialize_system(self) -> bool:
        """
        Initialize the RAG system with existing PDFs
        """
        try:
            if os.path.exists(self.pdf_directory):
                # Check if vectorstore already exists and has documents
                stats = self.rag_system.get_system_status()
                if stats['vector_store']['total_documents'] == 0:
                    print("🔄 Indexing PDF documents...")
                    success = self.rag_system.index_pdf_directory(self.pdf_directory)
                    if success:
                        print("✅ RAG system initialized successfully")
                        return True
                    else:
                        print("❌ Failed to initialize RAG system")
                        return False
                else:
                    print(f"✅ RAG system loaded with {stats['vector_store']['total_documents']} documents")
                    return True
            else:
                print(f"⚠️ PDF directory not found: {self.pdf_directory}")
                print("   Create directory and add PDFs to enable full functionality")
                return False
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
    
    @log_performance("document_search")
    async def search_documents(self, 
                              query: str, 
                              top_k: int = 8,
                              similarity_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Search documents with enhanced RAG capabilities
        
        Args:
            query: Search query
            top_k: Number of documents to return
            similarity_threshold: Minimum similarity threshold
        """
        try:
            # Check cache first
            search_params = {"top_k": top_k, "similarity_threshold": similarity_threshold}
            cached_results = self.cache.get_search_results(query, search_params)
            
            if cached_results is not None:
                return {
                    'results': cached_results,
                    'total_found': len(cached_results),
                    'query_processed': query,
                    'cached': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Use enhanced search if available (without creating new event loop)
            if self.enhanced_mode and hasattr(self.rag_system, 'search_documents'):
                # Use direct search function to avoid event loop issues
                results = self.rag_system.search_documents(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
            else:
                # Fallback to basic search
                results = self.rag_system.search_documents(
                    query=query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
            
            # Format for API response
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result['content'],
                    'source': result['metadata'].get('source', 'Unknown'),
                    'page': result['metadata'].get('page', 'N/A'),
                    'similarity': round(result['similarity'], 3),
                    'priority_section': result['metadata'].get('priority_section', 'regular'),
                    'matched_query': result.get('matched_query', query),
                    'boost_applied': result.get('priority_boost', 1.0)
                })
            
            # Cache the results
            self.cache.cache_search_results(query, search_params, formatted_results)
            
            return {
                'results': formatted_results,
                'total_found': len(results),
                'query_processed': query,
                'cached': False,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.log_error("document_search", e, query=query, top_k=top_k)
            self.error_tracker.track_error(e, {
                "component": "search",
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold
            })
            return {
                'results': [],
                'total_found': 0,
                'error': str(e),
                'query_processed': query,
                'timestamp': datetime.now().isoformat()
            }
    
    @log_performance("question_answering")
    async def ask_question(self, 
                          question: str,
                          conversation_id: Optional[str] = None,
                          top_k: int = 6,
                          include_sources: bool = True) -> Dict[str, Any]:
        """
        Ask question using enhanced RAG system
        
        Args:
            question: User question
            conversation_id: Optional conversation ID
            top_k: Number of context documents
            include_sources: Whether to include source information
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Create context hash for caching
            search_results = self.rag_system.search_documents(question, top_k=top_k)
            context_data = [r['content'] for r in search_results[:3]]  # Top 3 for hash
            context_hash = hashlib.md5(str(context_data).encode()).hexdigest()
            
            # Check cache for QA response
            cached_response = self.cache.get_qa_response(question, context_hash)
            if cached_response is not None:
                cached_response['conversation_id'] = conversation_id
                cached_response['cached'] = True
                cached_response['timestamp'] = datetime.now().isoformat()
                return cached_response
            
            # Use enhanced ask_question if available (without event loops)
            if self.enhanced_mode and hasattr(self.rag_system, 'ask_question'):
                # Use direct ask_question to avoid event loop issues
                result = self.rag_system.ask_question(
                    question=question,
                    top_k=top_k,
                    include_sources=include_sources,
                    conversation_id=conversation_id
                )
            else:
                # Fallback to basic ask_question
                result = self.rag_system.ask_question(
                    question=question,
                    top_k=top_k,
                    include_sources=include_sources
                )
            
            # Store conversation (in production, use Redis with TTL)
            self._ensure_conversation_slot(conversation_id)
            
            self.conversations[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources_count': result['total_sources']
            })
            
            # Format response for API
            api_response = {
                'answer': result['answer'],
                'confidence': result['confidence'],
                'conversation_id': conversation_id,
                'total_sources': result['total_sources'],
                'cached': False,
                'enhanced_mode': self.enhanced_mode,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add enhanced features info if available
            if self.enhanced_mode and 'enhancements_used' in result:
                api_response['enhancements_used'] = result['enhancements_used']
            
            if include_sources and result['sources']:
                api_response['sources'] = result['sources']
            
            # Cache the response (without conversation_id for reusability)
            cache_response = api_response.copy()
            cache_response.pop('conversation_id', None)
            cache_response.pop('timestamp', None)
            self.cache.cache_qa_response(question, context_hash, cache_response)
            
            return api_response
            
        except Exception as e:
            self.log_error("question_answering", e, 
                          question=question, conversation_id=conversation_id)
            self.error_tracker.track_error(e, {
                "component": "qa",
                "question": question[:100],  # Limit for privacy
                "conversation_id": conversation_id,
                "top_k": top_k
            })
            return {
                'answer': f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}",
                'confidence': 0.0,
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'total_sources': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def ask_question_basic(self, 
                               question: str,
                               conversation_id: Optional[str] = None,
                               top_k: int = 6,
                               include_sources: bool = True) -> Dict[str, Any]:
        """
        Ask question using BASIC RAG system only (for v1)
        Forces use of basic functionality even if enhanced is available
        
        Args:
            question: User question
            conversation_id: Optional conversation ID
            top_k: Number of context documents
            include_sources: Whether to include source information
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Use basic functionality - search + generate without enhancements
            search_results = self.rag_system.search_documents(question, top_k=top_k)
            
            if not search_results:
                return {
                    'answer': "Não encontrei informações relevantes nos documentos indexados.",
                    'confidence': 0.0,
                    'conversation_id': conversation_id,
                    'total_sources': 0,
                    'generation_mode': 'basic_v1',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Use basic context building - no enhancements
            context_docs = [result['content'] for result in search_results[:top_k]]
            answer = self.rag_system.llm_client.generate_with_context(
                question=question,
                context_documents=context_docs
            )
            
            result = {
                'answer': answer,
                'confidence': sum(r.get('similarity', 0) for r in search_results[:3]) / min(3, len(search_results)),
                'total_sources': len(search_results),
                'sources': search_results if include_sources else []
            }
            
            # Store conversation
            self._ensure_conversation_slot(conversation_id)
            
            self.conversations[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources_count': result.get('total_sources', 0)
            })
            
            # Format response for API (basic mode)
            api_response = {
                'answer': result['answer'],
                'confidence': result['confidence'],
                'conversation_id': conversation_id,
                'total_sources': result.get('total_sources', 0),
                'generation_mode': 'basic_v1',
                'enhanced': False,  # Explicitly mark as non-enhanced
                'timestamp': datetime.now().isoformat()
            }
            
            # Add sources if requested
            if include_sources and 'sources' in result:
                api_response['sources'] = result['sources'][:5]  # Limit sources
            
            return api_response
            
        except Exception as e:
            self.log_error("basic_question_answering", e, 
                          question=question, conversation_id=conversation_id)
            return {
                'answer': f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}",
                'confidence': 0.0,
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'total_sources': 0,
                'generation_mode': 'basic_v1',
                'enhanced': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def ask_question_v3(self,
                             question: str,
                             conversation_id: Optional[str] = None,
                             top_k: int = 6,
                             include_sources: bool = True) -> Dict[str, Any]:
        """
        Ask question using RAG v3 system (English responses with academic citations)

        Args:
            question: User question
            conversation_id: Optional conversation ID
            top_k: Number of context documents
            include_sources: Whether to include source information
        """
        if not self.rag_v3_system:
            return {
                'answer': "RAG v3 system is not available. Please check system configuration.",
                'confidence': 0.0,
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'total_sources': 0,
                'generation_mode': 'rag_v3',
                'enhanced': True,
                'error': "RAG v3 not initialized",
                'timestamp': datetime.now().isoformat()
            }

        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Use RAG v3 system
            result = self.rag_v3_system.ask_question(
                question=question,
                top_k=top_k,
                include_sources=include_sources
            )

            # Store conversation
            self._ensure_conversation_slot(conversation_id)

            self.conversations[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources_count': result.get('total_sources', 0)
            })

            # Format response for API (v3 mode)
            api_response = {
                'answer': result['answer'],
                'confidence': result['confidence'],
                'conversation_id': conversation_id,
                'total_sources': result.get('total_sources', 0),
                'generation_mode': 'rag_v3',
                'enhanced': True,
                'timestamp': datetime.now().isoformat()
            }

            # Add sources if requested
            if include_sources and 'sources' in result:
                api_response['sources'] = result['sources'][:5]  # Limit sources

            return api_response

        except Exception as e:
            self.log_error("rag_v3_question_answering", e,
                          question=question, conversation_id=conversation_id)
            return {
                'answer': f"Sorry, an error occurred while processing your question: {str(e)}",
                'confidence': 0.0,
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'total_sources': 0,
                'generation_mode': 'rag_v3',
                'enhanced': True,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def ask_question_v3_stream(self,
                                     question: str,
                                     conversation_id: Optional[str] = None,
                                     top_k: int = 8,
                                     include_sources: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """Stream RAG v3 response with real token-level streaming."""
        if not self.rag_v3_system:
            yield {
                'type': 'complete',
                'answer': "RAG v3 system is not available.",
                'confidence': 0.0,
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'total_sources': 0,
                'generation_mode': 'rag_v3',
                'enhanced': True,
            }
            return

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        yield {'type': 'start', 'conversation_id': conversation_id}

        full_answer = ""
        final_chunk = None

        try:
            async for chunk in self.rag_v3_system.ask_question_stream(
                question=question,
                top_k=top_k,
                include_sources=include_sources
            ):
                if chunk.get('type') == 'token':
                    full_answer += chunk.get('content', '')
                    yield chunk
                elif chunk.get('type') == 'complete':
                    # Buffer complete — enrich before yielding
                    final_chunk = chunk
                else:
                    yield chunk

        except Exception as e:
            self.log_error("rag_v3_stream", e, question=question)
            yield {
                'type': 'error',
                'message': str(e),
                'conversation_id': conversation_id,
            }
            return

        # Persist conversation then yield enriched complete chunk
        if final_chunk:
            self._ensure_conversation_slot(conversation_id)
            self.conversations[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': full_answer,
                'confidence': final_chunk.get('confidence', 0.0),
                'sources_count': final_chunk.get('total_sources', 0),
            })
            final_chunk['conversation_id'] = conversation_id
            final_chunk['timestamp'] = datetime.now().isoformat()
            yield final_chunk

    async def stream_response(self, 
                             question: str,
                             conversation_id: Optional[str] = None,
                             top_k: int = 6) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream response for real-time chat experience
        
        Args:
            question: User question
            conversation_id: Optional conversation ID
            top_k: Number of context documents
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Start streaming response
            yield {
                'type': 'start',
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Search for relevant documents (async simulation)
            yield {
                'type': 'search',
                'message': 'Buscando documentos relevantes...',
                'timestamp': datetime.now().isoformat()
            }
            
            # Simulate async processing
            await asyncio.sleep(0.1)
            
            # Get enhanced response
            if self.enhanced_mode and hasattr(self.rag_system, 'enhanced_ask_question'):
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.rag_system.enhanced_ask_question(
                            question=question,
                            top_k=top_k,
                            include_sources=True,
                            use_all_enhancements=True,
                            conversation_id=conversation_id
                        )
                    )
                finally:
                    loop.close()
            else:
                result = self.rag_system.ask_question(
                    question=question,
                    top_k=top_k,
                    include_sources=True
                )
            
            # Stream context information
            yield {
                'type': 'context',
                'message': f'Encontrados {result["total_sources"]} documentos relevantes',
                'confidence': result['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Stream the answer (simulate word-by-word streaming)
            answer_words = result['answer'].split()
            current_answer = ""
            
            for i, word in enumerate(answer_words):
                current_answer += word + " "
                
                # Stream every few words to simulate real-time generation
                if i % 5 == 0 or i == len(answer_words) - 1:
                    yield {
                        'type': 'answer_chunk',
                        'content': current_answer.strip(),
                        'is_complete': i == len(answer_words) - 1,
                        'timestamp': datetime.now().isoformat()
                    }
                    await asyncio.sleep(0.05)  # Small delay for realistic streaming
            
            # Send sources
            if result.get('sources'):
                yield {
                    'type': 'sources',
                    'sources': result['sources'][:3],  # Top 3 sources
                    'timestamp': datetime.now().isoformat()
                }
            
            # Final completion
            yield {
                'type': 'complete',
                'final_answer': result['answer'],
                'confidence': result['confidence'],
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store conversation
            self._ensure_conversation_slot(conversation_id)
            
            self.conversations[conversation_id].append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources_count': result['total_sources']
            })
            
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e),
                'conversation_id': conversation_id or str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat()
            }
    
    async def reindex_documents(self, force_reindex: bool = False) -> AsyncIterator[Dict[str, Any]]:
        """
        Reindex documents with progress updates
        
        Args:
            force_reindex: Whether to force reindexing
        """
        try:
            yield {
                'type': 'start',
                'message': 'Iniciando reindexação...',
                'timestamp': datetime.now().isoformat()
            }
            
            if not os.path.exists(self.pdf_directory):
                yield {
                    'type': 'error',
                    'error': f'Diretório de PDFs não encontrado: {self.pdf_directory}',
                    'timestamp': datetime.now().isoformat()
                }
                return
            
            # Count PDFs
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
            total_files = len(pdf_files)
            
            yield {
                'type': 'progress',
                'message': f'Encontrados {total_files} arquivos PDF',
                'total_files': total_files,
                'timestamp': datetime.now().isoformat()
            }
            
            # Perform reindexing
            success = self.rag_system.index_pdf_directory(
                self.pdf_directory, 
                force_reindex=force_reindex
            )
            
            if success:
                # Get final stats
                stats = self.rag_system.get_system_status()
                
                yield {
                    'type': 'complete',
                    'message': 'Reindexação concluída com sucesso',
                    'total_documents': stats['vector_store']['total_documents'],
                    'embedder_model': stats['embedder']['model'],
                    'llm_model': stats['llm']['model'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                yield {
                    'type': 'error',
                    'error': 'Falha na reindexação dos documentos',
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            yield {
                'type': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        try:
            # Get RAG system status (enhanced or basic)
            if self.enhanced_mode and hasattr(self.rag_system, 'get_enhanced_status'):
                rag_status = self.rag_system.get_enhanced_status()
            else:
                rag_status = self.rag_system.get_system_status()
            
            # Get cache statistics
            cache_stats = self.cache.get_cache_stats()
            
            # Add service-specific information
            service_status = {
                'service': 'Enhanced RAG Service' if self.enhanced_mode else 'Basic RAG Service',
                'enhanced_mode': self.enhanced_mode,
                'enhanced_available': ENHANCED_AVAILABLE,
                'pdf_directory': self.pdf_directory,
                'vectorstore_path': self.vectorstore_path,
                'active_conversations': len(self.conversations),
                'cache': cache_stats,
                'timestamp': datetime.now().isoformat(),
                'capabilities': {
                    'multilingual_search': True,
                    'context_enhancement': True,
                    'priority_boosting': True,
                    'streaming_responses': True,
                    'acronym_definitions': True,
                    'semantic_chunking': self.enhanced_mode,
                    'rag_fusion': self.enhanced_mode,
                    'cross_encoder_reranking': self.enhanced_mode,
                    'self_critique': self.enhanced_mode,
                    'redis_caching': cache_stats.get('connected', False)
                }
            }
            
            # Combine status information
            return {**service_status, **rag_status}
            
        except Exception as e:
            return {
                'service': 'Enhanced RAG Service',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def add_pdf_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Add a new PDF document to the system
        
        Args:
            file_path: Path to the PDF file
            filename: Original filename
        """
        try:
            # Process single PDF
            chunks = self.rag_system.pdf_processor.process_pdf(file_path)
            
            if not chunks:
                return {
                    'success': False,
                    'error': 'No content extracted from PDF',
                    'filename': filename
                }
            
            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self.rag_system.embedder.embed_batch(texts, batch_size=32)
            
            # Prepare data for vector store
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]
            
            # Add to vector store
            success = self.rag_system.vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            if success:
                return {
                    'success': True,
                    'message': f'PDF {filename} processado e indexado com sucesso',
                    'chunks_created': len(chunks),
                    'filename': filename,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': 'Falha ao adicionar documentos ao vector store',
                    'filename': filename
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
    
    async def ask_question_stream(self, 
                                question: str,
                                conversation_id: Optional[str] = None,
                                top_k: int = 4,
                                include_sources: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """
        Ask question using enhanced RAG system with streaming response
        
        Args:
            question: User question
            conversation_id: Optional conversation ID
            top_k: Number of context documents
            include_sources: Whether to include source information
            
        Yields:
            Dict: Stream chunks with response data
        """
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Use enhanced streaming if available
            if self.enhanced_mode and hasattr(self.rag_system, 'enhanced_ask_question_stream'):
                async for chunk in self.rag_system.enhanced_ask_question_stream(
                    question=question,
                    top_k=top_k,
                    include_sources=include_sources,
                    use_all_enhancements=True,
                    conversation_id=conversation_id
                ):
                    yield chunk
            else:
                # Fallback to non-streaming
                yield {
                    'type': 'start',
                    'message': 'Processando pergunta...',
                    'conversation_id': conversation_id
                }
                
                result = await self.ask_question(
                    question=question,
                    conversation_id=conversation_id,
                    top_k=top_k,
                    include_sources=include_sources
                )
                
                yield {
                    'type': 'complete',
                    **result
                }
                
        except Exception as e:
            print(f"❌ Error in streaming Q&A: {e}")
            yield {
                'type': 'error',
                'message': f"Erro ao processar pergunta: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Create global service instance
enhanced_rag_service = None

def get_rag_service() -> EnhancedRAGService:
    """
    Get or create the enhanced RAG service instance
    """
    global enhanced_rag_service
    if enhanced_rag_service is None:
        enhanced_rag_service = EnhancedRAGService()
    return enhanced_rag_service