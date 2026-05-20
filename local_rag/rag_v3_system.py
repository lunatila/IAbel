"""
RAG v3 System - English responses with (Author, Year) citations
Uses Google Gemini API and enhanced metadata extraction
"""

import os
import sys
from typing import List, Dict, Any, Optional
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
    RAG v3 System with English responses and academic citations
    """

    def __init__(self,
                 vectorstore_path: str = "data/vectorstore_v3",
                 embedder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 llm_model: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 80):
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

                # Process PDF into chunks
                print("   🔨 Creating chunks...")
                chunks = self.pdf_processor.process_pdf(pdf_path)

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
                    top_k: int = 6,
                    include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer question with English response and (Author, Year) citations

        Args:
            question: User question
            top_k: Number of context documents
            include_sources: Whether to include sources

        Returns:
            Answer with metadata
        """
        # Search for relevant documents
        search_results = self.search_documents(question, top_k=top_k)

        if not search_results:
            return {
                'answer': "I couldn't find relevant information in the indexed documents to answer your question.",
                'confidence': 0.0,
                'total_sources': 0,
                'sources': []
            }

        # Build context with citation mapping
        context_docs = []
        citation_map = {}

        for i, result in enumerate(search_results):
            content = result['content']
            metadata = result['metadata']

            # Get citation
            citation = metadata.get('citation', f"(Unknown, n.d.)")
            filename = metadata.get('filename', metadata.get('source', 'Unknown'))

            # Map document to its citation
            doc_key = f"doc_{i+1}"
            citation_map[doc_key] = citation

            # Add to context with marker
            context_docs.append(f"[{doc_key}] {content}")

        context_text = "\n\n".join(context_docs)

        # Create citation instructions
        citation_instructions = "When referencing information from the documents, use inline citations in the format (Author, Year) or (Author et al., Year). "
        citation_instructions += f"Citation mapping: {'; '.join([f'{k}={v}' for k, v in citation_map.items()])}"

        # Build system prompt (English)
        system_prompt = f"""You are an expert assistant in Reservoir Engineering.

REFERENCE DOCUMENTS:
{context_text}

{citation_instructions}

CRITICAL RESPONSE INSTRUCTIONS:
- Respond in English (even if documents are in Portuguese)
- Be technical and precise
- IMPORTANT: Use ONLY information that is explicitly stated in the reference documents above
- DO NOT make assumptions, infer meanings, or provide definitions not present in the documents
- Use inline citations in (Author, Year) format when referencing specific information
- Example: "The INSIM-FT method (Dimary, 2024) uses streamline simulation..."
- If information is not in the documents, clearly state: "This information is not available in the provided documents"
- When providing definitions or acronym meanings, quote the EXACT definition from the documents
- Use appropriate reservoir engineering terminology as found in the documents
- Provide clear and detailed explanations based strictly on document content"""

        # Generate response
        answer = self.llm_client.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.7
        )

        # Replace document markers with actual citations
        for doc_key, citation in citation_map.items():
            # Replace [doc_1] with actual citation
            answer = answer.replace(f"[{doc_key}]", citation)

        # Calculate confidence
        confidences = [r.get('similarity', 0) for r in search_results]
        avg_confidence = sum(confidences[:3]) / min(3, len(confidences)) if confidences else 0

        # Format sources
        sources = []
        if include_sources:
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

        return {
            'answer': answer,
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
                    'dimension': self.embedder.dimension
                },
                'llm': {
                    'provider': 'gemini',
                    'model': self.llm_client.model_name
                },
                'features': {
                    'language': 'English',
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
