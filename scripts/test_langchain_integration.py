#!/usr/bin/env python3
"""
Teste da integração com LangChain
Verifica se o novo processador de documentos está funcionando corretamente
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "local_rag"))

def test_langchain_processor():
    """Test LangChain document processor standalone"""
    try:
        from local_rag.processors.langchain_document_processor import LangChainDocumentProcessor
        
        print("🧪 Testing LangChain Document Processor...")
        
        # Initialize processor
        processor = LangChainDocumentProcessor(chunk_size=700, chunk_overlap=120)
        
        # Test with PDF directory
        pdf_directory = "./backend/data/pdfs"
        
        if not os.path.exists(pdf_directory):
            print(f"❌ PDF directory not found: {pdf_directory}")
            return False
        
        # Process documents
        chunks = processor.process_directory(pdf_directory)
        
        if chunks:
            print(f"✅ Successfully processed {len(chunks)} chunks")
            
            # Show sample chunk
            sample_chunk = chunks[0]
            print(f"📄 Sample chunk:")
            print(f"   Title: {sample_chunk.metadata.get('title', 'N/A')}")
            print(f"   Page: {sample_chunk.metadata.get('page', 'N/A')}")
            print(f"   Priority: {sample_chunk.metadata.get('priority_section', 'N/A')}")
            print(f"   Content length: {len(sample_chunk.content)} chars")
            print(f"   Content preview: {sample_chunk.content[:200]}...")
            
            return True
        else:
            print("❌ No chunks created")
            return False
            
    except Exception as e:
        print(f"❌ Error testing LangChain processor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_rag_with_langchain():
    """Test Enhanced RAG System with LangChain enabled"""
    try:
        from local_rag.enhanced_rag_system import EnhancedRAGSystem
        
        print("\n🧪 Testing Enhanced RAG System with LangChain...")
        
        # Initialize with LangChain enabled
        rag_system = EnhancedRAGSystem(
            vectorstore_path="./vectorstore_langchain_test",
            use_langchain_loader=True,
            enable_semantic_chunking=False,  # LangChain has its own chunking
            enable_rag_fusion=False,
            enable_reranking=False,
            enable_context_compression=False,
            enable_citation_tracking=True,
            enable_feedback_learning=False
        )
        
        # Test directory processing
        pdf_directory = "./backend/data/pdfs"
        
        if not os.path.exists(pdf_directory):
            print(f"❌ PDF directory not found: {pdf_directory}")
            return False
        
        print("🔄 Testing document indexing...")
        success = rag_system.index_pdf_directory(pdf_directory, force_reindex=True)
        
        if success:
            print("✅ Indexing successful!")
            
            # Test search
            print("🔍 Testing document search...")
            results = rag_system.search_documents("INSIM simulador", top_k=3)
            
            if results:
                print(f"✅ Search successful! Found {len(results)} results")
                for i, result in enumerate(results):
                    print(f"   Result {i+1}:")
                    print(f"     Source: {result.get('metadata', {}).get('title', 'N/A')}")
                    print(f"     Page: {result.get('metadata', {}).get('page', 'N/A')}")
                    print(f"     Similarity: {result.get('similarity', 0):.3f}")
                    print(f"     Preview: {result.get('content', '')[:100]}...")
                return True
            else:
                print("❌ Search returned no results")
                return False
        else:
            print("❌ Indexing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Enhanced RAG with LangChain: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 TESTING LANGCHAIN INTEGRATION")
    print("=" * 70)
    
    # Test 1: LangChain processor standalone
    test1_success = test_langchain_processor()
    
    # Test 2: Enhanced RAG with LangChain
    test2_success = test_enhanced_rag_with_langchain()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"✅ LangChain Processor Test: {'PASSED' if test1_success else 'FAILED'}")
    print(f"✅ Enhanced RAG Integration Test: {'PASSED' if test2_success else 'FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! LangChain integration is working correctly.")
        print("   You can now use Enhanced RAG with better document processing!")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)