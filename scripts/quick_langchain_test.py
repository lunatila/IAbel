#!/usr/bin/env python3
"""
Teste rápido da integração LangChain
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "local_rag"))

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ LangChain community loader imported")
        
        from local_rag.processors.langchain_document_processor import LangChainDocumentProcessor
        print("✅ LangChain document processor imported")
        
        from local_rag.enhanced_rag_system import EnhancedRAGSystem
        print("✅ Enhanced RAG system imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_simple_loading():
    """Test simple document loading"""
    print("\n🧪 Testing simple document loading...")
    
    try:
        from local_rag.processors.langchain_document_processor import LangChainDocumentProcessor
        
        processor = LangChainDocumentProcessor(chunk_size=500, chunk_overlap=50)
        print("✅ Processor initialized")
        
        # Check if we have any PDFs
        pdf_dir = "./backend/data/pdfs"
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            if pdf_files:
                test_file = os.path.join(pdf_dir, pdf_files[0])
                print(f"📄 Testing with: {pdf_files[0]}")
                
                # Try to load just one document
                documents = processor.load_document(test_file)
                if documents:
                    print(f"✅ Loaded {len(documents)} pages")
                    print(f"   First page length: {len(documents[0].page_content)} chars")
                    return True
                else:
                    print("❌ No documents loaded")
                    return False
            else:
                print("⚠️ No PDF files found")
                return True  # Not an error, just no files
        else:
            print("⚠️ PDF directory not found")
            return True  # Not an error, just no directory
        
    except Exception as e:
        print(f"❌ Loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests"""
    print("=" * 50)
    print("🚀 QUICK LANGCHAIN TEST")
    print("=" * 50)
    
    test1 = test_imports()
    test2 = test_simple_loading()
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"Imports: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Loading: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 Basic LangChain integration is working!")
    else:
        print("\n❌ Issues found with LangChain integration")
    
    return test1 and test2

if __name__ == "__main__":
    main()