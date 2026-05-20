#!/usr/bin/env python3
"""
Script especializado para processar PDFs grandes que causam erro de memória
Processa apenas o arquivo 'tese insim_compressed.pdf' com otimizações especiais
"""

import os
import sys
from pathlib import Path
import gc

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "local_rag"))

def process_large_insim_pdf():
    """Process the large INSIM PDF with memory optimizations"""
    
    # Target file
    pdf_file = "/home/lacucaratila/Projetos/IAbel/backend/data/pdfs/tese insim_compressed.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ Arquivo não encontrado: {pdf_file}")
        return False
    
    print("🚀 Iniciando processamento otimizado do PDF grande INSIM...")
    print(f"📄 Arquivo: {os.path.basename(pdf_file)}")
    
    try:
        from local_rag.enhanced_rag_system import EnhancedRAGSystem
        
        # Create enhanced RAG system with smaller batch sizes for memory optimization
        print("🔧 Configurando sistema RAG com otimizações de memória...")
        
        rag_system = EnhancedRAGSystem(
            vectorstore_path="./vectorstore_enhanced",
            embedder_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            llm_model="llama3.2:3b",
            enable_semantic_chunking=True,
            enable_rag_fusion=True,
            enable_reranking=True,
            enable_compression=False,  # Disable compression to save memory
            enable_citation_tracking=False,  # Disable citation tracking to save memory
            enable_feedback_learning=False  # Disable feedback learning to save memory
        )
        
        # Configure optimized batch sizes for large PDFs
        rag_system.pdf_processor.batch_size = 100  # Larger batches to maintain context
        rag_system.pdf_processor.chunk_size = 700  # Standard technical chunk size
        rag_system.pdf_processor.chunk_overlap = 120  # Better context preservation
        
        print("📚 Processando PDF grande em modo otimizado...")
        
        # Process just this one file
        pdf_directory = os.path.dirname(pdf_file)
        pdf_filename = os.path.basename(pdf_file)
        
        # Create a temporary directory with just this file
        temp_files = [pdf_filename]
        
        # Create a temporary directory with just this file
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf_path = os.path.join(temp_dir, pdf_filename)
            shutil.copy2(pdf_file, temp_pdf_path)
            
            print(f"📁 Processando arquivo isoladamente em diretório temporário...")
            
            # Process with memory optimization
            success = rag_system.index_pdf_directory(temp_dir, force_reindex=False)
        
        if success:
            print("✅ PDF grande processado com sucesso!")
            
            # Verify the indexing worked
            status = rag_system.get_system_status()
            print(f"📊 Total de documentos no vectorstore: {status['vector_store']['total_documents']}")
            
            return True
        else:
            print("❌ Falha no processamento do PDF grande")
            return False
            
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up memory
        gc.collect()

def main():
    """Main function"""
    print("=" * 70)
    print("🔧 PROCESSADOR ESPECIALIZADO PARA PDFs GRANDES")
    print("   Otimizado para 'tese insim_compressed.pdf'")
    print("=" * 70)
    
    success = process_large_insim_pdf()
    
    if success:
        print("\n✅ Processamento concluído com sucesso!")
        print("   O arquivo grande foi indexado e está disponível para consulta.")
        print("   Agora você pode usar o sistema normalmente.")
    else:
        print("\n❌ Falha no processamento!")
        print("   Verifique os logs acima para mais detalhes.")
    
    return success

if __name__ == "__main__":
    main()