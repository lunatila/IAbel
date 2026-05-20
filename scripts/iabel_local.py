#!/usr/bin/env python3
"""
IAbel Local - Interface principal para o sistema RAG local
Sistema completo de IA para Engenharia de Reservatórios sem APIs externas
"""

import os
import sys
import argparse
from pathlib import Path

# Adiciona o diretório local_rag ao path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / "local_rag"))

# Garante que vectorstore existe antes de importar
vectorstore_dir = current_dir / "local_rag" / "vectorstore"
vectorstore_dir.mkdir(parents=True, exist_ok=True)
init_file = vectorstore_dir / "__init__.py"
if not init_file.exists():
    with open(init_file, 'w') as f:
        f.write("# Vector Store Module\n")

from local_rag.rag_system import LocalRAGSystem, setup_isabel_rag

def main():
    parser = argparse.ArgumentParser(
        description="IAbel Local - Assistente de Engenharia de Reservatórios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s --setup                     # Setup inicial do sistema
  %(prog)s --chat                      # Inicia chat interativo
  %(prog)s --index /path/to/pdfs       # Indexa PDFs específicos
  %(prog)s --query "O que é porosidade?" # Pergunta direta
  %(prog)s --status                    # Mostra status do sistema
        """
    )
    
    # Argumentos principais
    parser.add_argument('--setup', action='store_true',
                       help='Executa setup inicial do sistema')
    
    parser.add_argument('--chat', action='store_true',
                       help='Inicia sessão de chat interativo')
    
    parser.add_argument('--query', type=str,
                       help='Faz uma pergunta direta ao sistema')
    
    parser.add_argument('--index', type=str,
                       help='Indexa PDFs do diretório especificado')
    
    parser.add_argument('--status', action='store_true',
                       help='Mostra status atual do sistema')
    
    # Configurações
    parser.add_argument('--pdf-dir', type=str, 
                       default='./backend/data/pdfs',
                       help='Diretório dos PDFs (padrão: ./backend/data/pdfs)')
    
    parser.add_argument('--vectorstore', type=str,
                       default='./local_rag/vectorstore',
                       help='Diretório do vector store (padrão: ./local_rag/vectorstore)')
    
    parser.add_argument('--model', type=str,
                       default='llama3.2:3b',
                       help='Modelo LLM a usar (padrão: llama3.2:3b)')
    
    parser.add_argument('--force-reindex', action='store_true',
                       help='Força reindexação mesmo se já existirem dados')
    
    args = parser.parse_args()
    
    # Se nenhum argumento foi fornecido, mostra ajuda
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Configurações
    pdf_directory = args.pdf_dir
    vectorstore_path = args.vectorstore 
    llm_model = args.model
    
    try:
        # Setup inicial
        if args.setup:
            print("🔧 Executando setup inicial do IAbel Local...")
            
            # Verifica se diretório de PDFs existe
            if not os.path.exists(pdf_directory):
                print(f"❌ Diretório de PDFs não encontrado: {pdf_directory}")
                print("   Crie o diretório e adicione arquivos PDF")
                return
            
            # Cria sistema e indexa
            isabel = setup_isabel_rag(pdf_directory, vectorstore_path)
            print("✅ Setup concluído! Use --chat para começar a usar.")
            return
        
        # Cria sistema RAG
        isabel = LocalRAGSystem(
            vectorstore_path=vectorstore_path,
            llm_model=llm_model
        )
        
        # Indexação específica
        if args.index:
            index_dir = args.index
            print(f"📁 Indexando PDFs de: {index_dir}")
            success = isabel.index_pdf_directory(index_dir, force_reindex=args.force_reindex)
            
            if success:
                print("✅ Indexação concluída!")
            else:
                print("❌ Falha na indexação")
            return
        
        # Status do sistema
        if args.status:
            print("📊 Status do Sistema IAbel Local:")
            print("=" * 50)
            
            status = isabel.get_system_status()
            
            print(f"Vector Store:")
            print(f"  - Documentos: {status['vector_store']['total_documents']}")
            print(f"  - Localização: {status['vector_store']['persist_directory']}")
            
            print(f"\nEmbeddings:")
            print(f"  - Modelo: {status['embedder']['model']}")
            print(f"  - Dimensão: {status['embedder']['dimension']}")
            print(f"  - Device: {status['embedder']['device']}")
            
            print(f"\nLLM:")
            print(f"  - Modelo: {status['llm']['model']}")
            print(f"  - Status: {status['llm']['status']}")
            print(f"  - URL: {status['llm']['base_url']}")
            
            print(f"\nProcessador:")
            print(f"  - Chunk size: {status['processor']['chunk_size']}")
            print(f"  - Overlap: {status['processor']['chunk_overlap']}")
            
            return
        
        # Pergunta direta
        if args.query:
            print(f"❓ Pergunta: {args.query}")
            print("-" * 50)
            
            result = isabel.ask_question(args.query)
            
            print(f"\n🤖 Resposta:\n{result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Fontes ({result['confidence']:.1%} confiança):")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {source['source']} (p.{source['page']})")
            
            return
        
        # Chat interativo
        if args.chat:
            # Verifica se há documentos indexados
            status = isabel.get_system_status()
            if status['vector_store']['total_documents'] == 0:
                print("⚠️  Nenhum documento indexado encontrado.")
                print(f"   Use: python {sys.argv[0]} --setup")
                print(f"   Ou: python {sys.argv[0]} --index {pdf_directory}")
                return
            
            isabel.chat_session()
            return
    
    except KeyboardInterrupt:
        print("\n\n👋 Operação cancelada pelo usuário")
    
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


def check_requirements():
    """
    Verifica se dependências estão instaladas
    """
    required_packages = [
        ('sentence-transformers', 'sentence_transformers'),
        ('chromadb', 'chromadb'), 
        ('PyMuPDF', 'fitz'),
        ('requests', 'requests'),
        ('numpy', 'numpy')
    ]
    
    missing = []
    for pip_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print("❌ Pacotes em falta:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstale com: pip install " + " ".join(missing))
        return False
    
    return True


if __name__ == "__main__":
    print("🤖 IAbel Local - Sistema RAG para Engenharia de Reservatórios")
    print("=" * 60)
    
    # Verifica dependências
    if not check_requirements():
        sys.exit(1)
    
    # Executa aplicação principal
    main()