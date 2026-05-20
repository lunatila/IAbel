#!/usr/bin/env python3
"""
Script helper para garantir que o vectorstore esteja sempre disponível
"""

import os
import sys
from pathlib import Path

def setup_vectorstore():
    """
    Garante que o módulo vectorstore existe e é importável
    """
    project_root = Path(__file__).parent
    vectorstore_dir = project_root / "local_rag" / "vectorstore"
    
    # Cria diretório se não existir
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    # Cria __init__.py se não existir
    init_file = vectorstore_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write("# Vector Store Module\n")
    
    # Verifica se chroma_store.py existe
    chroma_file = vectorstore_dir / "chroma_store.py"
    if not chroma_file.exists():
        print("❌ Arquivo chroma_store.py não encontrado!")
        return False
    
    print("✅ Módulo vectorstore configurado corretamente")
    return True

if __name__ == "__main__":
    setup_vectorstore()