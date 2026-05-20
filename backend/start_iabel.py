#!/usr/bin/env python3
"""
Script para iniciar o servidor IAbel com ambiente virtual ativo
Resolve problemas de importação e ambiente automaticamente
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Iniciando IAbel Backend Server...")
    
    # Verificar se estamos no diretório correto
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Caminho para o Python do ambiente virtual
    venv_python = "./venv/bin/python"
    
    if not os.path.exists(venv_python):
        print("❌ Ambiente virtual não encontrado!")
        print("Execute primeiro: python3 install_simple.py")
        return False
    
    # Verificar se o arquivo main.py existe
    app_main = "./app/main.py"
    if not os.path.exists(app_main):
        print(f"❌ Arquivo {app_main} não encontrado!")
        return False
    
    # Testar importações críticas
    print("🧪 Testando dependências...")
    test_cmd = [venv_python, "-c", """
import sys
sys.path.insert(0, '../')  # Add project root for local_rag

# Test critical imports
modules_to_test = [
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('numpy', 'NumPy'),
    ('local_rag.rag_system', 'RAG System')
]

failed = []
for module, name in modules_to_test:
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
        failed.append(name)

if failed:
    print(f'FAILED: {failed}')
    sys.exit(1)
else:
    print('ALL TESTS PASSED')
"""]
    
    try:
        result = subprocess.run(test_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Teste de dependências falhou:")
        print(e.stdout)
        print(e.stderr)
        print("\nTente executar: python3 install_simple.py")
        return False
    
    # Iniciar servidor
    print("\n" + "="*50)
    print("🌐 Iniciando servidor FastAPI...")
    print("📍 URL: http://localhost:8000")
    print("📍 Docs: http://localhost:8000/docs")
    print("⏹️  Para parar: Ctrl+C")
    print("="*50)
    
    try:
        # Adicionar project root ao PYTHONPATH para garantir que local_rag seja encontrado
        env = os.environ.copy()
        project_root = str(Path.cwd().parent)
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = project_root
        
        # Executar o servidor
        subprocess.run([venv_python, app_main], check=True, env=env)
        
    except KeyboardInterrupt:
        print("\n⏹️  Servidor parado pelo usuário")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro ao iniciar servidor:")
        print(f"Código de saída: {e.returncode}")
        if e.stdout:
            print(f"Saída: {e.stdout}")
        if e.stderr:
            print(f"Erro: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)