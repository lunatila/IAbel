# IAbel Local - Sistema RAG Offline

Sistema RAG completamente local para engenharia de reservatórios, sem necessidade de APIs externas.

## 🚀 Instalação Rápida

### 1. Instalar Ollama (LLM local)
```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: baixe de https://ollama.com/
```

### 2. Instalar dependências Python
```bash
pip install -r requirements_local.txt
```

### 3. Baixar modelo LLM
```bash
ollama pull llama3.2:3b  # Modelo leve (2GB)
# ou
ollama pull llama3.2:7b  # Modelo maior (4GB)
```

### 4. Setup inicial
```bash
python isabel_local.py --setup
```

## 📁 Estrutura do Projeto

```
IAbel/
├── isabel_local.py              # Interface principal
├── requirements_local.txt       # Dependências
├── local_rag/                   # Sistema RAG local
│   ├── rag_system.py           # Sistema completo integrado
│   ├── processors/             # Processamento de PDFs
│   │   └── pdf_processor.py
│   ├── embeddings/             # Embeddings locais
│   │   └── local_embedder.py
│   ├── vectorstore/            # Base vetorial local
│   │   └── chroma_store.py
│   ├── models/                 # Cliente LLM local
│   │   └── ollama_client.py
│   └── vectorstore/            # Dados persistidos
└── backend/data/pdfs/          # PDFs para indexação
```

## 🎯 Uso

### Chat Interativo
```bash
python isabel_local.py --chat
```

### Pergunta Direta
```bash
python isabel_local.py --query "O que é permeabilidade?"
```

### Indexar Novos PDFs
```bash
python isabel_local.py --index /caminho/para/pdfs
```

### Status do Sistema
```bash
python isabel_local.py --status
```

## 🔧 Componentes Locais

### 1. **Embeddings**: sentence-transformers
- Modelo: `all-MiniLM-L6-v2` (22MB)
- Execução: CPU/GPU local
- Cache automático

### 2. **Vector Store**: ChromaDB
- Persistência local
- Busca por similaridade cosine
- Metadados estruturados

### 3. **LLM**: Ollama
- Modelos suportados: Llama, Mistral, CodeLlama
- Execução local (CPU/GPU)
- API REST local

### 4. **PDF Processing**: PyMuPDF
- Extração de texto
- Chunking inteligente
- Metadados preservados

## 📊 Especificações Técnicas

### Modelos Recomendados
- **llama3.2:3b** - Rápido, 2GB RAM (recomendado)
- **llama3.2:7b** - Equilibrado, 4GB RAM
- **mistral:7b** - Alternativa, 4GB RAM

### Requisitos de Sistema
- **Mínimo**: 4GB RAM, 2GB disco
- **Recomendado**: 8GB RAM, 5GB disco
- **Optimal**: 16GB RAM, GPU, 10GB disco

## 🛠️ Configurações Avançadas

### Personalizar Modelos
```python
# Embeddings
isabel = LocalRAGSystem(
    embedder_model="sentence-transformers/all-mpnet-base-v2"
)

# LLM
isabel = LocalRAGSystem(
    llm_model="mistral:7b"
)
```

### Ajustar Chunking
```python
isabel = LocalRAGSystem(
    chunk_size=1500,    # Chunks maiores
    chunk_overlap=300   # Mais sobreposição
)
```

## 🔍 Troubleshooting

### Ollama não conecta
```bash
# Verificar se está rodando
ollama list

# Iniciar servidor
ollama serve

# Verificar porta
curl http://localhost:11434/api/version
```

### Embeddings muito lentos
```bash
# Instalar com GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erro de memória
- Use modelo menor: `llama3.2:3b`
- Reduza batch_size nos embeddings
- Limite chunk_size para 800

## 📈 Performance

### Benchmarks (llama3.2:3b)
- **Indexação**: ~50 docs/min
- **Busca**: <1s por query
- **Resposta**: 2-5s dependendo do contexto
- **Memória**: ~2GB base + 1GB por 1000 docs

## 🔒 Privacidade

✅ **Completamente offline**
✅ **Dados permanecem locais**
✅ **Sem telemetria**
✅ **Sem APIs externas**

## 📚 Exemplos de Uso

### Python API
```python
from local_rag.rag_system import LocalRAGSystem

isabel = LocalRAGSystem()
result = isabel.ask_question("Como calcular fator de recuperação?")
print(result['answer'])
```

### Processamento de PDFs
```python
from local_rag.processors.pdf_processor import PDFProcessor

processor = PDFProcessor()
chunks = processor.process_pdf("manual_simulador.pdf")
```

## 🤝 Contribuição

Sistema modular - fácil de estender:
- Novos processadores de documento
- Diferentes modelos de embedding
- Outros LLMs locais
- Interfaces customizadas