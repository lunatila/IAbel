# 🤖 IAbel - Agente IA para Engenharia de Reservatórios

Sistema RAG (Retrieval-Augmented Generation) especializado em engenharia de reservatórios com suporte local e web.

## 🚀 Início Rápido

### Backend (Python + FastAPI)

```bash
cd backend
python3 start_iabel.py
```

O servidor iniciará em: http://localhost:8000
Documentação API: http://localhost:8000/docs

### Frontend (React + TypeScript)

```bash
cd frontend
npm start
```

A interface web iniciará em: http://localhost:3000

## 📋 Pré-requisitos

### Backend
- Python 3.12+
- Ambiente virtual (criado automaticamente)

### Frontend
- Node.js 16+
- npm ou yarn

### Opcional: LLM Local (Ollama)

Para usar o sistema completamente offline:

```bash
# Instalar Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Baixar modelo
ollama pull llama3.2:3b
```

## 🏗️ Estrutura do Projeto

```
IAbel/
├── backend/
│   ├── app/
│   │   ├── main.py              # Servidor FastAPI
│   │   ├── api/                 # Endpoints REST
│   │   └── services/            # Lógica de negócio
│   ├── data/
│   │   ├── pdfs/                # PDFs para indexação
│   │   └── vectorstore/         # Base vetorial
│   ├── venv/                    # Ambiente virtual Python
│   ├── requirements.txt         # Dependências Python
│   └── start_iabel.py           # Script de inicialização
│
├── frontend/
│   ├── src/
│   │   ├── components/          # Componentes React
│   │   ├── pages/               # Páginas da aplicação
│   │   └── utils/               # Utilitários
│   ├── package.json
│   └── README.md
│
├── local_rag/                   # Sistema RAG
│   ├── embeddings/              # Modelos de embedding
│   ├── vectorstore/             # ChromaDB/Qdrant
│   ├── processors/              # Processamento de PDFs
│   └── models/                  # Cliente LLM (Ollama)
│
└── data/                        # Dados do projeto
```

## ⚙️ Instalação Completa

### Primeira vez

```bash
# 1. Instalar dependências do backend
cd backend
python3 install_simple.py

# 2. Instalar dependências do frontend
cd ../frontend
npm install

# 3. Voltar para o root
cd ..
```

## 🎯 Funcionalidades

✅ **Chat Inteligente** - Interface conversacional com contexto
✅ **Upload de PDFs** - Adicionar novos documentos técnicos
✅ **Busca Semântica** - Recuperação precisa de informações
✅ **RAG Avançado** - Sistema de geração aumentada por recuperação
✅ **Modo Local** - Funcionamento offline com Ollama
✅ **Streaming** - Respostas em tempo real via WebSocket
✅ **Multilíngue** - Suporte para português e inglês

## 🔧 Tecnologias

### Backend
- **Framework**: FastAPI
- **Embeddings**: sentence-transformers (multilíngue)
- **Vector DB**: ChromaDB + Qdrant (fallback)
- **LLM**: Ollama (local) ou OpenAI/Anthropic (API)
- **PDF Processing**: PyMuPDF

### Frontend
- **Framework**: React + TypeScript
- **UI**: Tailwind CSS + shadcn/ui
- **Estado**: Zustand
- **Build**: Vite

## 📚 Documentação Adicional

- [Instalação Completa Backend](backend/INSTALACAO_COMPLETA.md)
- [Apresentação do Projeto](APRESENTACAO_PROJETO_IABEL.md)
- [Recomendações de Limpeza](CLEANUP_RECOMMENDATIONS.md)

## 🛠️ Desenvolvimento

### Testar Backend

```bash
cd backend
source venv/bin/activate
python -m pytest tests/
```

### Build Frontend para Produção

```bash
cd frontend
npm run build
```

## 🔍 Troubleshooting

### Backend não inicia

```bash
cd backend
python3 install_simple.py  # Reinstalar dependências
```

### Erro de porta já em uso

```bash
# Alterar porta no backend/app/main.py
# Ou matar processo na porta 8000
lsof -ti:8000 | xargs kill -9
```

### Frontend não conecta ao backend

Verifique se o backend está rodando em http://localhost:8000

## 📊 Requisitos de Sistema

**Mínimo**:
- 4GB RAM
- 5GB disco livre
- CPU dual-core

**Recomendado**:
- 8GB+ RAM
- 10GB+ disco livre
- CPU quad-core
- GPU (opcional, para embeddings mais rápidos)

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é de uso acadêmico/educacional.

## 🙏 Agradecimentos

Desenvolvido para auxiliar engenheiros de reservatórios com documentação técnica especializada.
