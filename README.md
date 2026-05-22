<div align="center">
  <img src="frontend/src/images/IAbel_transp.png" alt="IAbel Logo" width="120" />

  # IAbel — Agente IA para Engenharia de Reservatórios

  **RAG system especializado em engenharia de reservatórios de petróleo**

  [![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.108-009688?logo=fastapi)](https://fastapi.tiangolo.com)
  [![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.2-3178C6?logo=typescript)](https://typescriptlang.org)
  [![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)](https://www.trychroma.com)
  [![License](https://img.shields.io/badge/Licença-Acadêmica-green)](LICENSE)

</div>

---

## Sobre o Projeto

O **IAbel** é um assistente de IA especializado em engenharia de reservatórios de petróleo. Ele usa um pipeline de **RAG (Retrieval-Augmented Generation)** para responder perguntas técnicas com base em documentos indexados — manuais, artigos acadêmicos e dissertações da área.

O sistema funciona **100% local** (sem enviar dados para a nuvem) e suporta consultas em **português e inglês**.

---

## Screenshots

### Tela Inicial

Interface estilo ChatGPT com seletor de versão RAG e indicadores de status em tempo real.

![Tela Inicial](docs/screenshots/welcome.png)

### Conversa com Respostas e Citações

Resposta com **95% de confiança**, contexto extraído dos documentos indexados e referências acadêmicas automáticas.

![Chat em andamento](docs/screenshots/chat.png)

### Dashboard de Status do Sistema

**1792 documentos indexados** · LLM Online (Gemini Flash) · 10 capacidades ativas simultaneamente.

![Status do Sistema](docs/screenshots/status.png)

---

## Funcionalidades

| Funcionalidade | Descrição |
|---|---|
| **Chat RAG Multilíngue** | Respostas em PT/BR e EN com base nos documentos indexados |
| **3 Versões de RAG** | v1 Básico · v2 Avançado com RAG Fusion · v3 English + Citações Acadêmicas |
| **Streaming em tempo real** | Respostas aparecem token a token via WebSocket |
| **Upload de PDFs** | Adicione novos documentos e eles são indexados automaticamente |
| **Citações acadêmicas** | O modo v3 rastreia e formata as fontes consultadas |
| **Memória de conversa** | O histórico da sessão é mantido para contexto contínuo |
| **Fine-tuning LoRA** | Pipeline para ajuste fino do LLM com dados do domínio |
| **Dashboard de monitoramento** | Visualização em tempo real de LLM, vectorstore, cache e erros |

---

## Arquitetura

```
┌─────────────────────────────────────────────────┐
│              Frontend  (React + TypeScript)      │
│  Chat Interface · Upload · System Status         │
│  WebSocket streaming · RAG v1 / v2 / v3          │
└─────────────────────┬───────────────────────────┘
                      │ HTTP / WebSocket
┌─────────────────────▼───────────────────────────┐
│              Backend  (FastAPI)                  │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │            RAG Pipeline                 │    │
│  │  1. Query expansion (5 variações)       │    │
│  │  2. Busca vetorial (ChromaDB)           │    │
│  │  3. RAG Fusion + reranking              │    │
│  │  4. Compressão de contexto              │    │
│  │  5. Geração (Gemini / Ollama)           │    │
│  │  6. Rastreamento de citações            │    │
│  └─────────────────────────────────────────┘    │
└────────┬────────────────────────┬───────────────┘
         │                        │
┌────────▼──────────┐   ┌─────────▼──────────────┐
│  ChromaDB         │   │  Google Gemini / Ollama  │
│  Embeddings 768D  │   │  LLM para geração       │
│  HNSW indexing    │   │                         │
└───────────────────┘   └─────────────────────────┘
         │
┌────────▼──────────┐
│  Redis Cache      │
│  TTL + embedding  │
└───────────────────┘
```

---

## Stack Tecnológica

### Backend
- **[FastAPI](https://fastapi.tiangolo.com)** — API REST + WebSocket
- **[ChromaDB](https://www.trychroma.com)** — banco vetorial com HNSW
- **[Sentence Transformers](https://www.sbert.net)** — embeddings multilíngues 768D (`paraphrase-multilingual-mpnet-base-v2`)
- **[Google Gemini](https://ai.google.dev)** — LLM principal (ou Ollama para uso offline)
- **[Redis](https://redis.io)** — cache multi-camada
- **[LangChain](https://langchain.com)** — processamento de documentos
- **[PEFT / LoRA](https://huggingface.co/docs/peft)** — fine-tuning eficiente do LLM

### Frontend
- **[React 18](https://react.dev)** + **[TypeScript](https://typescriptlang.org)**
- **[Vite](https://vitejs.dev)** — build tool
- **[TailwindCSS](https://tailwindcss.com)** — estilização
- **[Framer Motion](https://www.framer.motion.com)** — animações
- **[Zustand](https://zustand-demo.pmnd.rs)** — gerenciamento de estado
- **[Axios](https://axios-http.com)** — cliente HTTP

---

## Pré-requisitos

| Requisito | Versão mínima |
|---|---|
| Python | 3.12+ |
| Node.js | 18+ |
| Redis | 7+ (opcional, usa cache em memória como fallback) |
| Google Gemini API Key | — |

**Requisitos de hardware:**
- RAM: 4 GB mínimo (8 GB recomendado)
- Disco: ~3 GB para modelos de embedding
- GPU: opcional (acelera embeddings)

---

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/lunatila/IAbel.git
cd IAbel
```

### 2. Configure as variáveis de ambiente

```bash
cp backend/.env.example backend/.env
```

Edite `backend/.env` e adicione sua chave da API:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-flash-latest
```

### 3. Backend (Python)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# ou: venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 4. Frontend (Node.js)

```bash
cd frontend
npm install
```

---

## Executando o Projeto

### Linux / macOS / WSL

**Terminal 1 — Backend:**
```bash
cd backend
source venv/bin/activate
python app/main.py
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

### Windows (nativo)

**Terminal 1 — Backend (via WSL):**
```bash
wsl bash -c "cd /caminho/para/IAbel/backend && python3.12 run_windows.py"
```

**Terminal 2 — Frontend:**
```powershell
cd frontend
npm run dev
```

### Acessar

| Serviço | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger (docs) | http://localhost:8000/docs |

---

## Adicionando Documentos

Coloque seus PDFs na pasta `backend/data/pdfs/` e use a interface de upload ou chame o endpoint de reindexação:

```bash
curl -X POST http://localhost:8000/reindex/
```

Ou faça upload direto pela interface web em http://localhost:3000.

---

## Modos de RAG

O IAbel oferece três versões do pipeline RAG, selecionáveis na interface:

| Modo | Descrição | Melhor para |
|---|---|---|
| **v1** | RAG básico com busca semântica | Perguntas simples e diretas |
| **v2** | RAG Fusion + compressão de contexto + feedback learning | Perguntas complexas em português |
| **v3** | English + citações acadêmicas formatadas | Uso acadêmico com referências |

---

## Estrutura do Projeto

```
IAbel/
├── backend/
│   ├── app/
│   │   ├── main.py               # Servidor FastAPI + endpoints
│   │   ├── services/
│   │   │   ├── rag_service.py    # Orquestrador do pipeline RAG
│   │   │   └── cache_service.py  # Serviço de cache Redis
│   │   └── utils/
│   │       └── logging_config.py
│   ├── fine_tuning/              # Pipeline de fine-tuning LoRA
│   ├── data/
│   │   └── pdfs/                 # Adicione seus PDFs aqui
│   ├── requirements.txt
│   └── run_windows.py            # Script de inicialização no Windows
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── chat/             # ChatInterface, ChatMessage, WelcomeScreen
│       │   ├── dashboard/        # SystemStatus
│       │   ├── upload/           # PDFUpload
│       │   └── ui/               # Componentes base
│       ├── services/api.ts       # Cliente da API
│       └── stores/appStore.ts    # Estado global (Zustand)
│
├── local_rag/                    # Módulo RAG avançado
│   ├── embeddings/               # Modelos de embedding locais
│   ├── vectorstore/              # Interface ChromaDB
│   ├── fusion/                   # RAG Fusion
│   ├── citations/                # Rastreamento de citações
│   ├── memory/                   # Memória de conversa
│   └── models/                   # Clientes Gemini e Ollama
│
├── scripts/                      # Scripts de indexação e testes
├── docker-compose.yml
└── README.md
```

---

## API Reference

### Endpoints principais

```
POST /chat/             — Enviar mensagem (RAG padrão)
POST /chat/stream/      — Streaming de resposta (SSE)
WS   /chat/stream       — WebSocket para streaming
POST /upload-pdf/       — Upload de novo documento
POST /reindex/          — Reindexar todos os PDFs
GET  /status/           — Status do sistema
GET  /health/           — Health check
POST /feedback/         — Submeter feedback de resposta
```

Documentação interativa completa em http://localhost:8000/docs

---

## Fine-tuning LoRA (opcional)

O projeto inclui um pipeline de fine-tuning com LoRA para adaptar o LLM ao domínio de reservatórios:

```bash
cd backend/fine_tuning
python train_lora_light.py    # versão leve, recomendada para começar
```

Modelos treinados ficam em `backend/fine_tuning/outputs/`.

---

## Usando com Docker

```bash
docker compose up -d
```

Serviços iniciados: backend, frontend, Redis, ChromaDB.

---

## Troubleshooting

**Backend demora para iniciar**
> Normal — os modelos de embedding (~430 MB) são carregados na memória na primeira execução. Aguarde ~1–5 minutos.

**`ModuleNotFoundError` ao iniciar**
> Verifique se o ambiente virtual está ativado e execute `pip install -r requirements.txt`.

**Frontend não conecta ao backend**
> Confirme que o backend está rodando em http://localhost:8000. Verifique o CORS em `backend/app/main.py`.

**Porta 8000 já em uso**
```bash
# Linux/macOS
lsof -ti:8000 | xargs kill -9

# Windows PowerShell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

---

## Contribuindo

1. Fork o projeto
2. Crie uma branch: `git checkout -b feature/minha-feature`
3. Commit: `git commit -m 'feat: adiciona minha feature'`
4. Push: `git push origin feature/minha-feature`
5. Abra um Pull Request

---

## Licença

Projeto de uso acadêmico/educacional. Desenvolvido como parte de pesquisa de doutorado.

---

<div align="center">
  Desenvolvido para auxiliar engenheiros de reservatórios com documentação técnica especializada.<br/>
  <img src="frontend/src/images/IAbel_cigarro.png" alt="IAbel" width="48" />
</div>
