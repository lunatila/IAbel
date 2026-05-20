# 🚀 Como Configurar Google Gemini no IAbel

## 📋 Passo a Passo

### 1. Obter API Key do Google

1. Acesse: https://aistudio.google.com/app/apikey
2. Faça login com sua conta Google
3. Clique em "Create API Key"
4. Copie a chave gerada

### 2. Configurar Variáveis de Ambiente

Edite o arquivo `.env` no diretório `backend/`:

```bash
cd backend
nano .env
```

Adicione (ou edite):

```env
# Google Gemini Configuration
GOOGLE_API_KEY=sua_chave_api_aqui
GEMINI_MODEL=gemini-2.0-flash-exp

# Ou use gemini-1.5-pro para modelo mais robusto:
# GEMINI_MODEL=gemini-1.5-pro
```

**Salve o arquivo** (Ctrl+O, Enter, Ctrl+X no nano)

### 3. Testar Gemini

Execute o teste rápido:

```bash
cd /home/lacucaratila/Projetos/IAbel
source backend/venv/bin/activate

python -c "
import sys
import os
sys.path.insert(0, '.')

# Configure sua API key aqui para teste
os.environ['GOOGLE_API_KEY'] = 'SUA_CHAVE_AQUI'

from local_rag.models.gemini_client import GeminiClient

# Teste conexão
client = GeminiClient(model_name='gemini-2.0-flash-exp')
client.test_connection()
"
```

### 4. Testar RAG com Gemini

```bash
source backend/venv/bin/activate

python -c "
import sys
import os
sys.path.insert(0, '.')

# Configure sua API key
os.environ['GOOGLE_API_KEY'] = 'SUA_CHAVE_AQUI'

from local_rag.rag_system import LocalRAGSystem

# Cria sistema RAG com Gemini
rag = LocalRAGSystem(
    vectorstore_path='backend/data/vectorstore',
    llm_provider='gemini',
    llm_model='gemini-2.0-flash-exp'
)

# Testa pergunta
result = rag.ask_question('O que é INSIM?', top_k=3)
print('Resposta:')
print(result['answer'])
"
```

### 5. Iniciar Backend com Gemini

O backend precisa ser configurado para usar Gemini. Edite o arquivo que inicializa o RAG service.

## 🎯 Modelos Disponíveis

| Modelo | Velocidade | Qualidade | Custo | Recomendado |
|--------|-----------|-----------|-------|-------------|
| `gemini-2.0-flash-exp` | ⚡⚡⚡ Muito Rápido | ⭐⭐⭐⭐ Ótima | 💰 Grátis* | ✅ Sim |
| `gemini-1.5-flash` | ⚡⚡⚡ Muito Rápido | ⭐⭐⭐ Boa | 💰 Grátis* | ✅ Sim |
| `gemini-1.5-pro` | ⚡⚡ Rápido | ⭐⭐⭐⭐⭐ Excelente | 💰💰 Pago | Para prod |

\* Grátis com limites de uso (veja: https://ai.google.dev/pricing)

## 💡 Dicas

### Gemini vs Ollama Local

**Gemini (API):**
- ✅ Respostas muito melhores
- ✅ Mais rápido que modelos locais pequenos
- ✅ Não usa recursos da máquina
- ❌ Requer internet
- ❌ Limite de uso gratuito

**Ollama Local:**
- ✅ Completamente offline
- ✅ Sem limites de uso
- ✅ Privacidade total
- ❌ Respostas menos precisas (modelos pequenos)
- ❌ Usa GPU/CPU local

### Para Melhor Qualidade

Use `gemini-1.5-pro` se precisar das melhores respostas e estiver disposto a pagar um pouco.

### Para Desenvolvimento/Testes

Use `gemini-2.0-flash-exp` - é grátis e tem ótima qualidade.

## 🔧 Troubleshooting

### Erro: "API Key não encontrada"

Certifique-se que a variável `GOOGLE_API_KEY` está configurada no `.env`

### Erro: "API Key inválida"

Verifique se copiou a chave completa do Google AI Studio.

### Erro: "Quota exceeded"

Você atingiu o limite gratuito. Espere 1 minuto ou considere upgrade.

## 📚 Mais Informações

- Google AI Studio: https://aistudio.google.com
- Documentação Gemini: https://ai.google.dev/docs
- Preços: https://ai.google.dev/pricing
