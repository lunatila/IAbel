# 🚀 Como Iniciar o IAbel com Google Gemini

## ✅ Tudo Configurado!

O backend já está configurado para usar **Google Gemini** automaticamente.

## 📋 Iniciar o Servidor

### Opção 1: Usando o script start_iabel.py

```bash
cd /home/lacucaratila/Projetos/IAbel/backend
python3 start_iabel.py
```

### Opção 2: Diretamente com Python

```bash
cd /home/lacucaratila/Projetos/IAbel/backend
source venv/bin/activate
python app/main.py
```

## 📡 Verificar se está usando Gemini

Quando o servidor iniciar, você deve ver:

```
🤖 LLM Provider: gemini
📦 Gemini Model: gemini-flash-latest
...
✅ Google Gemini configurado: gemini-flash-latest
   🤖 Using Google Gemini: gemini-flash-latest
```

## 🌐 Acessar

- **API**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs
- **Frontend**: http://localhost:3000 (após iniciar com `npm start`)

## 🎯 Testar

Abra http://localhost:8000/docs e teste o endpoint `/chat`:

```json
{
  "message": "O que é INSIM?",
  "mode": "rag_v2"
}
```

Você verá respostas **muito melhores** que antes! 🚀

## 🔧 Se der erro "Address already in use"

```bash
lsof -ti:8000 | xargs kill -9
```

Depois inicie novamente.

## 📊 Modelos Disponíveis

Você pode alterar o modelo no `.env`:

```env
GEMINI_MODEL=gemini-flash-latest  # Rápido e bom (recomendado)
# GEMINI_MODEL=gemini-2.5-flash   # Mais novo (pode ter quota)
# GEMINI_MODEL=gemini-1.5-pro     # Melhor qualidade (pago)
```

---

**Pronto!** O IAbel agora usa Google Gemini e está muito mais inteligente! 🤖✨
