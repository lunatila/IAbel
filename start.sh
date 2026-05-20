#!/bin/bash

echo "🚀 Iniciando IAbel - Agente IA para Engenharia de Reservatórios"
echo "=================================================="

# Verificar se o arquivo .env existe no backend
if [ ! -f "backend/.env" ]; then
    echo "⚠️  Arquivo .env não encontrado no backend!"
    echo "📝 Copiando .env.example para .env..."
    cp backend/.env.example backend/.env
    echo "🔑 Por favor, edite o arquivo backend/.env com suas chaves de API antes de continuar."
    echo "   Especialmente: OPENAI_API_KEY"
    echo ""
    read -p "Pressione Enter quando terminar de configurar o .env..."
fi

# Criar diretórios necessários
mkdir -p backend/data/{pdfs,vectorstore}

echo "🐳 Iniciando containers com Docker Compose..."
docker-compose up --build

echo "✅ IAbel está pronto!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 Documentação da API: http://localhost:8000/docs"