#!/bin/bash
echo "🚀 Iniciando IAbel Frontend..."
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend: http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "⏹️  Para parar: Ctrl+C"
echo "=========================================="

# Verificar se backend está rodando
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend conectado"
else
    echo "⚠️ Backend não detectado em http://localhost:8000"
    echo "   Certifique-se que o backend está rodando"
fi

echo ""
echo "🌐 Iniciando servidor de desenvolvimento..."

npm start