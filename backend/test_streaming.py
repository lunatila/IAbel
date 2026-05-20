#!/usr/bin/env python3
"""
Teste do sistema de streaming de resposta
"""

import requests
import json
import sys
import time

def test_streaming():
    print("🧪 Testando Streaming de Resposta...")
    
    url = "http://localhost:8000/chat/stream/"
    
    data = {
        "message": "o que é insim",
        "mode": "rag_v2",
        "top_k": 4,
        "include_sources": True
    }
    
    try:
        print(f"📤 Enviando pergunta: '{data['message']}'")
        print("⏳ Aguardando streaming...")
        print("-" * 50)
        
        response = requests.post(url, json=data, stream=True, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Erro HTTP: {response.status_code}")
            print(response.text)
            return False
        
        buffer = ""
        token_count = 0
        start_time = time.time()
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith('data: '):
                    try:
                        chunk_data = json.loads(line[6:])
                        chunk_type = chunk_data.get('type', 'unknown')
                        
                        if chunk_type == 'start':
                            print(f"🚀 {chunk_data.get('message', 'Iniciando...')}")
                            
                        elif chunk_type == 'search':
                            print(f"🔍 {chunk_data.get('message', 'Buscando...')}")
                            
                        elif chunk_type == 'search_complete':
                            sources = chunk_data.get('sources_count', 0)
                            print(f"📚 {chunk_data.get('message', 'Concluído')} ({sources} fontes)")
                            
                        elif chunk_type == 'compression':
                            print(f"🗜️ {chunk_data.get('message', 'Comprimindo...')}")
                            
                        elif chunk_type == 'generation_start':
                            print(f"🤖 {chunk_data.get('message', 'Gerando resposta...')}")
                            print("\n📝 Resposta em tempo real:")
                            print("-" * 30)
                            
                        elif chunk_type == 'token':
                            token = chunk_data.get('content', '')
                            print(token, end='', flush=True)
                            token_count += 1
                            
                        elif chunk_type == 'complete':
                            print("\n" + "-" * 30)
                            elapsed = time.time() - start_time
                            
                            confidence = chunk_data.get('confidence', 0)
                            sources_count = chunk_data.get('total_sources', 0)
                            
                            print(f"\n✅ Streaming completo!")
                            print(f"⏱️ Tempo total: {elapsed:.2f}s")
                            print(f"🔤 Tokens recebidos: {token_count}")
                            print(f"🎯 Confiança: {confidence:.3f}")
                            print(f"📚 Fontes: {sources_count}")
                            
                            if chunk_data.get('compression_stats'):
                                comp = chunk_data['compression_stats']
                                ratio = comp.get('compression_ratio', 0)
                                print(f"🗜️ Compressão: {ratio:.1%}")
                            
                            return True
                            
                        elif chunk_type == 'error':
                            print(f"\n❌ Erro: {chunk_data.get('message', 'Erro desconhecido')}")
                            return False
                            
                        elif chunk_type == 'end':
                            print("\n🏁 Stream finalizado")
                            return True
                            
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️ Erro ao decodificar JSON: {e}")
                        continue
                        
    except requests.exceptions.Timeout:
        print("⏰ Timeout na requisição")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Erro de conexão - verifique se o backend está rodando")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

if __name__ == "__main__":
    print("🔧 TESTE DE STREAMING - IAbel")
    print("=" * 40)
    
    success = test_streaming()
    
    if success:
        print("\n🎉 STREAMING FUNCIONANDO PERFEITAMENTE!")
        print("💡 Agora teste no frontend e veja os tokens aparecendo em tempo real")
    else:
        print("\n💥 STREAMING COM PROBLEMAS")
        print("💡 Verifique se o backend está rodando e tente novamente")
    
    sys.exit(0 if success else 1)