#!/usr/bin/env python3
"""
Teste do sistema de memória conversacional
Demonstra como o sistema mantém contexto entre perguntas
"""

import os
import sys
from pathlib import Path

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_conversation_memory():
    print("🧪 Testando Sistema de Memória Conversacional")
    print("=" * 50)
    
    try:
        from app.services.rag_service import get_rag_service
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Generate conversation ID
        import uuid
        conversation_id = str(uuid.uuid4())
        print(f"💬 Conversation ID: {conversation_id}")
        
        # Test scenario: User asks for a list, then refers to items
        test_scenarios = [
            {
                "question": "Liste 3 conceitos importantes em engenharia de reservatórios",
                "description": "Pergunta inicial que deve gerar uma lista"
            },
            {
                "question": "Explique melhor o item 1",
                "description": "Referência ao item 1 da lista anterior"
            },
            {
                "question": "O que você mencionou sobre isso na resposta anterior?",
                "description": "Referência genérica ao contexto anterior"
            },
            {
                "question": "Compare o primeiro com o segundo item da lista",
                "description": "Referência a múltiplos itens da lista original"
            }
        ]
        
        print(f"\n🎯 Executando {len(test_scenarios)} cenários de teste...\n")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"📝 CENÁRIO {i}: {scenario['description']}")
            print(f"❓ Pergunta: {scenario['question']}")
            
            # Ask question with conversation memory
            response = await rag_service.ask_question(
                question=scenario['question'],
                conversation_id=conversation_id,
                top_k=4,
                include_sources=True
            )
            
            print(f"💡 Resposta:")
            answer = response.get('answer', 'Sem resposta')
            # Limit response length for readability
            if len(answer) > 300:
                answer = answer[:300] + "..."
            print(f"   {answer}")
            
            print(f"🎯 Confiança: {response.get('confidence', 0):.3f}")
            print(f"📚 Fontes: {response.get('total_sources', 0)}")
            
            # Check if enhanced features were used
            if response.get('enhanced'):
                print(f"✅ Enhanced RAG ativo")
                if response.get('compression_stats'):
                    comp_ratio = response['compression_stats'].get('compression_ratio', 0)
                    print(f"🗜️ Compressão: {comp_ratio:.1%}")
            
            print(f"─" * 60)
            print()
        
        # Test memory inspection
        print("🔍 INSPEÇÃO DA MEMÓRIA CONVERSACIONAL:")
        
        try:
            memory = rag_service.rag_system.conversation_memory
            if memory and conversation_id in memory.active_conversations:
                context = memory.active_conversations[conversation_id]
                
                print(f"📊 Estatísticas da conversa:")
                print(f"   🔢 Total de interações: {len(context.turns)}")
                print(f"   ⏰ Criada em: {context.created_at.strftime('%H:%M:%S')}")
                print(f"   🕐 Última atividade: {context.last_activity.strftime('%H:%M:%S')}")
                
                # Show extracted lists
                lists = context.extract_list_items()
                if lists:
                    print(f"   📋 Listas detectadas: {len(lists)}")
                    for list_name, items in lists.items():
                        print(f"      • {list_name}: {len(items)} itens")
                        for j, item in enumerate(items[:3], 1):  # Show first 3 items
                            print(f"        {j}. {item[:60]}...")
                
                print(f"   🧠 Entidades mencionadas:")
                all_entities = set()
                for turn in context.turns:
                    all_entities.update(turn.entities_mentioned)
                
                for entity in list(all_entities)[:5]:  # Show first 5 entities
                    print(f"      • {entity}")
                
            else:
                print("⚠️ Conversa não encontrada na memória")
        
        except Exception as mem_error:
            print(f"❌ Erro ao inspecionar memória: {mem_error}")
        
        print(f"\n🎉 TESTE CONCLUÍDO!")
        print(f"💡 Agora teste no site:")
        print(f"   1. Pergunte: 'Liste 3 métodos de recuperação de petróleo'")
        print(f"   2. Depois: 'Explique melhor o item 1'")
        print(f"   3. Observe se o sistema mantém contexto!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(test_conversation_memory())
    sys.exit(0 if success else 1)