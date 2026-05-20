#!/usr/bin/env python3
"""
Testar resultados da reindexação melhorada
"""

import os
import sys
from pathlib import Path

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_reindex_results():
    print("🧪 Testando resultados da reindexação melhorada...")
    
    try:
        from app.services.rag_service import get_rag_service
        
        # Get RAG service
        rag_service = get_rag_service()
        
        # Get system statistics
        stats = rag_service.rag_system.get_system_status()
        total_docs = stats.get('vector_store', {}).get('total_documents', 0)
        chunk_size = stats.get('processor', {}).get('chunk_size', 'N/A')
        
        print(f"📊 STATUS APÓS REINDEXAÇÃO:")
        print(f"   📚 Total documentos: {total_docs}")
        print(f"   📏 Chunk size: {chunk_size}")
        print(f"   🧠 Embedder: {stats.get('embedder', {}).get('model', 'N/A')}")
        
        # Compare with previous
        previous_docs = 17363
        if total_docs != previous_docs:
            if total_docs < previous_docs:
                reduction = ((previous_docs - total_docs) / previous_docs) * 100
                print(f"   📉 Redução: {reduction:.1f}% ({previous_docs} → {total_docs})")
                print(f"   ✅ Menos chunks = melhor qualidade!")
            else:
                increase = ((total_docs - previous_docs) / previous_docs) * 100
                print(f"   📈 Aumento: {increase:.1f}% ({previous_docs} → {total_docs})")
                
                if total_docs > 20000:
                    print(f"   ⚠️ Muitos chunks podem prejudicar qualidade")
                else:
                    print(f"   🟡 Aumento moderado, chunk size deve ter compensado")
        else:
            print(f"   🔄 Mesmo número de documentos")
        
        # Test search quality
        test_query = "o que é insim"
        print(f"\n🔍 Testando qualidade com: '{test_query}'")
        
        # Test basic search
        search_results = rag_service.rag_system.search_documents(
            query=test_query,
            top_k=5,
            similarity_threshold=0.4
        )
        
        if search_results:
            avg_similarity = sum(r.get('similarity', 0) for r in search_results[:3]) / min(3, len(search_results))
            print(f"   📊 Similaridade média: {avg_similarity:.3f}")
            print(f"   🔢 Documentos encontrados: {len(search_results)}")
            
            print(f"   📚 Top 3:")
            for i, result in enumerate(search_results[:3]):
                sim = result.get('similarity', 0)
                source = result.get('metadata', {}).get('source', 'Unknown')
                preview = result.get('content', '')[:80] + "..."
                print(f"      {i+1}. {sim:.3f} - {source}")
                print(f"         {preview}")
            
            # Quality assessment
            if avg_similarity > 0.9:
                print(f"   ✅ EXCELENTE qualidade!")
            elif avg_similarity > 0.7:
                print(f"   🟢 BOA qualidade")
            elif avg_similarity > 0.5:
                print(f"   🟡 REGULAR qualidade")
            else:
                print(f"   🔴 BAIXA qualidade - precisa ajustes")
        else:
            print(f"   ❌ Nenhum resultado encontrado")
        
        # Test enhanced search
        print(f"\n🚀 Testando RAG v2 Enhanced:")
        
        import asyncio
        
        async def test_enhanced():
            enhanced_results = await rag_service.rag_system.enhanced_search_documents(
                query=test_query,
                top_k=4,
                similarity_threshold=0.4,
                use_fusion=True,
                use_reranking=False  # Mantém desabilitado
            )
            return enhanced_results
        
        enhanced_results = asyncio.run(test_enhanced())
        
        if enhanced_results:
            avg_enhanced = sum(r.get('similarity', 0) for r in enhanced_results[:3]) / min(3, len(enhanced_results))
            print(f"   📊 Enhanced similaridade: {avg_enhanced:.3f}")
            print(f"   🔢 Enhanced resultados: {len(enhanced_results)}")
            
            # Compare with basic
            if search_results:
                if avg_enhanced > avg_similarity:
                    improvement = ((avg_enhanced - avg_similarity) / avg_similarity) * 100
                    print(f"   ✅ Enhanced {improvement:.1f}% melhor que básico")
                elif avg_enhanced < avg_similarity:
                    degradation = ((avg_similarity - avg_enhanced) / avg_similarity) * 100
                    print(f"   ⚠️ Enhanced {degradation:.1f}% pior que básico")
                else:
                    print(f"   🟡 Enhanced e básico com qualidade similar")
        
        # Final assessment
        print(f"\n📋 AVALIAÇÃO FINAL:")
        
        if total_docs < 15000:
            print(f"   ✅ Número de chunks otimizado")
        elif total_docs < 20000:
            print(f"   🟡 Número de chunks aceitável")
        else:
            print(f"   ⚠️ Muitos chunks - pode precisar de mais otimização")
        
        if search_results and avg_similarity > 0.8:
            print(f"   ✅ Qualidade de busca excelente")
        elif search_results and avg_similarity > 0.6:
            print(f"   🟡 Qualidade de busca boa")
        else:
            print(f"   ⚠️ Qualidade de busca precisa melhorar")
        
        print(f"\n💡 PRÓXIMOS PASSOS:")
        print(f"   1. Reiniciar backend para garantir carregamento completo")
        print(f"   2. Testar ambas versões RAG no site")
        print(f"   3. Comparar com backup se necessário")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_reindex_results()