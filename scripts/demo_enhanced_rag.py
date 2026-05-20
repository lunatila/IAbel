#!/usr/bin/env python3
"""
Demo Enhanced RAG System
Demonstração das capacidades avançadas do sistema RAG melhorado
"""

import asyncio
import sys
import os
from pathlib import Path
import time
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    """Print demo banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🚀 IAbel Enhanced RAG Demo                            ║
║                                                                              ║
║  Demonstração das melhorias implementadas no sistema RAG:                   ║
║  • 🧠 Semantic Chunking    • 🔀 RAG Fusion    • 🎯 Cross-Encoder           ║
║  • ⚡ Redis Cache          • 🔍 Self-Critique  • 🧩 Context Enhancement     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_section(title: str, emoji: str = "🔧"):
    """Print section header"""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))

async def demo_basic_vs_enhanced():
    """Demonstrate difference between basic and enhanced RAG"""
    print_section("Comparação: Sistema Básico vs Enhanced", "⚖️")
    
    try:
        # Import both systems
        from local_rag.rag_system import LocalRAGSystem
        from local_rag.enhanced_rag_system import EnhancedRAGSystem
        
        # Check if PDFs exist
        pdf_dir = project_root / "backend" / "data" / "pdfs"
        if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
            print("⚠️  Nenhum PDF encontrado em backend/data/pdfs/")
            print("   Adicione alguns PDFs técnicos para a demonstração completa")
            return
        
        print("📚 Inicializando sistemas...")
        
        # Initialize basic system
        print("   🔧 Sistema básico...")
        basic_rag = LocalRAGSystem(
            vectorstore_path="./vectorstore_basic_demo"
        )
        
        # Initialize enhanced system
        print("   ✨ Sistema enhanced...")
        enhanced_rag = EnhancedRAGSystem(
            vectorstore_path="./vectorstore_enhanced_demo",
            enable_semantic_chunking=True,
            enable_rag_fusion=True,
            enable_reranking=True
        )
        
        # Demo query
        demo_query = "O que é INSIM e como funciona a simulação?"
        
        print(f"\n🔍 Query de teste: '{demo_query}'")
        
        # Basic system search
        print("\n📊 Resultado do Sistema Básico:")
        start_time = time.time()
        basic_results = basic_rag.search_documents(demo_query, top_k=3)
        basic_time = time.time() - start_time
        
        print(f"   ⏱️  Tempo: {basic_time:.2f}s")
        print(f"   📄 Resultados: {len(basic_results)}")
        if basic_results:
            avg_similarity = sum(r.get('similarity', 0) for r in basic_results) / len(basic_results)
            print(f"   🎯 Similaridade média: {avg_similarity:.3f}")
        
        # Enhanced system search
        print("\n✨ Resultado do Sistema Enhanced:")
        start_time = time.time()
        enhanced_results = await enhanced_rag.enhanced_search_documents(
            demo_query, 
            top_k=3,
            use_fusion=True,
            use_reranking=True
        )
        enhanced_time = time.time() - start_time
        
        print(f"   ⏱️  Tempo: {enhanced_time:.2f}s")
        print(f"   📄 Resultados: {len(enhanced_results)}")
        if enhanced_results:
            avg_similarity = sum(r.get('similarity', 0) for r in enhanced_results) / len(enhanced_results)
            print(f"   🎯 Similaridade média: {avg_similarity:.3f}")
            
            # Show enhanced features
            fusion_count = sum(1 for r in enhanced_results if 'fusion_score' in r)
            rerank_count = sum(1 for r in enhanced_results if 'cross_encoder_score' in r)
            
            print(f"   🔀 Com RAG Fusion: {fusion_count}/{len(enhanced_results)}")
            print(f"   🎯 Com Re-ranking: {rerank_count}/{len(enhanced_results)}")
        
        # Performance comparison
        if basic_time > 0:
            speedup = (basic_time - enhanced_time) / basic_time * 100
            print(f"\n📈 Performance: {speedup:+.1f}% {'mais rápido' if speedup > 0 else 'mais lento'}")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("   Execute 'python setup_enhanced_rag.py' primeiro")
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

async def demo_semantic_chunking():
    """Demonstrate semantic chunking capabilities"""
    print_section("Demonstração: Chunking Semântico", "🧠")
    
    try:
        from local_rag.chunking.semantic_chunker import get_semantic_chunker
        
        # Sample technical text
        sample_text = """
        INSIM-FT (Interwell Numerical Simulation Model with Front Tracking) é um modelo de simulação numérica desenvolvido para simular o fluxo de fluidos em reservatórios de petróleo.
        
        O modelo utiliza técnicas de rastreamento de frente para capturar a dinâmica de deslocamento de fluidos entre poços injetores e produtores.
        
        Características principais:
        1. Simulação rápida de conectividade entre poços
        2. Rastreamento preciso de frentes de saturação
        3. Aplicável a reservatórios com heterogeneidade complexa
        
        A permeabilidade é um parâmetro fundamental no INSIM-FT, medida tipicamente em milidarcy (mD). 
        Valores típicos variam de 1 mD a 1000 mD para reservatórios convencionais.
        
        A equação básica de Darcy é utilizada para calcular o fluxo:
        q = (k * A * Δp) / (μ * L)
        
        Onde:
        - q: vazão
        - k: permeabilidade 
        - A: área da seção transversal
        - Δp: diferencial de pressão
        - μ: viscosidade do fluido
        - L: comprimento
        """
        
        print("📄 Texto de exemplo (simulação técnica)...")
        print("🔄 Aplicando chunking semântico...")
        
        # Get semantic chunker
        chunker = get_semantic_chunker()
        
        # Create semantic chunks
        metadata = {
            'source': 'demo_document.pdf',
            'page': 1,
            'section': 'methodology'
        }
        
        chunks = chunker.chunk_document(sample_text, metadata)
        
        print(f"\n✅ Chunks criados: {len(chunks)}")
        
        # Show chunking statistics
        stats = chunker.get_chunking_stats(chunks)
        
        print(f"📊 Estatísticas:")
        print(f"   📏 Tamanho médio: {stats.get('avg_chunk_size', 0):.0f} caracteres")
        print(f"   🎯 Chunks de alta qualidade: {stats.get('high_quality_chunks', 0)}")
        print(f"   🔬 Chunks técnicos: {stats.get('technical_chunks', 0)}")
        
        # Show sample chunks
        print(f"\n📋 Amostra dos chunks:")
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"\n   {i}. Chunk (score: {chunk.semantic_score:.2f}):")
            print(f"      Tipo: {chunk.section_type}")
            print(f"      Tamanho: {len(chunk.content)} chars")
            print(f"      Keywords: {', '.join(chunk.keywords[:3])}...")
            print(f"      Conteúdo: {chunk.content[:100]}...")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

async def demo_rag_fusion():
    """Demonstrate RAG Fusion capabilities"""
    print_section("Demonstração: RAG Fusion", "🔀")
    
    try:
        from local_rag.fusion.rag_fusion import get_rag_fusion
        
        print("🔄 Inicializando RAG Fusion...")
        
        fusion = get_rag_fusion()
        
        # Demo query expansion
        original_query = "permeabilidade do reservatório"
        
        print(f"📝 Query original: '{original_query}'")
        print("🔄 Expandindo query...")
        
        # Get query variations
        expanded_queries = fusion.query_expander.expand_query(original_query, max_variations=5)
        
        print(f"\n✅ Variações geradas ({len(expanded_queries)}):")
        for i, query in enumerate(expanded_queries, 1):
            print(f"   {i}. {query}")
        
        # Demo result fusion (mock function)
        def mock_search_function(query, top_k=3):
            """Mock search function for demonstration"""
            return [
                {
                    'content': f"Resultado simulado para '{query}' - conteúdo sobre permeabilidade...",
                    'metadata': {'source': 'doc1.pdf', 'page': 1},
                    'similarity': 0.8 + (hash(query) % 10) * 0.02
                },
                {
                    'content': f"Segundo resultado para '{query}' - mais detalhes técnicos...",
                    'metadata': {'source': 'doc2.pdf', 'page': 5},
                    'similarity': 0.7 + (hash(query) % 8) * 0.025
                }
            ]
        
        print("\n🔍 Simulando busca com fusão...")
        
        # Simulate fusion search
        fusion_result = await fusion.enhanced_search(
            original_query=original_query,
            search_function=mock_search_function,
            max_variations=3,
            fusion_method="reciprocal_rank",
            top_k=5
        )
        
        print(f"✅ Resultados fusionados: {len(fusion_result['results'])}")
        print(f"📊 Análise de qualidade:")
        analysis = fusion_result['fusion_analysis']
        print(f"   🎯 Score médio: {analysis.get('avg_fusion_score', 0):.3f}")
        print(f"   🔗 Matches por query: {analysis.get('avg_query_matches', 0):.1f}")
        
        # Show sample results
        print(f"\n📋 Amostra dos resultados:")
        for i, result in enumerate(fusion_result['results'][:2], 1):
            print(f"   {i}. Score fusão: {result.get('fusion_score', 0):.3f}")
            print(f"      Matches: {result.get('query_matches', 0)} queries")
            print(f"      Preview: {result['content'][:80]}...")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

async def demo_self_critique():
    """Demonstrate self-critique system"""
    print_section("Demonstração: Self-Critique System", "🔍")
    
    try:
        from local_rag.quality.self_critique import get_self_critique_system
        
        print("🔄 Inicializando sistema de auto-crítica...")
        
        critic = get_self_critique_system()
        
        # Demo question and answers (good vs bad)
        question = "O que é permeabilidade e qual sua unidade de medida?"
        
        good_answer = """
        Permeabilidade é a propriedade da rocha que indica sua capacidade de permitir o fluxo de fluidos através de seus poros. 
        É medida tipicamente em darcy (D) ou milidarcy (mD), onde 1 darcy equivale a aproximadamente 9.87 × 10⁻¹³ m².
        Em reservatórios de petróleo, valores típicos variam de 1 mD a 1000 mD para reservatórios convencionais.
        A permeabilidade é fundamental na equação de Darcy para cálculo de fluxo de fluidos.
        """
        
        bad_answer = """
        Permeabilidade é algo relacionado a rochas. É importante para petróleo.
        Não tenho certeza da unidade, mas deve ser em metros ou algo assim.
        """
        
        sources = [
            "A permeabilidade é uma propriedade fundamental da rocha reservatório, medida em darcy ou milidarcy.",
            "A equação de Darcy relaciona permeabilidade, viscosidade e gradiente de pressão para calcular vazão.",
            "Valores típicos de permeabilidade em reservatórios convencionais variam de 1 a 1000 mD."
        ]
        
        print(f"❓ Pergunta: {question}")
        
        # Critique good answer
        print(f"\n✅ Analisando resposta BOA...")
        good_critique = critic.critique_answer(question, good_answer, sources)
        
        print(f"📊 Resultados:")
        print(f"   🎯 Score geral: {good_critique.overall_score:.2f}")
        print(f"   📝 Relevância: {good_critique.relevance_score:.2f}")
        print(f"   ✔️  Factual: {good_critique.factual_score:.2f}")
        print(f"   🔧 Técnico: {good_critique.technical_accuracy_score:.2f}")
        print(f"   🏆 Validado: {'✅' if good_critique.validated else '❌'}")
        
        if good_critique.issues_found:
            print(f"   ⚠️  Issues: {'; '.join(good_critique.issues_found)}")
        
        # Critique bad answer
        print(f"\n❌ Analisando resposta RUIM...")
        bad_critique = critic.critique_answer(question, bad_answer, sources)
        
        print(f"📊 Resultados:")
        print(f"   🎯 Score geral: {bad_critique.overall_score:.2f}")
        print(f"   📝 Relevância: {bad_critique.relevance_score:.2f}")
        print(f"   ✔️  Factual: {bad_critique.factual_score:.2f}")
        print(f"   🔧 Técnico: {bad_critique.technical_accuracy_score:.2f}")
        print(f"   🏆 Validado: {'✅' if bad_critique.validated else '❌'}")
        
        if bad_critique.issues_found:
            print(f"   ⚠️  Issues: {'; '.join(bad_critique.issues_found[:2])}...")
        
        if bad_critique.suggestions:
            print(f"   💡 Sugestões: {'; '.join(bad_critique.suggestions[:2])}...")
        
        # Show comparison
        improvement = good_critique.overall_score - bad_critique.overall_score
        print(f"\n📈 Diferença de qualidade: {improvement:.2f} pontos ({improvement/bad_critique.overall_score*100:.0f}% melhor)")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

async def demo_cache_performance():
    """Demonstrate caching performance"""
    print_section("Demonstração: Cache Performance", "⚡")
    
    try:
        from local_rag.caching.enhanced_cache_service import get_enhanced_cache_service
        
        print("🔄 Inicializando sistema de cache...")
        
        cache = get_enhanced_cache_service()
        
        # Check cache health
        health = cache.health_check()
        print(f"🏥 Cache health: {'✅' if health['healthy'] else '❌'}")
        print(f"📊 Backend: {health['backend']}")
        
        if health['healthy'] and 'latency_ms' in health:
            print(f"⚡ Latência: {health['latency_ms']:.1f}ms")
        
        # Demo caching operations
        print(f"\n🔄 Testando operações de cache...")
        
        # Test data
        test_query = "teste permeabilidade"
        test_params = {"top_k": 5, "threshold": 0.3}
        test_results = [
            {"content": "Resultado 1", "similarity": 0.8},
            {"content": "Resultado 2", "similarity": 0.7}
        ]
        
        # Cache write
        start_time = time.time()
        cache_success = cache.cache_search_results(test_query, test_params, test_results)
        write_time = (time.time() - start_time) * 1000
        
        print(f"   💾 Cache write: {'✅' if cache_success else '❌'} ({write_time:.1f}ms)")
        
        # Cache read
        start_time = time.time()
        cached_results = cache.get_search_results(test_query, test_params)
        read_time = (time.time() - start_time) * 1000
        
        cache_hit = cached_results is not None
        print(f"   📖 Cache read: {'✅' if cache_hit else '❌'} ({read_time:.1f}ms)")
        
        if cache_hit:
            print(f"   🎯 Resultados recuperados: {len(cached_results)}")
        
        # Performance simulation
        print(f"\n📈 Simulação de performance:")
        
        # Simulate uncached query (slower)
        uncached_time = 850  # ms
        cached_time = read_time
        
        speedup = (uncached_time - cached_time) / uncached_time * 100
        print(f"   🐌 Sem cache: ~{uncached_time:.0f}ms")
        print(f"   ⚡ Com cache: {cached_time:.1f}ms")
        print(f"   🚀 Speedup: {speedup:.0f}% mais rápido")
        
        # Cache statistics
        stats = cache.get_cache_stats()
        if stats.get('connected'):
            print(f"\n📊 Estatísticas do cache:")
            operations = stats.get('operations', {})
            total_ops = sum(operations.values())
            print(f"   🔢 Total operações: {total_ops}")
            
            if 'hit' in operations and 'get' in operations:
                hit_ratio = operations['hit'] / max(1, operations['get']) * 100
                print(f"   🎯 Hit ratio: {hit_ratio:.1f}%")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

async def demo_system_status():
    """Show comprehensive system status"""
    print_section("Status Geral do Sistema", "📊")
    
    try:
        from local_rag.enhanced_rag_system import EnhancedRAGSystem
        
        print("🔄 Verificando status do sistema enhanced...")
        
        # Initialize enhanced system (lightweight)
        rag = EnhancedRAGSystem()
        
        # Get comprehensive status
        status = rag.get_enhanced_status()
        
        print(f"\n🏆 Sistema Enhanced RAG v{status.get('enhanced_rag', {}).get('version', '2.0')}")
        
        # Vector store status
        vs_status = status.get('vector_store', {})
        print(f"\n📚 Vector Store:")
        print(f"   📄 Documentos: {vs_status.get('total_documents', 0)}")
        print(f"   📁 Coleção: {vs_status.get('collection_name', 'N/A')}")
        
        # LLM status
        llm_status = status.get('llm', {})
        print(f"\n🤖 LLM:")
        print(f"   📦 Modelo: {llm_status.get('model', 'N/A')}")
        print(f"   🔗 Status: {llm_status.get('status', 'unknown')}")
        
        # Embedder status
        emb_status = status.get('embedder', {})
        print(f"\n🧠 Embeddings:")
        print(f"   📦 Modelo: {emb_status.get('model', 'N/A')}")
        print(f"   📐 Dimensões: {emb_status.get('dimension', 'N/A')}")
        print(f"   💻 Device: {emb_status.get('device', 'N/A')}")
        
        # Enhanced features
        enhanced = status.get('enhanced_rag', {})
        print(f"\n✨ Features Enhanced:")
        features = [
            ('🧠 Semantic Chunking', enhanced.get('semantic_chunking', False)),
            ('🔀 RAG Fusion', enhanced.get('rag_fusion', False)),
            ('🎯 Cross-Encoder', enhanced.get('cross_encoder_reranking', False)),
            ('🧩 Context Enhancement', enhanced.get('context_enhancement', False))
        ]
        
        for feature_name, enabled in features:
            status_icon = '✅' if enabled else '❌'
            print(f"   {feature_name}: {status_icon}")
        
        # Performance features
        perf_features = status.get('performance_features', {})
        if perf_features:
            print(f"\n⚡ Performance Features:")
            for feature, enabled in perf_features.items():
                if enabled:
                    readable_name = feature.replace('_', ' ').title()
                    print(f"   ✅ {readable_name}")
        
        # Overall health score
        health_score = 0
        total_checks = 0
        
        # Check core components
        if vs_status.get('total_documents', 0) > 0:
            health_score += 1
        total_checks += 1
        
        if llm_status.get('status') == 'online':
            health_score += 1
        total_checks += 1
        
        if enhanced.get('semantic_chunking'):
            health_score += 1
        if enhanced.get('rag_fusion'):
            health_score += 1
        if enhanced.get('cross_encoder_reranking'):
            health_score += 1
        total_checks += 3
        
        health_percentage = (health_score / total_checks) * 100 if total_checks > 0 else 0
        
        print(f"\n🏥 Health Score: {health_percentage:.0f}% ({health_score}/{total_checks})")
        
        if health_percentage >= 80:
            print("   🟢 Sistema funcionando perfeitamente!")
        elif health_percentage >= 60:
            print("   🟡 Sistema funcional com algumas limitações")
        else:
            print("   🔴 Sistema precisa de configuração adicional")
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("   Execute 'python setup_enhanced_rag.py' primeiro")
    except Exception as e:
        print(f"❌ Erro ao verificar status: {e}")

async def main():
    """Main demo function"""
    print_banner()
    
    # Check if basic setup is done
    venv_path = project_root / "venv"
    if not venv_path.exists():
        print("⚠️  Ambiente virtual não encontrado!")
        print("   Execute primeiro: python setup_enhanced_rag.py")
        return
    
    # Menu de demonstrações
    demos = [
        ("System Status", demo_system_status),
        ("Semantic Chunking", demo_semantic_chunking),
        ("RAG Fusion", demo_rag_fusion),
        ("Self-Critique", demo_self_critique),
        ("Cache Performance", demo_cache_performance),
        ("Basic vs Enhanced", demo_basic_vs_enhanced),
    ]
    
    print("\n🎯 Demonstrações Disponíveis:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"   {i}. {name}")
    print("   0. Executar todas")
    print("   q. Sair")
    
    while True:
        try:
            choice = input("\n🔍 Escolha uma demonstração (número): ").strip()
            
            if choice.lower() == 'q':
                print("👋 Até logo!")
                break
            elif choice == '0':
                print("\n🚀 Executando todas as demonstrações...")
                for name, demo_func in demos:
                    print(f"\n{'='*60}")
                    await demo_func()
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(demos):
                name, demo_func = demos[int(choice) - 1]
                print(f"\n🚀 Executando: {name}")
                await demo_func()
            else:
                print("❌ Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n👋 Demonstração interrompida. Até logo!")
            break
        except Exception as e:
            print(f"❌ Erro na demonstração: {e}")

if __name__ == "__main__":
    # Run async main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrompida. Até logo!")
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        sys.exit(1)