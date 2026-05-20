"""
Sistema RAG completo local para IAbel
Integra processamento de PDFs, embeddings, vector store e LLM
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Importa componentes locais
import sys
import os
from pathlib import Path

# Adiciona diretório local_rag ao path para importações relativas
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from processors.pdf_processor import PDFProcessor, DocumentChunk
from embeddings.local_embedder import LocalEmbedder
from vectorstore.chroma_store import LocalVectorStore
from models.ollama_client import OllamaClient
from models.gemini_client import GeminiClient
from query_optimizer import QueryOptimizer
from context_enhancer import ContextEnhancer

class LocalRAGSystem:
    def __init__(self,
                 vectorstore_path: str = "./vectorstore",
                 embedder_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 llm_model: str = "llama3.2:3b",
                 llm_provider: str = "ollama",
                 chunk_size: int = 700,
                 chunk_overlap: int = 120):
        """
        Inicializa sistema RAG completo

        Args:
            vectorstore_path: Caminho para armazenar vector database
            embedder_model: Modelo para embeddings
            llm_model: Modelo LLM a usar
            llm_provider: Provedor do LLM ("ollama", "gemini", "openai")
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
        """
        print("🚀 Inicializando Sistema RAG para IAbel...")

        # Inicializa componentes
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = LocalEmbedder(model_name=embedder_model)
        self.vector_store = LocalVectorStore(persist_directory=vectorstore_path)

        # Inicializa LLM Client baseado no provider
        self.llm_provider = llm_provider.lower()
        if self.llm_provider == "gemini":
            self.llm_client = GeminiClient(model_name=llm_model)
        elif self.llm_provider == "ollama":
            self.llm_client = OllamaClient(model_name=llm_model)
        else:
            raise ValueError(f"Provider não suportado: {llm_provider}. Use 'ollama' ou 'gemini'")

        self.query_optimizer = QueryOptimizer()
        self.context_enhancer = ContextEnhancer()

        # Configurações
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print("✅ Sistema RAG inicializado com sucesso!")
        print(f"   - PDF Processor: {chunk_size} chars/chunk, overlap: {chunk_overlap}")
        print(f"   - Embedder: {embedder_model}")
        print(f"   - LLM Provider: {llm_provider}")
        print(f"   - Vector Store: {vectorstore_path}")
        print(f"   - LLM: {llm_model}")
    
    def index_pdf_directory(self, pdf_directory: str, force_reindex: bool = False) -> bool:
        """
        Indexa todos os PDFs de um diretório
        
        Args:
            pdf_directory: Diretório com PDFs
            force_reindex: Se deve reindexar mesmo se já existirem dados
        """
        if not os.path.exists(pdf_directory):
            print(f"❌ Diretório não encontrado: {pdf_directory}")
            return False
        
        # Verifica se já existem documentos indexados
        stats = self.vector_store.get_collection_stats()
        if stats['total_documents'] > 0 and not force_reindex:
            print(f"ℹ️  Já existem {stats['total_documents']} documentos indexados")
            print("   Use force_reindex=True para reindexar")
            return True
        
        # Limpa coleção se force_reindex
        if force_reindex:
            print("🧹 Limpando índice existente...")
            self.vector_store.clear_collection()
        
        print(f"📁 Processando PDFs em: {pdf_directory}")
        
        # Processa todos os PDFs
        all_chunks = self.pdf_processor.process_directory(pdf_directory)
        
        if not all_chunks:
            print("❌ Nenhum chunk encontrado nos PDFs")
            return False
        
        print(f"📄 Total de chunks extraídos: {len(all_chunks)}")
        
        # Gera embeddings em lotes
        print("🔄 Gerando embeddings...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedder.embed_batch(texts, batch_size=32)
        
        # Prepara dados para o vector store
        documents = [chunk.content for chunk in all_chunks]
        metadatas = [chunk.metadata for chunk in all_chunks]
        ids = [chunk.chunk_id for chunk in all_chunks]
        
        # Adiciona ao vector store
        print("💾 Adicionando ao vector store...")
        success = self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        if success:
            print("✅ Indexação concluída com sucesso!")
            stats = self.vector_store.get_collection_stats()
            print(f"   - Total de documentos: {stats['total_documents']}")
            return True
        else:
            print("❌ Erro na indexação")
            return False
    
    def search_documents(self, 
                        query: str, 
                        top_k: int = 8,
                        similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Busca documentos similares à query
        
        Args:
            query: Pergunta/busca do usuário
            top_k: Número de documentos a retornar
            similarity_threshold: Limite mínimo de similaridade
        """
        print(f"🔍 Buscando: '{query}'")
        
        # Otimiza query para múltiplas variações
        optimized_queries = self.query_optimizer.optimize_query(query)
        print(f"🔄 Testando {len(optimized_queries)} variações da query")
        
        all_results = {}
        
        # Busca com cada variação otimizada
        for opt_query in optimized_queries:
            query_embedding = self.embedder.embed_text(opt_query)
            
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                n_results=top_k
            )
            
            # Processa resultados desta query
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'],
                results['metadatas'], 
                results['distances']
            )):
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    doc_id = metadata.get('source', '') + str(metadata.get('page', ''))
                    
                    # Se documento já existe, mantém a melhor similaridade
                    if doc_id in all_results:
                        if similarity > all_results[doc_id]['similarity']:
                            all_results[doc_id]['similarity'] = similarity
                            all_results[doc_id]['distance'] = distance
                            all_results[doc_id]['matched_query'] = opt_query
                    else:
                        all_results[doc_id] = {
                            'content': doc,
                            'metadata': metadata,
                            'similarity': similarity,
                            'distance': distance,
                            'matched_query': opt_query
                        }
        
        # Converte para lista e ordena por similaridade
        processed_results = []
        for i, (doc_id, result) in enumerate(sorted(all_results.items(), 
                                                   key=lambda x: x[1]['similarity'], 
                                                   reverse=True)):
            result['rank'] = i + 1
            processed_results.append(result)
        
        # Aplica boost para termos técnicos
        processed_results = self.query_optimizer.boost_technical_relevance(query, processed_results)
        
        # Aplica boost para seções prioritárias
        processed_results = self._boost_priority_sections(processed_results)
        
        print(f"📋 Encontrados {len(processed_results)} documentos relevantes")
        
        # Mostra top 3 resultados para debug
        for i, result in enumerate(processed_results[:3], 1):
            similarity_pct = result['similarity'] * 100
            print(f"   {i}. {similarity_pct:.1f}% - {result['metadata'].get('source', 'Unknown')} (p.{result['metadata'].get('page', '?')})")
        
        return processed_results
    
    def _boost_priority_sections(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aplica boost baseado na prioridade da seção (abstract, definição, etc.)
        """
        for result in results:
            metadata = result.get('metadata', {})
            priority_section = metadata.get('priority_section', 'regular')
            source = metadata.get('source', '')
            
            # Boost baseado no tipo de seção
            section_boosts = {
                'abstract': 1.4,      # Abstract: +40%
                'definition': 1.5,    # Definições: +50%
                'equation': 1.2,      # Equações: +20%
                'regular': 1.0        # Regular: sem boost
            }
            
            # Boost especial para arquivo INSIM_MALU (tem definições importantes)
            source_boosts = {
                'INSIM_MALU.pdf': 1.3,     # +30% para INSIM_MALU
                'Manual - INSIM_FT_compressed.pdf': 1.1  # +10% para manual
            }
            
            # Aplica boosts
            section_boost = section_boosts.get(priority_section, 1.0)
            source_boost = source_boosts.get(source, 1.0)
            
            # Combinação multiplicativa
            total_boost = section_boost * source_boost
            
            if total_boost > 1.0:
                original_similarity = result.get('similarity', 0.0)
                result['similarity'] = min(1.0, original_similarity * total_boost)
                result['priority_boost'] = total_boost
                result['section_type'] = priority_section
        
        # Reordena por similaridade atualizada
        return sorted(results, key=lambda x: x.get('similarity', 0), reverse=True)
    
    def ask_question(self, 
                    question: str,
                    top_k: int = 6,
                    include_sources: bool = True) -> Dict[str, Any]:
        """
        Responde pergunta usando RAG completo
        
        Args:
            question: Pergunta do usuário
            top_k: Número de documentos de contexto
            include_sources: Se deve incluir fontes na resposta
        """
        print(f"❓ Pergunta: {question}")
        
        # Busca documentos relevantes
        search_results = self.search_documents(question, top_k=top_k)
        
        if not search_results:
            return {
                'answer': "Não encontrei informações relevantes nos documentos indexados para responder sua pergunta.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Filtra apenas resultados com boa similaridade
        high_quality_results = [r for r in search_results if r['similarity'] >= 0.4]
        
        if not high_quality_results:
            # Se não há resultados de alta qualidade, usa os melhores disponíveis
            high_quality_results = search_results[:3]
        
        # Extrai contexto dos documentos de alta qualidade
        context_docs = [result['content'] for result in high_quality_results]
        
        # Enriquece contexto com definições de siglas e conceitos relacionados
        enhanced_context = self.context_enhancer.build_enhanced_context(question, context_docs)
        
        # Detecta lacunas nas definições
        definition_gaps = self.context_enhancer.detect_definition_gaps(question, high_quality_results)
        if definition_gaps:
            print(f"⚠️ Lacunas detectadas: {'; '.join(definition_gaps)}")
        
        # Gera resposta usando LLM com contexto enriquecido
        print("🤖 Gerando resposta com contexto enriquecido...")
        answer = self.llm_client.generate_with_context(
            question=question,
            context_documents=[enhanced_context],
            max_context_length=3000  # Mais contexto para incluir definições
        )
        
        # Calcula confiança baseada na similaridade dos melhores resultados
        if high_quality_results:
            # Pesa mais os resultados com maior similaridade
            weighted_similarity = sum(r['similarity'] ** 2 for r in high_quality_results[:3])
            avg_similarity = weighted_similarity / len(high_quality_results[:3])
        else:
            avg_similarity = 0.0
        
        # Prepara fontes se solicitado (apenas as de alta qualidade)
        sources = []
        if include_sources:
            for result in high_quality_results:
                sources.append({
                    'source': result['metadata'].get('source', 'Documento desconhecido'),
                    'page': result['metadata'].get('page', 'N/A'),
                    'similarity': round(result['similarity'], 3),
                    'preview': result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                    'matched_query': result.get('matched_query', question)
                })
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': round(avg_similarity, 3),
            'total_sources': len(high_quality_results)
        }
    
    def chat_session(self, max_interactions: int = 10):
        """
        Inicia sessão de chat interativo
        """
        print("\n" + "="*60)
        print("🤖 IAbel - Assistente de Engenharia de Reservatórios")
        print("   Sistema RAG Local Ativo")
        print("   Digite 'sair' para encerrar")
        print("="*60 + "\n")
        
        interaction_count = 0
        
        while interaction_count < max_interactions:
            try:
                question = input("\n👤 Você: ").strip()
                
                if question.lower() in ['sair', 'exit', 'quit']:
                    print("\n👋 Até logo!")
                    break
                
                if not question:
                    continue
                
                print("\n" + "-"*50)
                
                # Processa pergunta
                result = self.ask_question(question)
                
                # Exibe resposta
                print(f"\n🤖 IAbel: {result['answer']}")
                
                # Exibe fontes se houver
                if result['sources']:
                    print(f"\n📚 Fontes (confiança: {result['confidence']}):")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"   {i}. {source['source']} (p.{source['page']}) - {source['similarity']:.1%}")
                
                print("-"*50)
                interaction_count += 1
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrompido. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Retorna status do sistema RAG
        """
        # Status do vector store
        vs_stats = self.vector_store.get_collection_stats()
        
        # Status do LLM
        llm_status = self.llm_client.is_ollama_running()
        
        # Status do embedder
        embed_dim = self.embedder.get_embedding_dimension()
        
        return {
            'vector_store': {
                'total_documents': vs_stats['total_documents'],
                'collection_name': vs_stats['collection_name'],
                'persist_directory': vs_stats['persist_directory']
            },
            'embedder': {
                'model': self.embedder.model_name,
                'dimension': embed_dim,
                'device': self.embedder.device
            },
            'llm': {
                'model': self.llm_client.model_name,
                'status': 'online' if llm_status else 'offline',
                'base_url': self.llm_client.base_url
            },
            'processor': {
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }
    
    def export_knowledge_base(self, output_path: str) -> bool:
        """
        Exporta base de conhecimento para arquivo
        """
        return self.vector_store.export_collection(output_path)


# Função principal para setup rápido
def setup_isabel_rag(pdf_directory: str, 
                     vectorstore_path: str = "./vectorstore") -> LocalRAGSystem:
    """
    Setup rápido do sistema RAG para IAbel
    
    Args:
        pdf_directory: Diretório com PDFs de engenharia
        vectorstore_path: Caminho para vector store
    """
    print("🔧 Configurando IAbel RAG System...")
    
    # Cria sistema
    rag = LocalRAGSystem(vectorstore_path=vectorstore_path)
    
    # Indexa PDFs
    rag.index_pdf_directory(pdf_directory)
    
    # Exibe status
    status = rag.get_system_status()
    print("\n📊 Status do Sistema:")
    print(f"   - Documentos indexados: {status['vector_store']['total_documents']}")
    print(f"   - Modelo LLM: {status['llm']['model']} ({status['llm']['status']})")
    print(f"   - Embeddings: {status['embedder']['dimension']}D ({status['embedder']['device']})")
    
    return rag


if __name__ == "__main__":
    # Exemplo de uso
    pdf_dir = "../backend/data/pdfs"
    
    if os.path.exists(pdf_dir):
        isabel = setup_isabel_rag(pdf_dir)
        isabel.chat_session()
    else:
        print(f"❌ Diretório de PDFs não encontrado: {pdf_dir}")
        print("   Ajuste o caminho ou adicione PDFs ao diretório")