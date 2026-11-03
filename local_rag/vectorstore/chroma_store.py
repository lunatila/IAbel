"""
Sistema de Vector Database Local com ChromaDB
Armazenamento local para busca semântica de documentos de engenharia
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import logging

# Disable ChromaDB telemetry to avoid posthog errors
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

logger = logging.getLogger(__name__)

class LocalVectorStore:
    def __init__(self, 
                 persist_directory: str = "./vectorstore",
                 collection_name: str = "reservoir_engineering_docs"):
        """
        Inicializa o vector store local com ChromaDB
        
        Args:
            persist_directory: Diretório para persistir dados
            collection_name: Nome da coleção
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Cria diretório se não existir
        os.makedirs(persist_directory, exist_ok=True)
        
        # Inicializa ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Configura ChromaDB
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Cria ou obtém coleção
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"ChromaDB collection '{collection_name}' loaded")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "description": "Documentos multilíngues com seções prioritárias",
                        "created_at": datetime.now().isoformat()
                    }
                )
                logger.info(f"ChromaDB collection '{collection_name}' created")
            
            print(f"✅ Vector store inicializado: {persist_directory}")
            print(f"   Coleção: {collection_name}")
            print(f"   Documentos: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"❌ Falha ao inicializar vector store: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str],
                     embeddings: List[np.ndarray],
                     metadatas: List[Dict[str, Any]],
                     ids: List[str]) -> bool:
        """
        Adiciona documentos ao vector store
        
        Args:
            documents: Lista de textos dos documentos
            embeddings: Lista de embeddings (numpy arrays)
            metadatas: Lista de metadados
            ids: Lista de IDs únicos
        """
        try:
            # Verifica IDs únicos
            unique_ids = set(ids)
            if len(unique_ids) != len(ids):
                logger.error(f"❌ {len(ids) - len(unique_ids)} IDs duplicados encontrados")
                return False
            
            # Converte numpy arrays para listas
            embeddings_list = []
            for emb in embeddings:
                if isinstance(emb, np.ndarray):
                    embeddings_list.append(emb.tolist())
                else:
                    embeddings_list.append(emb)
            
            # Verifica dimensões consistentes
            if embeddings_list:
                first_dim = len(embeddings_list[0])
                for i, emb in enumerate(embeddings_list):
                    if len(emb) != first_dim:
                        logger.error(f"Embedding {i} tem dimensão {len(emb)}, esperado {first_dim}")
                        return False
            
            # Conta seções prioritárias
            priority_count = sum(1 for m in metadatas if m.get('priority_section', 'regular') != 'regular')
            logger.info(f"Seções prioritárias detectadas: {priority_count}")
            
            # Adiciona documentos em lotes
            batch_size = 25
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_embeddings = embeddings_list[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"✅ {len(documents)} documentos adicionados")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao adicionar documentos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def search_similar(self,
                      query_embedding: np.ndarray,
                      n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Busca documentos similares usando embedding da query

        Args:
            query_embedding: Embedding da query
            n_results: Número de resultados
            where: Filtros de metadados
        """
        try:
            # Converte numpy array para lista
            query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding

            # Executa busca
            results = self.collection.query(
                query_embeddings=[query_list],
                n_results=n_results,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )

            # Processa resultados
            processed_results = {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'total_results': len(results['documents'][0]) if results['documents'] else 0
            }

            return processed_results

        except Exception as e:
            logger.error(f"❌ Erro na busca: {e}")
            import traceback
            traceback.print_exc()
            return {'documents': [], 'metadatas': [], 'distances': [], 'total_results': 0}

    def search(self,
              query_embedding: np.ndarray,
              top_k: int = 5,
              similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents (RAG v3 compatible format)

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of search results with content, metadata, and similarity
        """
        try:
            # Use search_similar internally
            raw_results = self.search_similar(
                query_embedding=query_embedding,
                n_results=top_k
            )

            # Convert to RAG v3 expected format
            formatted_results = []
            for i in range(len(raw_results['documents'])):
                # Convert distance to similarity (cosine distance to cosine similarity)
                # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity: similarity = 1 - (distance / 2)
                distance = raw_results['distances'][i]
                similarity = 1.0 - (distance / 2.0)

                # Apply similarity threshold
                if similarity < similarity_threshold:
                    continue

                result = {
                    'content': raw_results['documents'][i],
                    'metadata': raw_results['metadatas'][i],
                    'similarity': similarity,
                    'distance': distance
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"❌ Erro na busca: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas da coleção
        """
        try:
            return {
                'total_documents': self.collection.count(),
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'type': 'ChromaDB'
            }
        except Exception as e:
            logger.error(f"❌ Erro ao obter estatísticas: {e}")
            return {
                'total_documents': 0,
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory
            }
    
    def clear_collection(self) -> bool:
        """
        Remove todos os documentos da coleção
        """
        try:
            # Obtém todos os IDs; apaga em lotes porque o ChromaDB limita o tamanho de cada operação
            all_ids = self.collection.get(include=[])['ids']
            batch_size = 5000
            for i in range(0, len(all_ids), batch_size):
                self.collection.delete(ids=all_ids[i:i + batch_size])
            
            print("✅ Coleção limpa com sucesso")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao limpar coleção: {e}")
            import traceback
            traceback.print_exc()
            return False


# Função de conveniência
def create_vector_store(persist_directory: str = "./vectorstore",
                       collection_name: str = "reservoir_engineering_docs") -> LocalVectorStore:
    """
    Cria um vector store local com configurações padrão
    """
    return LocalVectorStore(persist_directory, collection_name)