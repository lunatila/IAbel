"""
Sistema de embeddings locais usando sentence-transformers
Otimizado para textos técnicos de engenharia de reservatórios
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import os
import pickle
import torch
import re

class LocalEmbedder:
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 cache_dir: str = "./models_cache"):
        """
        Inicializa o embedder local
        
        Args:
            model_name: Nome do modelo sentence-transformers
            cache_dir: Diretório para cache dos modelos
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Verifica se CUDA está disponível
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando device: {self.device}")
        
        # Carrega o modelo
        self.model = self._load_model()
        
        # Cache para embeddings calculados
        self.embedding_cache = {}
        self.cache_file = os.path.join(cache_dir, "embedding_cache.pkl")
        self._load_cache()
    
    def _load_model(self) -> SentenceTransformer:
        """
        Carrega o modelo sentence-transformers
        """
        try:
            print(f"Carregando modelo: {self.model_name}")
            model = SentenceTransformer(
                self.model_name, 
                cache_folder=self.cache_dir,
                device=self.device
            )
            print(f"Modelo carregado com sucesso. Dimensão: {model.get_sentence_embedding_dimension()}")
            return model
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            # Fallback para modelo mais simples
            fallback_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
            print(f"Tentando modelo fallback: {fallback_model}")
            return SentenceTransformer(fallback_model, cache_folder=self.cache_dir)
    
    def _load_cache(self):
        """
        Carrega cache de embeddings salvos
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Cache carregado: {len(self.embedding_cache)} embeddings")
            except Exception as e:
                print(f"Erro ao carregar cache: {e}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """
        Salva cache de embeddings
        """
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Erro ao salvar cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """
        Gera chave única para o cache
        """
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Gera embedding para um texto
        """
        cache_key = self._get_cache_key(text)

        # Verifica cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        # Processa texto
        processed_text = self._preprocess_text(text)

        # Gera embedding
        embedding = self.model.encode([processed_text], convert_to_tensor=False)[0]

        # Salva no cache
        self.embedding_cache[cache_key] = embedding

        return embedding

    def embed_query(self, query: str) -> np.ndarray:
        """
        Gera embedding para uma query (alias para embed_text)
        """
        return self.embed_text(query)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Gera embeddings para múltiplos textos em lotes
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed_batch = [self._preprocess_text(text) for text in batch]
            
            # Verifica cache para cada texto do lote
            batch_embeddings = []
            texts_to_compute = []
            indices_to_compute = []
            
            for j, text in enumerate(batch):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[cache_key])
                else:
                    batch_embeddings.append(None)
                    texts_to_compute.append(processed_batch[j])
                    indices_to_compute.append(j)
            
            # Computa embeddings faltantes
            if texts_to_compute:
                new_embeddings = self.model.encode(
                    texts_to_compute, 
                    convert_to_tensor=False,
                    batch_size=min(batch_size, len(texts_to_compute))
                )
                
                # Insere embeddings computados
                for idx, embedding in zip(indices_to_compute, new_embeddings):
                    batch_embeddings[idx] = embedding
                    # Salva no cache
                    cache_key = self._get_cache_key(batch[idx])
                    self.embedding_cache[cache_key] = embedding
            
            embeddings.extend(batch_embeddings)
            
            print(f"Processados {min(i + batch_size, len(texts))}/{len(texts)} textos")
        
        # Salva cache periodicamente
        if len(texts) > 10:
            self._save_cache()
        
        return embeddings
    
    def _preprocess_text(self, text: str) -> str:
        """
        Pré-processa texto para melhorar embeddings multilíngues
        Adiciona contexto de engenharia de reservatórios e normaliza siglas
        """
        # Normaliza siglas e termos técnicos comuns
        text = self._normalize_technical_terms(text)
        
        # Adiciona contexto técnico se detectar termos específicos
        reservoir_terms = [
            # Português
            'permeabilidade', 'porosidade', 'saturação', 'simulador', 'reservatório', 
            'poço', 'produção', 'injeção', 'engenharia', 'petróleo', 'óleo', 'gás',
            'insim', 'eclipse', 'cmg', 'pvt', 'material balance', 'declínio',
            # Inglês
            'permeability', 'porosity', 'saturation', 'simulator', 'reservoir',
            'well', 'production', 'injection', 'engineering', 'petroleum', 'oil', 'gas',
            'insim', 'eclipse', 'cmg', 'pvt', 'material balance', 'decline'
        ]
        
        text_lower = text.lower()
        has_technical_terms = any(term in text_lower for term in reservoir_terms)
        
        if has_technical_terms:
            # Contexto multilíngue
            context_prefix = "Reservoir engineering simulation petroleum: "
            return context_prefix + text
        
        return text
    
    def _normalize_technical_terms(self, text: str) -> str:
        """
        Normaliza termos técnicos e siglas para melhorar busca
        """
        # Mapeamento de termos equivalentes
        term_mappings = {
            # Siglas expandidas
            'INSIM-FT': 'INSIM FT Interwell Numerical Simulation Fast Track',
            'INSIM-FT-3D': 'INSIM FT 3D Interwell Numerical Simulation Fast Track Three Dimensional',
            'PVT': 'PVT Pressure Volume Temperature',
            'BHP': 'BHP Bottom Hole Pressure',
            'WOR': 'WOR Water Oil Ratio',
            'GOR': 'GOR Gas Oil Ratio',
            'API': 'API American Petroleum Institute',
            'SPE': 'SPE Society of Petroleum Engineers',
            
            # Termos portugueses para inglês
            'permeabilidade absoluta': 'absolute permeability permeabilidade',
            'permeabilidade relativa': 'relative permeability permeabilidade',
            'fator de recuperação': 'recovery factor recuperação',
            'simulação numérica': 'numerical simulation simulação',
            'engenharia de reservatórios': 'reservoir engineering engenharia',
            'análise de declínio': 'decline curve analysis declínio',
            
            # Unidades
            'mD': 'milidarcy permeability',
            'cp': 'centipoise viscosity',
            'bbl': 'barrel volume',
            'scf': 'standard cubic feet gas',
            'psi': 'pounds per square inch pressure',
            'bar': 'bar pressure'
        }
        
        # Aplica mapeamentos
        normalized_text = text
        for original, expanded in term_mappings.items():
            # Case insensitive replacement
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            normalized_text = pattern.sub(f"{original} {expanded}", normalized_text)
        
        return normalized_text
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula similaridade entre dois embeddings (cosine similarity)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def find_most_similar(self, 
                         query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray],
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Encontra os embeddings mais similares à query
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Ordena por similaridade decrescente
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna a dimensão dos embeddings
        """
        return self.model.get_sentence_embedding_dimension()
    
    def __del__(self):
        """
        Salva cache ao destruir objeto
        """
        if hasattr(self, 'embedding_cache') and self.embedding_cache:
            self._save_cache()


# Função de conveniência para criar embedder padrão
def create_embedder(model_name: Optional[str] = None) -> LocalEmbedder:
    """
    Cria um embedder local com configurações padrão para engenharia de reservatórios
    """
    if model_name is None:
        # Modelos recomendados em ordem de preferência (multilíngue primeiro)
        preferred_models = [
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # MELHOR para multilíngue
            "sentence-transformers/distiluse-base-multilingual-cased",      # Alternativa multilíngue
            "sentence-transformers/all-mpnet-base-v2"                       # Fallback inglês
        ]
        model_name = preferred_models[0]
    
    return LocalEmbedder(model_name=model_name)