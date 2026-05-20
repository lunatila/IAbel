"""
Otimizador de queries para melhorar precisão do RAG
Expandir e reformular perguntas para melhor busca
"""

import re
from typing import List, Dict, Any

class QueryOptimizer:
    """
    Otimiza queries do usuário para melhorar precisão da busca
    """
    
    def __init__(self):
        # Dicionário de sinônimos técnicos multilíngues
        self.technical_synonyms = {
            # Siglas e acrônimos
            'insim': ['INSIM', 'INSIM-FT', 'INSIM-FT-3D', 'Interwell Numerical Simulation', 'Fast Track'],
            'insim-ft': ['INSIM', 'INSIM-FT-3D', 'Interwell Numerical Simulation Fast Track', 'simulação interpoços'],
            'eclipse': ['ECLIPSE', 'Schlumberger Eclipse', 'simulador Eclipse', 'reservoir simulator'],
            'cmg': ['CMG', 'Computer Modelling Group', 'IMEX', 'STARS', 'GEM'],
            
            # Propriedades da rocha (PT/EN)
            'permeabilidade': ['permeability', 'k', 'conductividade hidráulica', 'facilidade de fluxo', 'perm'],
            'permeability': ['permeabilidade', 'k', 'hydraulic conductivity', 'flow capacity'],
            'porosidade': ['porosity', 'phi', 'φ', 'espaço poroso', 'volume de poros'],
            'porosity': ['porosidade', 'phi', 'φ', 'pore space', 'pore volume'],
            'saturação': ['saturation', 'sw', 'so', 'sg', 'saturação de água', 'saturação de óleo'],
            'saturation': ['saturação', 'sw', 'so', 'sg', 'water saturation', 'oil saturation'],
            
            # Propriedades dos fluidos (PT/EN)
            'viscosidade': ['viscosity', 'mu', 'μ', 'resistência ao fluxo'],
            'viscosity': ['viscosidade', 'mu', 'μ', 'flow resistance'],
            'densidade': ['density', 'rho', 'ρ', 'peso específico', 'massa específica'],
            'density': ['densidade', 'rho', 'ρ', 'specific weight', 'specific mass'],
            'compressibilidade': ['compressibility', 'expansibilidade', 'compressão'],
            'compressibility': ['compressibilidade', 'expansion', 'compression'],
            
            # Simulação (PT/EN)
            'simulador': ['simulator', 'modelo numérico', 'software de simulação', 'numerical model'],
            'simulator': ['simulador', 'numerical model', 'simulation software'],
            'grid': ['malha', 'mesh', 'células', 'discretização', 'gridding'],
            'malha': ['grid', 'mesh', 'células', 'discretização', 'gridding'],
            'timestep': ['passo de tempo', 'dt', 'incremento temporal', 'time increment'],
            'passo de tempo': ['timestep', 'dt', 'time increment', 'temporal step'],
            
            # Poços (PT/EN)
            'poço': ['well', 'wellbore', 'furo', 'perfuração'],
            'well': ['poço', 'wellbore', 'borehole', 'hole'],
            'produção': ['production', 'extração', 'recuperação', 'producing'],
            'production': ['produção', 'extraction', 'recovery', 'producing'],
            'injeção': ['injection', 'injetor', 'injetividade', 'injector'],
            'injection': ['injeção', 'injector', 'injectivity'],
            
            # Fluidos (PT/EN)
            'óleo': ['oil', 'petróleo', 'crude', 'hidrocarboneto líquido', 'petroleum'],
            'oil': ['óleo', 'petróleo', 'crude', 'liquid hydrocarbon', 'petroleum'],
            'gás': ['gas', 'hidrocarboneto gasoso', 'fase gasosa', 'natural gas'],
            'gas': ['gás', 'gaseous hydrocarbon', 'gas phase', 'natural gas'],
            'água': ['water', 'fase aquosa', 'salmoura', 'brine', 'aqueous phase'],
            'water': ['água', 'aqueous phase', 'brine'],
            
            # Métodos e análises (PT/EN)
            'pvt': ['PVT', 'propriedades pvt', 'pressure volume temperature', 'análise pvt'],
            'material balance': ['balanço de materiais', 'balanço material', 'equação de balanço', 'material balance equation'],
            'balanço de materiais': ['material balance', 'material balance equation', 'mass balance'],
            'decline curve': ['curva de declínio', 'análise de declínio', 'declínio de produção', 'decline analysis'],
            'curva de declínio': ['decline curve', 'decline analysis', 'production decline'],
            'recovery factor': ['fator de recuperação', 'fr', 'eficiência de recuperação', 'recovery efficiency'],
            'fator de recuperação': ['recovery factor', 'rf', 'recovery efficiency'],
            
            # Unidades
            'md': ['milidarcy', 'millidarcy', 'permeabilidade', 'permeability unit'],
            'cp': ['centipoise', 'viscosidade', 'viscosity unit'],
            'bbl': ['barrel', 'barril', 'volume unit'],
            'scf': ['standard cubic feet', 'pé cúbico padrão', 'gas volume'],
            'psi': ['pounds per square inch', 'libra por polegada quadrada', 'pressure unit'],
            'bar': ['bar pressure', 'pressão em bar', 'pressure unit']
        }
        
        # Termos que devem ser expandidos com contexto
        self.context_expansions = {
            'darcy': 'lei de darcy fluxo permeabilidade',
            'buckley': 'buckley leverett frente de saturação',
            'black oil': 'modelo black oil pvt simulação',
            'compositional': 'modelo compositional equação de estado',
            'aquifer': 'aquífero água connate influxo',
            'relative permeability': 'permeabilidade relativa curvas kr',
            'capillary pressure': 'pressão capilar pc saturação'
        }
    
    def optimize_query(self, query: str) -> List[str]:
        """
        Otimiza uma query gerando múltiplas variações
        
        Args:
            query: Query original do usuário
            
        Returns:
            Lista de queries otimizadas
        """
        optimized_queries = [query.lower().strip()]
        
        # 1. Adiciona sinônimos técnicos
        synonymized_queries = self._add_synonyms(query)
        optimized_queries.extend(synonymized_queries)
        
        # 2. Expande termos técnicos com contexto
        expanded_queries = self._expand_technical_terms(query)
        optimized_queries.extend(expanded_queries)
        
        # 3. Reformula pergunta
        reformulated_queries = self._reformulate_question(query)
        optimized_queries.extend(reformulated_queries)
        
        # 4. Remove duplicatas mantendo ordem
        seen = set()
        unique_queries = []
        for q in optimized_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:5]  # Limita a 5 variações
    
    def _add_synonyms(self, query: str) -> List[str]:
        """
        Adiciona sinônimos técnicos à query
        """
        queries = []
        query_lower = query.lower()
        
        for term, synonyms in self.technical_synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Máximo 2 sinônimos
                    new_query = query_lower.replace(term, synonym)
                    if new_query != query_lower:
                        queries.append(new_query)
        
        return queries
    
    def _expand_technical_terms(self, query: str) -> List[str]:
        """
        Expande termos técnicos com contexto adicional
        """
        queries = []
        query_lower = query.lower()
        
        for term, expansion in self.context_expansions.items():
            if term in query_lower:
                # Adiciona contexto ao final
                expanded_query = f"{query_lower} {expansion}"
                queries.append(expanded_query)
        
        return queries
    
    def _reformulate_question(self, query: str) -> List[str]:
        """
        Reformula a pergunta em diferentes estilos
        """
        queries = []
        query_lower = query.lower().strip()
        
        # Remove palavras de pergunta para buscar conceitos diretos
        question_words = ['o que é', 'como', 'quando', 'onde', 'por que', 'qual', 'quais']
        
        for qw in question_words:
            if query_lower.startswith(qw):
                # Remove palavra de pergunta
                concept_query = query_lower.replace(qw, '').strip()
                if concept_query:
                    queries.append(concept_query)
                
                # Reformula como definição
                if qw in ['o que é', 'qual']:
                    definition_query = f"definição {concept_query}"
                    queries.append(definition_query)
                
                # Reformula como explicação
                elif qw == 'como':
                    explanation_query = f"explicação {concept_query}"
                    queries.append(explanation_query)
        
        # Se não é pergunta, adiciona palavras-chave relacionadas
        if not any(qw in query_lower for qw in question_words):
            concept_query = f"conceito {query_lower}"
            queries.append(concept_query)
        
        return queries
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extrai termos-chave técnicos da query
        """
        query_lower = query.lower()
        key_terms = []
        
        # Busca por termos técnicos conhecidos
        all_terms = list(self.technical_synonyms.keys()) + list(self.context_expansions.keys())
        
        for term in all_terms:
            if term in query_lower:
                key_terms.append(term)
        
        # Busca por números e unidades (importantes em engenharia)
        numbers_units = re.findall(r'\d+\s*(?:md|cp|bbl|scf|psi|bar|m³|m3|%)', query_lower)
        key_terms.extend(numbers_units)
        
        return key_terms
    
    def boost_technical_relevance(self, query: str, search_results: List[Dict]) -> List[Dict]:
        """
        Aumenta relevância de resultados com termos técnicos
        """
        key_terms = self.extract_key_terms(query)
        
        if not key_terms:
            return search_results
        
        # Boost baseado na presença de termos técnicos
        for result in search_results:
            content_lower = result.get('content', '').lower()
            
            # Conta quantos termos técnicos estão presentes
            term_matches = sum(1 for term in key_terms if term in content_lower)
            
            if term_matches > 0:
                # Aumenta similaridade baseado na presença de termos técnicos
                original_similarity = result.get('similarity', 0.0)
                boost_factor = 1 + (term_matches * 0.1)  # 10% boost por termo
                result['similarity'] = min(1.0, original_similarity * boost_factor)
                result['technical_boost'] = boost_factor
        
        # Reordena por similaridade atualizada
        return sorted(search_results, key=lambda x: x.get('similarity', 0), reverse=True)