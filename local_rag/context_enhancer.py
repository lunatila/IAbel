"""
Context Enhancer - Melhora contexto das respostas conectando siglas com definições
Especializado em terminologia de engenharia de reservatórios
"""

import re
from typing import List, Dict, Any

class ContextEnhancer:
    """
    Enriquece o contexto das respostas conectando siglas, definições e conceitos relacionados
    """
    
    def __init__(self):
        # Mapeamento explicito de siglas para suas definições completas
        self.acronym_definitions = {
            'INSIM': 'Interwell Numerical Simulation Model (INSIM) - Modelo de Simulação Numérica Interpoços',
            'INSIM-FT': 'Interwell Numerical Simulation Model with Front Tracking (INSIM-FT) - Modelo INSIM com Rastreamento de Frente',
            'INSIM-FT-3D': 'Interwell Numerical Simulation Model with Front Tracking in 3 Dimensions (INSIM-FT-3D) - Modelo INSIM com Rastreamento de Frente em 3 Dimensões',
            'FT': 'Front Tracking - Rastreamento de Frente (técnica de simulação)',
            'PVT': 'Pressure Volume Temperature (PVT) - Propriedades Pressão-Volume-Temperatura',
            'BHP': 'Bottom Hole Pressure (BHP) - Pressão no Fundo do Poço',
            'WOR': 'Water Oil Ratio (WOR) - Razão Água-Óleo',
            'GOR': 'Gas Oil Ratio (GOR) - Razão Gás-Óleo',
            'ESP': 'Electric Submersible Pump (ESP) - Bomba Centrífuga Submersa',
            'API': 'American Petroleum Institute (API) - Instituto Americano de Petróleo',
            'SPE': 'Society of Petroleum Engineers (SPE) - Sociedade dos Engenheiros de Petróleo',
            'ECLIPSE': 'ECLIPSE - Simulador de Reservatórios da Schlumberger',
            'CMG': 'Computer Modelling Group (CMG) - Grupo de Modelagem Computacional'
        }
        
        # Termos que frequentemente aparecem juntos
        self.related_concepts = {
            'insim': ['interwell', 'numerical simulation', 'reservoir modeling', 'well connectivity'],
            'front tracking': ['insim-ft', 'multiphase flow', 'saturation front', 'displacement'],
            '3d': ['three dimensional', 'tridimensional', 'spatial modeling', 'grid'],
            'simulator': ['numerical model', 'reservoir simulation', 'flow modeling'],
            'interwell': ['between wells', 'well connectivity', 'tracer flow', 'interference']
        }
    
    def enhance_search_results(self, query: str, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enriquece resultados de busca com contexto adicional sobre siglas e definições
        """
        enhanced_results = []
        
        for result in search_results:
            if isinstance(result, dict):
                enhanced_result = result.copy()
                content = result.get('content', '')
            else:
                enhanced_result = {'content': str(result)}
                content = str(result)
            
            # Adiciona definições de siglas encontradas no texto
            enhanced_content = self._add_acronym_context(content)
            
            # Adiciona contexto relacionado se a query contém termos específicos
            enhanced_content = self._add_related_context(query, enhanced_content)
            
            enhanced_result['enhanced_content'] = enhanced_content
            enhanced_result['acronyms_found'] = self._extract_acronyms(content)
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _add_acronym_context(self, content: str) -> str:
        """
        Adiciona definições explícitas de siglas encontradas no conteúdo
        """
        enhanced = content
        
        for acronym, definition in self.acronym_definitions.items():
            # Busca a sigla no texto (case insensitive)
            pattern = re.compile(r'\b' + re.escape(acronym) + r'\b', re.IGNORECASE)
            
            if pattern.search(content):
                # Adiciona definição no final se ainda não estiver presente
                if definition.lower() not in enhanced.lower():
                    enhanced += f"\n\n[DEFINIÇÃO: {definition}]"
        
        return enhanced
    
    def _add_related_context(self, query: str, content: str) -> str:
        """
        Adiciona contexto de conceitos relacionados baseado na query
        """
        query_lower = query.lower()
        enhanced = content
        
        for term, related in self.related_concepts.items():
            if term in query_lower:
                # Adiciona conceitos relacionados se relevantes
                related_info = ", ".join(related)
                context_note = f"\n\n[CONCEITOS RELACIONADOS A '{term.upper()}': {related_info}]"
                
                if context_note not in enhanced:
                    enhanced += context_note
        
        return enhanced
    
    def _extract_acronyms(self, content: str) -> List[str]:
        """
        Extrai siglas/acrônimos encontrados no conteúdo
        """
        found_acronyms = []
        
        for acronym in self.acronym_definitions.keys():
            pattern = re.compile(r'\b' + re.escape(acronym) + r'\b', re.IGNORECASE)
            if pattern.search(content):
                found_acronyms.append(acronym)
        
        return found_acronyms
    
    def build_enhanced_context(self, query: str, documents: List[str]) -> str:
        """
        Constrói contexto enriquecido para o LLM, conectando definições e conceitos
        """
        # Combina todos os documentos
        full_context = "\n\n".join(documents)
        
        # Identifica siglas na query
        query_acronyms = self._extract_acronyms(query)
        
        # Adiciona definições das siglas encontradas na query
        definitions_to_add = []
        for acronym in query_acronyms:
            if acronym in self.acronym_definitions:
                definitions_to_add.append(self.acronym_definitions[acronym])
        
        # Monta contexto final
        enhanced_context = full_context
        
        if definitions_to_add:
            definitions_section = "\n\n=== DEFINIÇÕES IMPORTANTES ===\n"
            definitions_section += "\n".join(f"• {defn}" for defn in definitions_to_add)
            enhanced_context = definitions_section + "\n\n" + full_context
        
        # Adiciona instruções especiais para o LLM sobre siglas
        if any(term in query.lower() for term in ['sigla', 'significa', 'acronym', 'stands for']):
            instruction = "\n\n=== INSTRUÇÃO ESPECIAL ===\nO usuário está perguntando sobre o significado de uma sigla/acrônimo. Procure por definições explícitas ou expansões da sigla nos documentos acima."
            enhanced_context += instruction
        
        return enhanced_context
    
    def detect_definition_gaps(self, query: str, search_results: List[Dict[str, Any]]) -> List[str]:
        """
        Detecta lacunas nas definições que deveriam estar presentes
        """
        gaps = []
        query_lower = query.lower()
        
        # Verifica se é uma pergunta sobre significado de sigla
        if any(indicator in query_lower for indicator in ['significa', 'sigla', 'stands for', 'acronym']):
            # Extrai possível sigla da query
            words = query.upper().split()
            potential_acronyms = [word for word in words if len(word) <= 10 and word.isalpha()]
            
            for acronym in potential_acronyms:
                if acronym in self.acronym_definitions:
                    # Verifica se a definição aparece nos resultados
                    definition_found = False
                    for result in search_results:
                        if isinstance(result, dict):
                            content = result.get('content', '').lower()
                        else:
                            content = str(result).lower()
                        definition = self.acronym_definitions[acronym].lower()
                        
                        if 'interwell' in definition and 'interwell' in content:
                            definition_found = True
                            break
                    
                    if not definition_found:
                        gaps.append(f"Definição completa de {acronym} não encontrada nos resultados")
        
        return gaps