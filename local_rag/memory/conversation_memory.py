"""
Sistema de Memória Conversacional para RAG
Mantém contexto entre perguntas para referências como "item 1", "isso", "anterior"
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import re

@dataclass
class ConversationTurn:
    """Uma interação da conversa (pergunta + resposta)"""
    turn_id: str
    user_question: str
    assistant_response: str
    timestamp: datetime
    entities_mentioned: List[str]  # Listas, itens, conceitos mencionados
    response_metadata: Dict[str, Any]  # Sources, confidence, etc
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'user_question': self.user_question,
            'assistant_response': self.assistant_response,
            'timestamp': self.timestamp.isoformat(),
            'entities_mentioned': self.entities_mentioned,
            'response_metadata': self.response_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        return cls(
            turn_id=data['turn_id'],
            user_question=data['user_question'],
            assistant_response=data['assistant_response'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            entities_mentioned=data.get('entities_mentioned', []),
            response_metadata=data.get('response_metadata', {})
        )

@dataclass
class ConversationContext:
    """Contexto completo de uma conversa"""
    conversation_id: str
    user_id: str
    turns: List[ConversationTurn]
    created_at: datetime
    last_activity: datetime
    
    def add_turn(self, turn: ConversationTurn):
        """Adiciona uma nova interação"""
        self.turns.append(turn)
        self.last_activity = datetime.now()
    
    def get_recent_context(self, max_turns: int = 5) -> str:
        """Gera contexto textual das últimas interações"""
        recent_turns = self.turns[-max_turns:]
        context_parts = []
        
        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"Pergunta {i}: {turn.user_question}")
            context_parts.append(f"Resposta {i}: {turn.assistant_response}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def extract_list_items(self) -> Dict[str, List[str]]:
        """Extrai listas mencionadas na conversa"""
        lists_found = {}
        
        for turn in self.turns:
            response = turn.assistant_response
            
            # Procura por listas numeradas
            numbered_lists = re.findall(r'(\d+)\.\s+([^\n]+)', response)
            if numbered_lists:
                lists_found[f"lista_turn_{turn.turn_id}"] = [item[1] for item in numbered_lists]
            
            # Procura por listas com bullets
            bullet_lists = re.findall(r'[•\-\*]\s+([^\n]+)', response)
            if bullet_lists:
                lists_found[f"bullets_turn_{turn.turn_id}"] = bullet_lists
        
        return lists_found

class ConversationMemory:
    """
    Sistema de memória conversacional que mantém contexto entre perguntas
    """
    
    def __init__(self, storage_path: str = "data/conversations"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.max_conversation_age = timedelta(hours=24)  # 24h de validade
        
    def start_conversation(self, user_id: str = "default") -> str:
        """Inicia uma nova conversa"""
        conversation_id = str(uuid.uuid4())
        
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            turns=[],
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_conversations[conversation_id] = context
        return conversation_id
    
    def add_interaction(self, 
                       conversation_id: str,
                       user_question: str,
                       assistant_response: str,
                       response_metadata: Dict[str, Any] = None) -> str:
        """Adiciona uma interação à conversa"""
        
        if conversation_id not in self.active_conversations:
            # Se conversa não existe, cria uma nova
            conversation_id = self.start_conversation()
        
        turn_id = str(uuid.uuid4())
        
        # Extrai entidades mencionadas na resposta
        entities = self._extract_entities(assistant_response)
        
        turn = ConversationTurn(
            turn_id=turn_id,
            user_question=user_question,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            entities_mentioned=entities,
            response_metadata=response_metadata or {}
        )
        
        self.active_conversations[conversation_id].add_turn(turn)
        
        # Salva periodicamente
        self._save_conversation(conversation_id)
        
        return turn_id
    
    def enhance_question_with_context(self, 
                                    conversation_id: str,
                                    current_question: str) -> Tuple[str, str]:
        """
        Enriquece a pergunta atual com contexto da conversa
        
        Returns:
            (enhanced_question, context_summary)
        """
        
        if conversation_id not in self.active_conversations:
            return current_question, ""
        
        context = self.active_conversations[conversation_id]
        
        # Verifica se pergunta contém referências contextuais
        contextual_refs = self._detect_contextual_references(current_question)
        
        if not contextual_refs:
            return current_question, ""
        
        # Gera contexto relevante
        context_summary = self._build_context_summary(context, contextual_refs)
        
        # Enriquece a pergunta
        enhanced_question = self._enhance_question(current_question, context_summary)
        
        return enhanced_question, context_summary
    
    def _detect_contextual_references(self, question: str) -> List[str]:
        """Detecta referências contextuais na pergunta"""
        refs = []
        
        # Referências a itens numerados
        if re.search(r'\bitem\s+\d+\b', question.lower()):
            refs.append('numbered_items')
        
        # Referências genéricas
        contextual_words = ['isso', 'esta', 'essa', 'anterior', 'acima', 'mencionado', 
                           'lista', 'primeiro', 'segundo', 'terceiro', 'último']
        
        for word in contextual_words:
            if word in question.lower():
                refs.append(f'reference_{word}')
        
        return refs
    
    def _build_context_summary(self, 
                              context: ConversationContext, 
                              refs: List[str]) -> str:
        """Constrói resumo do contexto relevante"""
        
        if not context.turns:
            return ""
        
        context_parts = []
        
        # Se há referências a itens numerados, inclui listas
        if 'numbered_items' in refs:
            lists = context.extract_list_items()
            for list_name, items in lists.items():
                context_parts.append("Lista mencionada anteriormente:")
                for i, item in enumerate(items, 1):
                    context_parts.append(f"{i}. {item}")
                context_parts.append("")
        
        # Inclui últimas 2-3 interações para contexto geral
        recent_context = context.get_recent_context(max_turns=3)
        if recent_context:
            context_parts.append("Contexto da conversa:")
            context_parts.append(recent_context)
        
        return "\n".join(context_parts)
    
    def _enhance_question(self, question: str, context: str) -> str:
        """Enriquece a pergunta com contexto"""
        if not context:
            return question
        
        enhanced = f"""CONTEXTO DA CONVERSA:
{context}

PERGUNTA ATUAL: {question}

Por favor, responda considerando o contexto da conversa anterior, especialmente se a pergunta se refere a itens, listas ou informações mencionadas anteriormente."""
        
        return enhanced
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extrai entidades/conceitos mencionados no texto"""
        entities = []
        
        # Procura por listas numeradas
        numbered_items = re.findall(r'\d+\.\s+([^\n]+)', text)
        entities.extend([f"item_{i+1}_{item[:30]}" for i, item in enumerate(numbered_items)])
        
        # Procura por conceitos técnicos (palavras em maiúscula ou siglas)
        technical_terms = re.findall(r'\b[A-Z]{2,}\b|\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text)
        entities.extend(technical_terms)
        
        return list(set(entities))  # Remove duplicatas
    
    def _save_conversation(self, conversation_id: str):
        """Salva conversa em arquivo"""
        if conversation_id not in self.active_conversations:
            return
        
        context = self.active_conversations[conversation_id]
        file_path = self.storage_path / f"{conversation_id}.json"
        
        data = {
            'conversation_id': context.conversation_id,
            'user_id': context.user_id,
            'created_at': context.created_at.isoformat(),
            'last_activity': context.last_activity.isoformat(),
            'turns': [turn.to_dict() for turn in context.turns]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Carrega conversa do arquivo"""
        file_path = self.storage_path / f"{conversation_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            turns = [ConversationTurn.from_dict(turn_data) for turn_data in data['turns']]
            
            context = ConversationContext(
                conversation_id=data['conversation_id'],
                user_id=data['user_id'],
                turns=turns,
                created_at=datetime.fromisoformat(data['created_at']),
                last_activity=datetime.fromisoformat(data['last_activity'])
            )
            
            # Verifica se conversa não expirou
            if datetime.now() - context.last_activity < self.max_conversation_age:
                self.active_conversations[conversation_id] = context
                return context
            
        except Exception as e:
            print(f"Erro ao carregar conversa {conversation_id}: {e}")
        
        return None
    
    def cleanup_old_conversations(self):
        """Remove conversas antigas"""
        cutoff_time = datetime.now() - self.max_conversation_age
        
        # Remove da memória
        expired_ids = []
        for conv_id, context in self.active_conversations.items():
            if context.last_activity < cutoff_time:
                expired_ids.append(conv_id)
        
        for conv_id in expired_ids:
            del self.active_conversations[conv_id]
        
        # Remove arquivos antigos
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                last_activity = datetime.fromisoformat(data['last_activity'])
                
                if last_activity < cutoff_time:
                    file_path.unlink()
            except:
                continue

# Singleton global
_conversation_memory = None

def get_conversation_memory() -> ConversationMemory:
    """Retorna instância singleton do sistema de memória"""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory