"""
Cliente para interação com Google Gemini API
Sistema de LLM via API do Google para engenharia de reservatórios
"""

import os
import json
from typing import Dict, List, Any, Optional
import requests
from pathlib import Path

class GeminiClient:
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.0-flash-exp",
                 timeout: int = 60):
        """
        Inicializa cliente Google Gemini

        Args:
            api_key: Chave API do Google (ou None para usar variável de ambiente)
            model_name: Nome do modelo Gemini (ex: gemini-2.0-flash-exp, gemini-1.5-pro)
            timeout: Timeout para requisições em segundos
        """
        # Busca API key
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "API Key do Google não encontrada. "
                "Configure a variável de ambiente GOOGLE_API_KEY ou GEMINI_API_KEY, "
                "ou passe api_key no construtor."
            )

        self.model_name = model_name
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        print(f"✅ Google Gemini configurado: {model_name}")

    def generate_response(self,
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         max_tokens: int = 2000,
                         temperature: float = 0.3,
                         stream: bool = False) -> str:
        """
        Gera resposta usando Gemini API

        Args:
            prompt: Pergunta/prompt do usuário
            system_prompt: Instruções do sistema (contexto)
            max_tokens: Número máximo de tokens na resposta
            temperature: Temperatura da geração (0.0 = determinístico, 1.0 = criativo)
            stream: Se deve retornar resposta em streaming

        Returns:
            Resposta gerada pelo modelo
        """
        try:
            # Combina system prompt com user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Monta payload para API
            url = f"{self.base_url}/models/{self.model_name}:generateContent"

            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.95,
                    "topK": 40
                }
            }

            # Faz requisição
            headers = {
                "Content-Type": "application/json"
            }

            params = {
                "key": self.api_key
            }

            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                error_msg = f"Erro na API Gemini: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
                raise Exception(error_msg)

            # Extrai resposta
            result = response.json()

            if 'candidates' not in result or len(result['candidates']) == 0:
                raise Exception("Nenhuma resposta gerada pelo modelo")

            # Tenta extrair texto de diferentes formatos de resposta
            candidate = result['candidates'][0]

            # Formato 1: content.parts[0].text
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text'].strip()

            # Formato 2: parts[0].text (direto)
            if 'parts' in candidate:
                parts = candidate['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text'].strip()

            # Formato 3: text direto
            if 'text' in candidate:
                return candidate['text'].strip()

            raise Exception(f"Formato de resposta não reconhecido: {candidate}")

        except Exception as e:
            print(f"❌ Erro ao gerar resposta com Gemini: {e}")
            raise

    def generate_with_context(self,
                            question: str,
                            context_documents: List[str] = None,
                            contexts: List[str] = None,
                            max_context_length: int = 3000,
                            max_tokens: int = 2000,
                            temperature: float = 0.3) -> str:
        """
        Gera resposta com contextos do RAG

        Args:
            question: Pergunta do usuário
            context_documents: Lista de contextos relevantes do RAG (novo formato)
            contexts: Lista de contextos relevantes do RAG (formato legado)
            max_context_length: Tamanho máximo do contexto (ignorado - compatibilidade)
            max_tokens: Número máximo de tokens
            temperature: Temperatura da geração

        Returns:
            Resposta gerada
        """
        # Aceita ambos os formatos de contexto
        ctx_list = context_documents or contexts or []

        # Monta contexto formatado
        context_text = "\n\n".join([
            f"[Documento {i+1}]\n{ctx}"
            for i, ctx in enumerate(ctx_list)
        ])

        # System prompt especializado para engenharia de reservatórios
        system_prompt = f"""Você é um assistente especializado em Engenharia de Reservatórios de Petróleo.
Sua função é responder perguntas técnicas com base nos documentos fornecidos.

DOCUMENTOS DE REFERÊNCIA:
{context_text}

INSTRUÇÕES:
- Responda em português brasileiro de forma natural e direta, sem se apresentar
- Use informações APENAS dos documentos fornecidos
- Seja técnico e preciso
- Se a informação não estiver nos documentos, diga claramente
- Cite os documentos quando relevante
- Use terminologia técnica apropriada de engenharia de reservatórios
- Forneça explicações claras e detalhadas"""

        return self.generate_response(
            prompt=question,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def is_ollama_running(self) -> bool:
        """
        Compatibilidade com OllamaClient - sempre retorna True para Gemini
        """
        return True

    def is_available(self) -> bool:
        """
        Verifica se o cliente Gemini está disponível

        Returns:
            True se API key configurada
        """
        return self.api_key is not None

    def test_connection(self) -> bool:
        """
        Testa conexão com a API

        Returns:
            True se conectado com sucesso
        """
        try:
            response = self.generate_response(
                prompt="Diga 'OK' se você estiver funcionando.",
                max_tokens=10
            )
            print(f"✅ Teste de conexão bem-sucedido: {response}")
            return True
        except Exception as e:
            print(f"❌ Falha no teste de conexão: {e}")
            return False

    async def generate_response_stream(self,
                                prompt: str,
                                system_prompt: Optional[str] = None,
                                max_tokens: int = 2000,
                                temperature: float = 0.7):
        """
        Gera resposta em streaming (simulado para compatibilidade)

        Nota: Gemini API não suporta streaming nativo na versão atual,
        então retornamos a resposta completa de uma vez.

        Args:
            prompt: Prompt do usuário
            system_prompt: Prompt do sistema
            max_tokens: Número máximo de tokens
            temperature: Temperatura da geração

        Yields:
            Chunks da resposta
        """
        import asyncio

        # Gera resposta completa
        response = self.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Simula streaming dividindo em palavras
        words = response.split()
        chunk_size = 5  # Palavras por chunk

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += ' '
            yield chunk
            await asyncio.sleep(0)  # Yield control to event loop

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status do cliente

        Returns:
            Dicionário com informações de status
        """
        return {
            "model": self.model_name,
            "status": "connected" if self.api_key else "not configured",
            "provider": "Google Gemini",
            "base_url": self.base_url
        }


# Função helper para criar cliente facilmente
def create_gemini_client(model_name: str = "gemini-2.0-flash-exp") -> GeminiClient:
    """
    Cria e retorna um cliente Gemini configurado

    Args:
        model_name: Nome do modelo a usar

    Returns:
        Cliente Gemini configurado
    """
    return GeminiClient(model_name=model_name)
