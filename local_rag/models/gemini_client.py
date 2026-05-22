"""
Cliente para interação com Google Gemini API
Sistema de LLM via API do Google para engenharia de reservatórios
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional

import requests
from pathlib import Path

logger = logging.getLogger(__name__)

_RETRYABLE_CODES = {429, 503, 502, 504}
_RATE_LIMIT_MSG = (
    "The API rate limit has been reached. Please wait a moment before sending another question. "
    "(Free tier: 15 requests/minute)"
)


class GeminiClient:
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.0-flash-exp",
                 timeout: int = 60):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')

        if not self.api_key:
            raise ValueError(
                "API Key do Google não encontrada. "
                "Configure a variável de ambiente GOOGLE_API_KEY ou GEMINI_API_KEY."
            )

        self.model_name = model_name
        self.timeout = timeout
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        logger.info("Google Gemini configurado: %s", model_name)

    def generate_response(self,
                          prompt: str,
                          system_prompt: Optional[str] = None,
                          max_tokens: int = 2000,
                          temperature: float = 0.3,
                          stream: bool = False) -> str:
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        last_error = None
        for attempt in range(3):
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return self._extract_text(response.json())
            if response.status_code in _RETRYABLE_CODES:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning("Gemini %s on attempt %d, retrying in %ds…", response.status_code, attempt + 1, wait)
                time.sleep(wait)
                last_error = response.status_code
                continue
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise RuntimeError(f"Gemini API error {response.status_code}: {detail}")

        raise RuntimeError(_RATE_LIMIT_MSG)

    def _extract_text(self, result: dict) -> str:
        """Extract generated text from a Gemini API response dict."""
        candidates = result.get("candidates", [])
        if not candidates:
            raise RuntimeError("Nenhuma resposta gerada pelo modelo")
        content = candidates[0].get("content", {})
        parts = content.get("parts", candidates[0].get("parts", []))
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
        if "text" in candidates[0]:
            return candidates[0]["text"].strip()
        raise RuntimeError(f"Formato de resposta não reconhecido: {candidates[0]}")

    def generate_with_context(self,
                              question: str,
                              context_documents: List[str] = None,
                              contexts: List[str] = None,
                              max_context_length: int = 3000,
                              max_tokens: int = 2000,
                              temperature: float = 0.3) -> str:
        ctx_list = context_documents or contexts or []
        context_text = "\n\n".join([
            f"[Documento {i+1}]\n{ctx}" for i, ctx in enumerate(ctx_list)
        ])

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
            temperature=temperature,
        )

    async def generate_response_stream(self,
                                       prompt: str,
                                       system_prompt: Optional[str] = None,
                                       max_tokens: int = 2000,
                                       temperature: float = 0.7):
        """
        Gera resposta em streaming real via SSE do endpoint streamGenerateContent.
        Yields text chunks as they arrive from the API.
        """
        import asyncio

        url = f"{self.base_url}/models/{self.model_name}:streamGenerateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.95,
                "topK": 40,
            }
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        last_status = None
        for attempt in range(3):
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key, "alt": "sse"},
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            if response.status_code in _RETRYABLE_CODES:
                wait = 2 ** attempt
                logger.warning("Gemini stream %s on attempt %d, retrying in %ds…", response.status_code, attempt + 1, wait)
                time.sleep(wait)
                last_status = response.status_code
                continue
            if response.status_code != 200:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise RuntimeError(f"Gemini API error {response.status_code}: {detail}")

            # Successful response — stream tokens
            try:
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line or not raw_line.startswith("data: "):
                        continue
                    json_str = raw_line[6:]
                    if json_str.strip() == "[DONE]":
                        return
                    try:
                        data = json.loads(json_str)
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text = part.get("text", "")
                                if text:
                                    yield text
                                    await asyncio.sleep(0)
                    except json.JSONDecodeError as e:
                        logger.debug("Skipping malformed SSE chunk: %s", e)
                        continue
            except Exception as e:
                logger.error("Gemini streaming failed: %s", e)
                raise
            return  # stream finished cleanly

        raise RuntimeError(_RATE_LIMIT_MSG)

    def is_ollama_running(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self.api_key is not None

    def test_connection(self) -> bool:
        try:
            response = self.generate_response(
                prompt="Diga 'OK' se você estiver funcionando.",
                max_tokens=10
            )
            logger.info("Teste de conexão bem-sucedido: %s", response)
            return True
        except Exception as e:
            logger.error("Falha no teste de conexão: %s", e)
            return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "status": "connected" if self.api_key else "not configured",
            "provider": "Google Gemini",
            "base_url": self.base_url,
        }


def create_gemini_client(model_name: str = "gemini-2.0-flash-exp") -> GeminiClient:
    return GeminiClient(model_name=model_name)
