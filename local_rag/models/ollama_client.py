"""
Cliente para interação com modelos Ollama locais
Sistema de LLM local para engenharia de reservatórios
"""

import requests
import json
from typing import Dict, List, Any, Optional, Generator, AsyncIterator
import time

class OllamaClient:
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2:3b",
                 timeout: int = 120):
        """
        Inicializa cliente Ollama
        
        Args:
            base_url: URL base do Ollama
            model_name: Nome do modelo a usar
            timeout: Timeout para requisições
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        
        # Verifica se Ollama está rodando
        if not self.is_ollama_running():
            print("⚠️  Ollama não está rodando. Inicie com: 'ollama serve'")
        else:
            print(f"✅ Ollama conectado: {base_url}")
            self.ensure_model_available()
    
    def is_ollama_running(self) -> bool:
        """
        Verifica se o Ollama está rodando
        """
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def ensure_model_available(self) -> bool:
        """
        Garante que o modelo está disponível, baixando se necessário
        """
        if self.is_model_available():
            print(f"✅ Modelo {self.model_name} já disponível")
            return True
        
        print(f"📥 Baixando modelo {self.model_name}...")
        return self.pull_model()
    
    def is_model_available(self) -> bool:
        """
        Verifica se o modelo está disponível localmente
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(model['name'] == self.model_name for model in models)
        except:
            pass
        return False
    
    def pull_model(self) -> bool:
        """
        Baixa o modelo do repositório Ollama
        """
        try:
            data = {"name": self.model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=data,
                timeout=600,  # 10 minutos para download
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        status = json.loads(line)
                        if 'status' in status:
                            print(f"Status: {status['status']}")
                print(f"✅ Modelo {self.model_name} baixado com sucesso")
                return True
            else:
                print(f"❌ Erro ao baixar modelo: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Erro ao baixar modelo: {e}")
            return False
    
    def generate_response(self, 
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         stream: bool = False) -> str:
        """
        Gera resposta usando o modelo local
        
        Args:
            prompt: Pergunta/prompt do usuário
            system_prompt: Prompt de sistema (contexto)
            max_tokens: Máximo de tokens na resposta
            temperature: Criatividade (0.0 = determinístico, 1.0 = criativo)
            stream: Se deve fazer streaming da resposta
        """
        try:
            # Monta mensagens
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Parâmetros da requisição
            data = {
                "model": self.model_name,
                "messages": messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["</response>", "[DONE]"]
                },
                "stream": stream
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=self.timeout,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    return self._handle_stream_response(response)
                else:
                    result = response.json()
                    return result.get('message', {}).get('content', '')
            else:
                return f"Erro na requisição: {response.status_code}"
        
        except Exception as e:
            return f"Erro ao gerar resposta: {e}"
    
    def _handle_stream_response(self, response) -> str:
        """
        Processa resposta em stream
        """
        full_response = ""
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        content = data['message']['content']
                        full_response += content
                        print(content, end='', flush=True)
                    
                    if data.get('done', False):
                        break
            print()  # Nova linha no final
        except Exception as e:
            print(f"\nErro no streaming: {e}")
        
        return full_response
    
    async def generate_response_stream(self, 
                                     prompt: str,
                                     system_prompt: Optional[str] = None,
                                     max_tokens: int = 1000,
                                     temperature: float = 0.7) -> AsyncIterator[str]:
        """
        Gera resposta em streaming assíncrono
        
        Args:
            prompt: Pergunta/prompt do usuário
            system_prompt: Prompt de sistema (contexto)
            max_tokens: Máximo de tokens na resposta
            temperature: Criatividade (0.0 = determinístico, 1.0 = criativo)
            
        Yields:
            str: Tokens da resposta conforme são gerados
        """
        import aiohttp
        import asyncio
        
        try:
            # Monta mensagens
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Parâmetros da requisição
            data = {
                "model": self.model_name,
                "messages": messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "stop": ["</response>", "[DONE]"]
                },
                "stream": True
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=data
                ) as response:
                    
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    line_str = line.decode('utf-8').strip()
                                    if line_str:
                                        data_chunk = json.loads(line_str)
                                        
                                        if 'message' in data_chunk and 'content' in data_chunk['message']:
                                            content = data_chunk['message']['content']
                                            if content:
                                                yield content
                                        
                                        if data_chunk.get('done', False):
                                            break
                                            
                                except json.JSONDecodeError:
                                    continue
                                except Exception as e:
                                    print(f"⚠️ Error processing chunk: {e}")
                                    continue
                    else:
                        yield f"Erro na requisição: {response.status}"
                        
        except Exception as e:
            yield f"Erro ao gerar resposta: {e}"
    
    def generate_with_context(self,
                            question: str,
                            context_documents: List[str],
                            max_context_length: int = 3000) -> str:
        """
        Gera resposta usando documentos como contexto (RAG)
        """
        # Monta contexto limitado
        context = self._build_context(context_documents, max_context_length)
        
        # System prompt especializado para engenharia de reservatórios
        system_prompt = self._get_reservoir_engineering_system_prompt()
        
        # Prompt com contexto
        full_prompt = f"""
Contexto dos documentos:
{context}

Pergunta: {question}

Responda e cite adequadamente as fontes. Se a informação não estiver nos documentos, informe que não há informação suficiente.
"""
        
        return self.generate_response(
            prompt=full_prompt,
            system_prompt=system_prompt,
            temperature=0.1  # Menos criatividade para respostas técnicas
        )
    
    def _build_context(self, documents: List[str], max_length: int) -> str:
        """
        Constrói contexto otimizado a partir dos documentos
        """
        context = ""
        current_length = 0
        
        for i, doc in enumerate(documents):
            doc_preview = f"[Documento {i+1}]\n{doc}\n\n"
            
            if current_length + len(doc_preview) > max_length:
                # Trunca documento se necessário
                remaining_space = max_length - current_length - 50
                if remaining_space > 100:
                    truncated_doc = doc[:remaining_space] + "..."
                    context += f"[Documento {i+1}]\n{truncated_doc}\n\n"
                break
            
            context += doc_preview
            current_length += len(doc_preview)
        
        return context
    
    def _get_reservoir_engineering_system_prompt(self) -> str:
        """
        Retorna system prompt especializado para engenharia de reservatórios
        """
        return """Você é um assistente especializado em Engenharia de Reservatórios e simulação de reservatórios.

Suas características:
- Especialista em modelagem e simulação de reservatórios
- Conhecimento profundo em propriedades de rochas e fluidos
- Experiência com simuladores como ECLIPSE, CMG, INTERSECT
- Foco em análises técnicas precisas e práticas

FORMATO DE RESPOSTA:
1. Responda de forma natural e direta, sem se apresentar
2. Use terminologia técnica apropriada
3. Use citações numeradas [1], [2], [3] para referenciar informações dos documentos
4. Forneça explicações práticas e aplicadas
5. Ao final da resposta, inclua uma seção "Referências:" listando:
   [1]: Autores, Título, Ano, Jornal/Conferência
   [2]: Autores, Título, Ano, Jornal/Conferência
6. Se não souber algo, diga que não encontrou informações suficientes nos documentos
7. Priorize precisão técnica sobre generalização

Áreas de expertise:
- Caracterização de reservatórios
- Modelagem PVT
- Análise de produção
- Simulação numérica
- Métodos de recuperação
- Análise de declínio
- Material balance"""
    
    def list_available_models(self) -> List[str]:
        """
        Lista modelos disponíveis localmente
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except:
            pass
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtém informações do modelo atual
        """
        try:
            data = {"name": self.model_name}
            response = requests.post(f"{self.base_url}/api/show", json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}


# Funções de conveniência
def create_ollama_client(model_name: str = "llama3.2:3b") -> OllamaClient:
    """
    Cria cliente Ollama com modelo padrão
    
    Modelos recomendados:
    - llama3.2:3b (rápido, ~2GB)
    - llama3.2:7b (equilibrado, ~4GB) 
    - codellama:7b (especializado em código)
    - mistral:7b (alternativa ao llama)
    """
    return OllamaClient(model_name=model_name)

def check_ollama_installation() -> Dict[str, Any]:
    """
    Verifica instalação e status do Ollama
    """
    client = OllamaClient()
    
    return {
        'ollama_running': client.is_ollama_running(),
        'available_models': client.list_available_models(),
        'recommended_models': [
            'llama3.2:3b',
            'llama3.2:7b', 
            'codellama:7b',
            'mistral:7b'
        ]
    }