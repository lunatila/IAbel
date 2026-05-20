"""
Hybrid RAG + LoRA System for IAbel
Combines retrieval-augmented generation with domain-specific fine-tuned models
"""

import os
import json
import asyncio
import logging
import traceback
from typing import Dict, List, Optional, Any, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np
from transformers import AutoTokenizer
from peft import PeftModel

# Import existing RAG components
import sys
# Add the project root to Python path (backend is subdirectory of main project)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_rag.rag_system import LocalRAGSystem
from local_rag.models.ollama_client import OllamaClient
from fine_tuning.lora_trainer import ReservoirEngineeringLoRATrainer, LoRATrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseMode(Enum):
    """Response generation modes"""
    RAG_V1 = "rag_v1"
    RAG_V2 = "rag_v2"
    RAG_ONLY = "rag_only"  # Backward compatibility
    LORA_ONLY = "lora_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class HybridConfig:
    """Configuration for hybrid RAG + LoRA system"""
    
    # Model paths
    lora_model_path: str = "/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/outputs/ultra_stable_lora_extended/final_ultra_stable_model"
    base_model_name: str = "microsoft/DialoGPT-small"
    
    # RAG settings
    rag_enabled: bool = True
    top_k_documents: int = 5
    similarity_threshold: float = 0.3
    
    # LoRA settings
    lora_enabled: bool = True
    lora_temperature: float = 0.7
    lora_max_tokens: int = 512
    
    # Hybrid settings
    default_mode: ResponseMode = ResponseMode.ADAPTIVE
    rag_confidence_threshold: float = 0.6
    hybrid_weight_rag: float = 0.6
    hybrid_weight_lora: float = 0.4
    
    # Performance settings
    max_context_length: int = 2048
    device: str = "auto"

class HybridRAGLoRASystem:
    """Hybrid system combining RAG retrieval with LoRA fine-tuned responses"""
    
    def __init__(self, config: HybridConfig, rag_service=None):
        self.config = config
        self.rag_service = rag_service  # Use existing RAG service instead of creating new one
        self.rag_system = None
        self.lora_model = None
        self.lora_tokenizer = None
        self.is_initialized = False
        
        # Performance tracking
        self.stats = {
            "rag_responses": 0,
            "lora_responses": 0,
            "hybrid_responses": 0,
            "total_queries": 0
        }
    
    async def initialize(self):
        """Initialize both RAG and LoRA systems"""
        logger.info("Initializing Hybrid RAG + LoRA System...")
        
        # Initialize RAG system
        if self.config.rag_enabled:
            if self.rag_service:
                logger.info("Using existing RAG service...")
                self.rag_system = self.rag_service
                logger.info("RAG service initialized successfully")
            else:
                logger.info("Creating new LocalRAG system...")
                self.rag_system = LocalRAGSystem()
                logger.info("RAG system initialized successfully")
        
        # Initialize LoRA model
        if self.config.lora_enabled:
            logger.info("Initializing LoRA model...")
            await self._initialize_lora_model()
        
        self.is_initialized = True
        logger.info("Hybrid system initialization complete")
    
    async def _initialize_lora_model(self):
        """Initialize the LoRA fine-tuned model"""
        try:
            # Check if LoRA model exists
            model_path = Path(self.config.lora_model_path)
            if not model_path.exists():
                logger.warning(f"LoRA model not found at {model_path}. Disabling LoRA.")
                self.config.lora_enabled = False
                return
            
            # Load tokenizer
            self.lora_tokenizer = AutoTokenizer.from_pretrained(self.config.lora_model_path)
            
            # Set pad token if missing
            if self.lora_tokenizer.pad_token is None:
                self.lora_tokenizer.pad_token = self.lora_tokenizer.eos_token
            
            # Load base model with safer configuration
            from transformers import AutoModelForCausalLM
            try:
                # Try loading with float32 for stability on CPU
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Use float32 for stability
                    low_cpu_mem_usage=True,
                    device_map="cpu"  # Force CPU for stability
                )
                
                # Load LoRA adapter
                self.lora_model = PeftModel.from_pretrained(base_model, self.config.lora_model_path)
                self.lora_model.eval()  # Set to evaluation mode
                
                # Force CPU-only inference due to current model instability
                logger.warning("Using CPU-only inference for LoRA model due to CUDA instability")
                self.lora_model = self.lora_model.to("cpu")
                        
            except Exception as model_error:
                logger.error(f"LoRA model loading failed: {model_error}")
                logger.warning("Disabling LoRA model due to loading error")
                self.config.lora_enabled = False
                return
            
            logger.info("LoRA model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LoRA model: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            self.config.lora_enabled = False
    
    async def ask_question(self, 
                          question: str, 
                          conversation_id: Optional[str] = None,
                          mode: Optional[ResponseMode] = None) -> Dict[str, Any]:
        """Ask a question using the hybrid system"""
        
        if not self.is_initialized:
            await self.initialize()
        
        self.stats["total_queries"] += 1
        mode = mode or self.config.default_mode
        
        logger.info(f"Processing question: {question[:100]}... (mode: {mode.value})")
        
        # Route to appropriate method based on mode
        if mode in [ResponseMode.RAG_V1, ResponseMode.RAG_ONLY]:
            return await self._rag_v1_response(question, conversation_id)
        elif mode == ResponseMode.RAG_V2:
            return await self._rag_v2_response(question, conversation_id)
        elif mode == ResponseMode.LORA_ONLY:
            return await self._lora_only_response(question)
        elif mode == ResponseMode.HYBRID:
            return await self._hybrid_response(question, conversation_id)
        elif mode == ResponseMode.ADAPTIVE:
            return await self._adaptive_response(question, conversation_id)
        else:
            raise ValueError(f"Unknown response mode: {mode}")
    
    async def _rag_v1_response(self, question: str, conversation_id: Optional[str]) -> Dict[str, Any]:
        """Generate response using basic RAG system (v1)"""
        if not self.config.rag_enabled or not self.rag_system:
            raise ValueError("RAG system not available")
        
        self.stats["rag_responses"] += 1
        
        # Use basic RAG method to avoid enhanced features for v1
        if hasattr(self.rag_system, 'ask_question_basic'):
            logger.info("Using Basic RAG method for v1 response (no enhancements)")
            response = await self.rag_system.ask_question_basic(
                question=question,
                top_k=self.config.top_k_documents,
                include_sources=True
            )
        else:
            # Fallback to regular method
            logger.info("Using regular RAG method for v1 response")
            response = await self.rag_system.ask_question(
                question=question,
                top_k=self.config.top_k_documents,
                include_sources=True
            )
        
        response["generation_mode"] = "rag_v1"
        return response
    
    async def _rag_v2_response(self, question: str, conversation_id: Optional[str]) -> Dict[str, Any]:
        """Generate response using enhanced RAG system (v2)"""
        if not self.config.rag_enabled:
            raise ValueError("RAG system not available")
        
        self.stats["rag_responses"] += 1
        
        # Check if rag_system is an EnhancedRAGService with enhanced capabilities
        if hasattr(self.rag_system, 'enhanced_mode') and self.rag_system.enhanced_mode:
            # Use enhanced service directly - it will route to enhanced_ask_question internally
            logger.info("Using Enhanced RAG Service for v2 response")
            response = await self.rag_system.ask_question(
                question=question,
                top_k=self.config.top_k_documents,
                include_sources=True
            )
        elif hasattr(self.rag_system, 'rag_system') and hasattr(self.rag_system.rag_system, 'enhanced_ask_question'):
            # RAG service with enhanced system inside
            logger.info("Using Enhanced RAG System via service for v2 response")
            response = await self.rag_system.ask_question(
                question=question,
                top_k=self.config.top_k_documents,
                include_sources=True
            )
        elif hasattr(self.rag_system, 'enhanced_ask_question'):
            # Direct enhanced system
            logger.info("Using Enhanced RAG System directly for v2 response")
            response = await self.rag_system.enhanced_ask_question(
                question=question,
                top_k=self.config.top_k_documents,
                include_sources=True
            )
        else:
            # Fallback to basic RAG
            logger.warning("Enhanced RAG not available, falling back to basic RAG for v2")
            response = await self.rag_system.ask_question(
                question=question,
                top_k=self.config.top_k_documents,
                include_sources=True
            )
        
        response["generation_mode"] = "rag_v2"
        return response
    
    async def _lora_only_response(self, question: str) -> Dict[str, Any]:
        """Generate response using LoRA model only"""
        if not self.config.lora_enabled or not self.lora_model:
            raise ValueError("LoRA model not available")
        
        self.stats["lora_responses"] += 1
        
        # Generate response with LoRA
        response_text = await self._generate_lora_response(question)
        
        return {
            "answer": response_text,
            "confidence": 0.8,  # Default confidence for LoRA
            "conversation_id": "",
            "total_sources": 0,
            "sources": [],
            "timestamp": "",
            "generation_mode": "lora_only"
        }
    
    async def _hybrid_response(self, question: str, conversation_id: Optional[str]) -> Dict[str, Any]:
        """Generate response using both RAG and LoRA, then combine"""
        self.stats["hybrid_responses"] += 1
        
        # Get both responses
        rag_response = None
        lora_response = None
        
        # RAG response
        if self.config.rag_enabled and self.rag_system:
            try:
                rag_response = await self._rag_only_response(question, conversation_id)
            except Exception as e:
                logger.warning(f"RAG response failed: {e}")
        
        # LoRA response
        if self.config.lora_enabled and self.lora_model:
            try:
                lora_response = await self._lora_only_response(question)
            except Exception as e:
                logger.warning(f"LoRA response failed: {e}")
        
        # Combine responses
        return self._combine_responses(question, rag_response, lora_response)
    
    async def _adaptive_response(self, question: str, conversation_id: Optional[str]) -> Dict[str, Any]:
        """Adaptively choose between RAG and LoRA based on question characteristics"""
        
        # Analyze question to determine best approach
        question_type = self._analyze_question_type(question)
        
        logger.info(f"Question type detected: {question_type}")
        
        # Decision logic
        if question_type in ["definition", "factual"] and self.config.rag_enabled:
            # Use RAG for factual questions with good retrieval
            rag_response = await self._rag_only_response(question, conversation_id)
            
            # Check if RAG confidence is high enough
            if rag_response.get("confidence", 0) >= self.config.rag_confidence_threshold:
                rag_response["generation_mode"] = "adaptive_rag"
                return rag_response
        
        # For complex questions or low RAG confidence, try hybrid
        if self.config.lora_enabled and self.config.rag_enabled:
            hybrid_response = await self._hybrid_response(question, conversation_id)
            hybrid_response["generation_mode"] = "adaptive_hybrid"
            return hybrid_response
        elif self.config.lora_enabled:
            lora_response = await self._lora_only_response(question)
            lora_response["generation_mode"] = "adaptive_lora"
            return lora_response
        elif self.config.rag_enabled:
            rag_response = await self._rag_only_response(question, conversation_id)
            rag_response["generation_mode"] = "adaptive_rag_fallback"
            return rag_response
        else:
            raise ValueError("No generation method available")
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze question to determine the best generation approach"""
        question_lower = question.lower()
        
        # Definition questions
        if any(phrase in question_lower for phrase in 
               ["o que é", "what is", "defina", "define", "explique", "explain"]):
            return "definition"
        
        # How-to questions
        if any(phrase in question_lower for phrase in 
               ["como", "how", "processo", "process", "procedimento"]):
            return "procedural"
        
        # Comparison questions
        if any(phrase in question_lower for phrase in 
               ["diferença", "difference", "comparar", "compare", "vs", "versus"]):
            return "comparison"
        
        # Calculation/technical questions
        if any(phrase in question_lower for phrase in 
               ["calcular", "calculate", "formula", "equação", "equation"]):
            return "calculation"
        
        # Default to factual
        return "factual"
    
    async def _generate_lora_response(self, question: str, context: str = "") -> str:
        """Generate response using the LoRA fine-tuned model"""
        
        # Format prompt for LoRA model (matching training format)
        if context:
            prompt = f"### Pergunta: {question}\n### Contexto: {context} ### Resposta:"
        else:
            prompt = f"### Pergunta: {question} ### Resposta:"
        
        # Tokenize
        inputs = self.lora_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_context_length
        )
        
        # Force CPU inference for stability
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        # Generate with safer parameters
        with torch.no_grad():
            try:
                outputs = self.lora_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=min(self.config.lora_max_tokens, 150),
                    do_sample=True,
                    temperature=0.7,  # Balanced temperature
                    top_p=0.9,
                    top_k=50,  # Add top_k for diversity
                    repetition_penalty=1.2,  # Moderate penalty
                    pad_token_id=self.lora_tokenizer.eos_token_id,
                    eos_token_id=self.lora_tokenizer.eos_token_id,
                    bad_words_ids=[[self.lora_tokenizer.encode("!", add_special_tokens=False)[0]] * 5],  # Prevent excessive exclamation marks
                    early_stopping=True
                )
            except Exception as gen_error:
                logger.error(f"Generation error: {gen_error}")
                # Fallback to simple generation
                outputs = self.lora_model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=100,
                    do_sample=False,  # Deterministic fallback
                    pad_token_id=self.lora_tokenizer.eos_token_id,
                    eos_token_id=self.lora_tokenizer.eos_token_id
                )
        
        # Decode response
        full_response = self.lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        if "### Resposta:" in full_response:
            response = full_response.split("### Resposta:")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def _combine_responses(self, question: str, rag_response: Optional[Dict], lora_response: Optional[Dict]) -> Dict[str, Any]:
        """Intelligently combine RAG and LoRA responses"""
        
        # If only one response available, return it
        if rag_response and not lora_response:
            rag_response["generation_mode"] = "hybrid_rag_only"
            return rag_response
        elif lora_response and not rag_response:
            lora_response["generation_mode"] = "hybrid_lora_only"
            return lora_response
        elif not rag_response and not lora_response:
            return {
                "answer": "Desculpe, não foi possível gerar uma resposta.",
                "confidence": 0.0,
                "generation_mode": "hybrid_error"
            }
        
        # Both responses available - combine them
        rag_confidence = rag_response.get("confidence", 0.5)
        
        # Weighted combination based on confidence
        if rag_confidence >= 0.7:
            # High RAG confidence - prefer RAG but enhance with LoRA
            combined_answer = self._enhance_rag_with_lora(
                rag_response["answer"], 
                lora_response["answer"]
            )
            confidence = rag_confidence * 0.8 + 0.2  # Slight boost for enhancement
            sources = rag_response.get("sources", [])
            total_sources = rag_response.get("total_sources", 0)
        else:
            # Lower RAG confidence - blend both responses
            combined_answer = self._blend_responses(
                rag_response["answer"], 
                lora_response["answer"],
                rag_confidence
            )
            confidence = (rag_confidence + 0.8) / 2  # Average with LoRA confidence
            sources = rag_response.get("sources", [])[:3]  # Limit sources
            total_sources = min(rag_response.get("total_sources", 0), 3)
        
        return {
            "answer": combined_answer,
            "confidence": confidence,
            "conversation_id": rag_response.get("conversation_id", ""),
            "total_sources": total_sources,
            "sources": sources,
            "timestamp": rag_response.get("timestamp", ""),
            "generation_mode": "hybrid_combined",
            "rag_confidence": rag_confidence
        }
    
    def _enhance_rag_with_lora(self, rag_answer: str, lora_answer: str) -> str:
        """Enhance RAG answer with LoRA insights"""
        # Use RAG as primary but add LoRA insights if they add value
        rag_length = len(rag_answer)
        lora_length = len(lora_answer)
        
        if lora_length > rag_length * 0.3:  # LoRA provides substantial additional content
            return f"{rag_answer}\n\nAdicionalmente: {lora_answer[:200]}..."
        else:
            return rag_answer
    
    def _blend_responses(self, rag_answer: str, lora_answer: str, rag_confidence: float) -> str:
        """Blend RAG and LoRA responses based on confidence"""
        # Simple blending - can be made more sophisticated
        if rag_confidence > 0.5:
            return f"{rag_answer}\n\n{lora_answer[:150]}..."
        else:
            return f"{lora_answer}\n\nFontes documentais: {rag_answer[:150]}..."
    
    async def stream_response(self, question: str, 
                            conversation_id: Optional[str] = None,
                            mode: Optional[ResponseMode] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response generation for real-time UI updates"""
        
        # Start with typing indicator
        yield {
            "type": "status",
            "status": "processing",
            "message": "Analisando pergunta..."
        }
        
        try:
            # Get response
            response = await self.ask_question(question, conversation_id, mode)
            
            # Stream the response in chunks
            answer = response["answer"]
            chunk_size = 50  # Characters per chunk
            
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield {
                    "type": "chunk",
                    "content": chunk,
                    "is_final": False
                }
                
                # Small delay for smooth streaming
                await asyncio.sleep(0.1)
            
            # Final message with metadata
            yield {
                "type": "final",
                "content": "",
                "is_final": True,
                "confidence": response["confidence"],
                "sources": response.get("sources", []),
                "generation_mode": response.get("generation_mode", "unknown")
            }
            
        except Exception as e:
            yield {
                "type": "error", 
                "message": f"Erro na geração: {str(e)}"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            **self.stats,
            "rag_enabled": self.config.rag_enabled,
            "lora_enabled": self.config.lora_enabled,
            "is_initialized": self.is_initialized
        }

# Integration with existing FastAPI service
class HybridRAGService:
    """Service wrapper for FastAPI integration"""
    
    def __init__(self, config_path: Optional[str] = None, rag_service=None):
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = HybridConfig(**config_dict)
        else:
            self.config = HybridConfig()
        
        self.hybrid_system = HybridRAGLoRASystem(self.config, rag_service=rag_service)
    
    async def initialize(self):
        """Initialize the hybrid system"""
        await self.hybrid_system.initialize()
    
    async def ask_question(self, 
                          question: str, 
                          conversation_id: Optional[str] = None,
                          top_k: int = 5,
                          include_sources: bool = True,
                          mode: str = "adaptive") -> Dict[str, Any]:
        """Ask question with hybrid system"""
        
        # Convert mode string to enum
        try:
            response_mode = ResponseMode(mode)
        except ValueError:
            response_mode = ResponseMode.ADAPTIVE
        
        return await self.hybrid_system.ask_question(
            question=question,
            conversation_id=conversation_id,
            mode=response_mode
        )
    
    async def stream_response(self, question: str, 
                            conversation_id: Optional[str] = None,
                            mode: str = "adaptive") -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response generation"""
        
        try:
            response_mode = ResponseMode(mode)
        except ValueError:
            response_mode = ResponseMode.ADAPTIVE
        
        async for chunk in self.hybrid_system.stream_response(
            question=question,
            conversation_id=conversation_id,
            mode=response_mode
        ):
            yield chunk
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        return {
            "status": "healthy" if self.hybrid_system.is_initialized else "initializing",
            "config": {
                "rag_enabled": self.config.rag_enabled,
                "lora_enabled": self.config.lora_enabled,
                "default_mode": self.config.default_mode.value
            },
            "stats": self.hybrid_system.get_stats()
        }

async def main():
    """Test the hybrid system"""
    async def test_hybrid():
        # Initialize system
        service = HybridRAGService()
        await service.initialize()
        
        # Test questions
        questions = [
            "O que é INSIM?",
            "Como funciona waterflooding?",
            "Explique a diferença entre INSIM e INSIM-FT",
            "What is reservoir simulation?",
            "Como calcular a permeabilidade relativa?"
        ]
        
        for question in questions:
            print(f"\nPergunta: {question}")
            print("-" * 50)
            
            # Test adaptive mode
            response = await service.ask_question(question, mode="adaptive")
            print(f"Resposta ({response['generation_mode']}): {response['answer'][:200]}...")
            print(f"Confiança: {response['confidence']:.2f}")
            
            if response.get('sources'):
                print(f"Fontes: {len(response['sources'])}")
        
        # Print stats
        stats = service.get_system_status()
        print(f"\nEstatísticas: {stats['stats']}")
    
    # Run the test
    await test_hybrid()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())