"""
IAbel LoRA Fine-tuning Module
Parameter-Efficient Fine-Tuning for Reservoir Engineering Domain Adaptation
"""

__version__ = "1.0.0"
__author__ = "IAbel Team"

from .data_processor import AcademicDataProcessor, DocumentChunk
from .lora_trainer import ReservoirEngineeringLoRATrainer, LoRATrainingConfig
from .hybrid_rag_lora import HybridRAGLoRASystem, HybridConfig, ResponseMode, HybridRAGService

__all__ = [
    "AcademicDataProcessor",
    "DocumentChunk", 
    "ReservoirEngineeringLoRATrainer",
    "LoRATrainingConfig",
    "HybridRAGLoRASystem",
    "HybridConfig",
    "ResponseMode",
    "HybridRAGService"
]