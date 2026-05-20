#!/usr/bin/env python3
"""
Portuguese-Optimized LoRA Training for IAbel
Using Portuguese-friendly models for better domain adaptation
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

# Import our enhanced training infrastructure
try:
    from train_lora_robust import RobustLoRAConfig, RobustLoRATrainer
except ImportError:
    logger.error("Could not import robust training classes")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Portuguese-optimized configurations
PORTUGUESE_CONFIGS = {
    "gpt2_portuguese": {
        "base_model": "pierreguillou/gpt2-small-portuguese",
        "description": "Portuguese GPT-2 Small - Best for Portuguese text generation",
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "fan_in_fan_out": True
    },
    
    "dialogpt_small": {
        "base_model": "microsoft/DialoGPT-small", 
        "description": "DialoGPT Small - Lighter, potentially better training",
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "fan_in_fan_out": True
    },
    
    "gpt2_multilingual": {
        "base_model": "gpt2",
        "description": "Original GPT-2 - Reliable baseline for fine-tuning",
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "fan_in_fan_out": False
    }
}

class PortugueseLoRAConfig(RobustLoRAConfig):
    """Portuguese-optimized LoRA configuration"""
    
    def __init__(self, model_name: str = "gpt2_portuguese"):
        super().__init__()
        
        # Get model config
        if model_name not in PORTUGUESE_CONFIGS:
            logger.warning(f"Unknown model {model_name}, using gpt2_portuguese")
            model_name = "gpt2_portuguese"
        
        model_config = PORTUGUESE_CONFIGS[model_name]
        
        # Update configuration
        self.base_model = model_config["base_model"]
        self.lora_target_modules = model_config["target_modules"]
        self.fan_in_fan_out = model_config["fan_in_fan_out"]
        
        # Enhanced hyperparameters for Portuguese
        self.num_train_epochs = 20  # More epochs for thorough learning
        self.learning_rate = 3e-5   # Conservative LR for stability
        self.per_device_train_batch_size = 6  # Balanced batch size
        self.gradient_accumulation_steps = 6   # Effective batch = 36
        
        # Improved LoRA parameters
        self.lora_r = 24  # Good balance between expressiveness and efficiency
        self.lora_alpha = 48  # 2x rank for proper scaling
        self.lora_dropout = 0.03  # Very low dropout for better learning
        
        # Advanced optimization
        self.warmup_ratio = 0.15  # Longer warmup for stability
        self.lr_scheduler_type = "cosine_with_restarts"
        self.weight_decay = 0.005  # Light regularization
        
        # Model-specific settings
        self.model_max_length = 1024  # Longer sequences for better context
        self.max_seq_length = 1024
        
        # Training stability
        self.early_stopping_patience = 8
        self.save_steps = 50
        self.eval_steps = 50
        
        # Update output directory
        self.output_dir = f"./outputs/portuguese_lora_{model_name}"
        self.run_name = f"portuguese_lora_{model_name}"
        
        logger.info(f"🇧🇷 Configured for {model_config['description']}")

def create_enhanced_training_data():
    """Create enhanced training data specifically for reservoir engineering"""
    
    # Enhanced training examples with diverse question patterns
    training_examples = [
        {
            "instruction": "O que é engenharia de reservatórios?",
            "output": "A engenharia de reservatórios é a área da engenharia de petróleo que se dedica ao estudo e desenvolvimento de técnicas para maximizar a extração de hidrocarbonetos dos reservatórios subterrâneos. Esta disciplina envolve a caracterização de reservatórios, modelagem de fluxo de fluidos, análise de dados de produção e implementação de métodos de recuperação avançada.",
            "source": "definition"
        },
        {
            "instruction": "Explique o método INSIM",
            "output": "INSIM (Interwell Numerical Simulation Model) é um método de simulação numérica interpoços baseado em resolver equações de escoamento bifásico em uma malha de nós e arestas construída a partir da trajetória dos poços. O método utiliza modelos capacitivos-resistivos para simular o fluxo de fluidos entre poços, oferecendo uma abordagem computacionalmente eficiente para ajuste de histórico e predição de desempenho de reservatórios.",
            "source": "insim_manual"
        },
        {
            "instruction": "Como funciona waterflooding?",
            "output": "Waterflooding é um método de recuperação secundária onde água é injetada em poços específicos para deslocar o óleo em direção aos poços produtores. O processo baseia-se na criação de um diferencial de pressão que força o movimento dos fluidos através do meio poroso. A eficiência do waterflooding depende da permeabilidade relativa, molhabilidade da rocha e razão de mobilidade entre água e óleo.",
            "source": "waterflooding_theory"
        },
        {
            "instruction": "O que é simulação de reservatórios?",
            "output": "A simulação de reservatórios é uma técnica computacional que utiliza modelos matemáticos para representar o comportamento de fluidos em meios porosos subterrâneos. Estes simuladores resolvem equações de fluxo multifásico considerando as propriedades da rocha e dos fluidos para prever a produção de hidrocarbonetos ao longo do tempo e avaliar diferentes estratégias de desenvolvimento.",
            "source": "simulation_basics"
        },
        {
            "instruction": "Defina permeabilidade relativa",
            "output": "Permeabilidade relativa é uma propriedade adimensional que quantifica a capacidade de um fluido específico escoar através de um meio poroso na presença de outros fluidos. É expressa como uma fração da permeabilidade absoluta e varia com a saturação dos fluidos presentes. As curvas de permeabilidade relativa são fundamentais para modelagem de fluxo multifásico em reservatórios.",
            "source": "rock_properties"
        },
        {
            "instruction": "O que são métodos de recuperação avançada?",
            "output": "Métodos de recuperação avançada (EOR - Enhanced Oil Recovery) são técnicas utilizadas para extrair petróleo adicional após a recuperação primária e secundária. Incluem métodos térmicos (injeção de vapor, combustão in-situ), químicos (injeção de polímeros, surfactantes) e miscíveis (injeção de CO2, nitrogênio). Estes métodos visam alterar as propriedades dos fluidos ou da rocha para mobilizar óleo residual.",
            "source": "eor_methods"
        },
        {
            "instruction": "Como funciona o INSIM-FT?",
            "output": "INSIM-FT é uma versão aprimorada do INSIM que utiliza front-tracking para calcular saturações. O método combina a eficiência computacional do INSIM com maior precisão no cálculo de saturação através do rastreamento de frentes de saturação. Isso permite modelar com maior fidelidade fenômenos como breakthrough de água e evolução de contatos fluido-fluido em reservatórios heterogêneos.",
            "source": "insim_ft_manual"
        },
        {
            "instruction": "O que é ajuste de histórico?",
            "output": "Ajuste de histórico é o processo de calibração de modelos de reservatório para reproduzir dados observados de produção, pressão e outros parâmetros dinâmicos. Este processo envolve a modificação sistemática de propriedades do modelo (permeabilidade, porosidade, transmissibilidades) até que as respostas simuladas coincidam adequadamente com os dados históricos, aumentando a confiabilidade das predições futuras.",
            "source": "history_matching"
        }
    ]
    
    return training_examples

def run_portuguese_training(model_name: str = "gpt2_portuguese"):
    """Run optimized training for Portuguese models"""
    
    logger.info(f"🚀 Starting Portuguese LoRA training with {model_name}")
    
    # Create configuration
    config = PortugueseLoRAConfig(model_name)
    
    # Create trainer
    trainer = RobustLoRATrainer(config)
    
    # Check for PDF directory
    pdf_directory = "/home/lacucaratila/Projetos/IAbel/backend/data/pdfs"
    
    if not Path(pdf_directory).exists():
        logger.warning(f"PDF directory not found: {pdf_directory}")
        logger.info("Creating synthetic training data...")
        
        # Create synthetic data directory
        synthetic_dir = Path("./synthetic_data")
        synthetic_dir.mkdir(exist_ok=True)
        
        # Save enhanced examples
        examples = create_enhanced_training_data()
        synthetic_file = synthetic_dir / "enhanced_examples.json"
        
        with open(synthetic_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Created {len(examples)} enhanced training examples")
        
        # For now, use a minimal training set
        pdf_directory = str(synthetic_dir)
    
    # Execute training
    try:
        success = trainer.train(pdf_directory)
        if success:
            logger.info("🎉 Portuguese LoRA training completed successfully!")
            return True
        else:
            logger.error("❌ Training failed")
            return False
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_models():
    """Train and compare different Portuguese models"""
    
    logger.info("🔬 Running comparative training across Portuguese models...")
    
    results = {}
    
    for model_name in PORTUGUESE_CONFIGS.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model: {model_name}")
        logger.info(f"Description: {PORTUGUESE_CONFIGS[model_name]['description']}")
        logger.info('='*60)
        
        try:
            success = run_portuguese_training(model_name)
            results[model_name] = "SUCCESS" if success else "FAILED"
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = f"ERROR: {str(e)}"
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🏆 TRAINING RESULTS SUMMARY:")
    logger.info("="*60)
    
    for model_name, result in results.items():
        status_emoji = "✅" if result == "SUCCESS" else "❌"
        logger.info(f"{status_emoji} {model_name}: {result}")
    
    return results

def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Portuguese LoRA Training")
    parser.add_argument("--model", default="gpt2_portuguese", 
                       choices=list(PORTUGUESE_CONFIGS.keys()),
                       help="Model to train")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all models")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        run_portuguese_training(args.model)

if __name__ == "__main__":
    main()