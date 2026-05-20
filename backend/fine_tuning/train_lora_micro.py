#!/usr/bin/env python3
"""
LoRA Training Script - Micro Version
Absolute minimal memory usage for testing
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_micro_dataset():
    """Create the smallest possible dataset for testing"""
    from datasets import Dataset
    
    # Create minimal synthetic data for testing
    data = {
        "text": [
            "### Pergunta: O que é INSIM? ### Resposta: INSIM é um método de ajuste de histórico para reservatórios de petróleo.",
            "### Pergunta: Como funciona o INSIM? ### Resposta: O INSIM usa modelos capacitivos-resistivos para simular fluxo.",
            "### Pergunta: O que é LoRA? ### Resposta: LoRA é uma técnica de fine-tuning eficiente para modelos de linguagem.",
            "### Pergunta: Engenharia de reservatórios? ### Resposta: É a área que estuda extração de hidrocarbonetos.",
            "### Pergunta: O que é PEFT? ### Resposta: Parameter-Efficient Fine-Tuning para otimizar modelos grandes."
        ]
    }
    
    logger.info(f"📊 Created micro dataset with {len(data['text'])} examples")
    return Dataset.from_dict(data)

def train_micro_lora():
    """Train LoRA with absolute minimal configuration"""
    logger.info("🚀 Starting Micro LoRA Training (Synthetic Data)")
    
    # Import here to avoid issues
    from lora_trainer import LoRATrainingConfig, ReservoirEngineeringLoRATrainer
    
    # Create tiny synthetic dataset
    logger.info("📝 Creating micro synthetic dataset...")
    dataset = create_micro_dataset()
    logger.info("✅ Micro dataset created!")
    
    # Split dataset (4 train, 1 eval)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    logger.info(f"📊 Training examples: {len(train_dataset)}")
    logger.info(f"📊 Evaluation examples: {len(eval_dataset)}")
    
    # Ultra-minimal training configuration
    config = LoRATrainingConfig(
        output_dir="./fine_tuning/outputs/micro_model",
        num_train_epochs=1,           # Just 1 epoch
        max_steps=3,                  # Only 3 steps for proof-of-concept
        per_device_train_batch_size=1,  # Minimal batch
        gradient_accumulation_steps=1,  # No accumulation
        learning_rate=1e-3,           # Higher LR for fast convergence
        fp16=False,                   # Disable fp16
        bf16=False,                   # Disable bf16
        use_4bit=False,               # No quantization
        save_steps=1,                 # Save every step
        logging_steps=1,              # Log every step
        lora_r=4,                     # Very small LoRA rank
        lora_alpha=8,                 # Small alpha
        warmup_steps=0,               # No warmup
        save_total_limit=1            # Keep only 1 checkpoint
    )
    
    logger.info(f"📊 Micro Training configuration:")
    logger.info(f"  Max steps: {config.max_steps} (micro mode)")
    logger.info(f"  Dataset size: {len(train_dataset)} examples")
    logger.info(f"  LoRA rank: {config.lora_r}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Memory optimization: Maximum")
    
    # Create trainer
    logger.info("🔧 Creating micro LoRA trainer...")
    trainer = ReservoirEngineeringLoRATrainer(config)
    logger.info("✅ Micro trainer created!")
    
    # Train
    logger.info("🎯 Starting micro training...")
    logger.info("⏳ This should complete in under 1 minute...")
    
    try:
        trainer.train(train_dataset, eval_dataset)
        logger.info("✅ Micro training completed successfully!")
        
        # Save model
        model_path = "./fine_tuning/outputs/micro_lora_model"
        trainer.save_model(model_path)
        logger.info(f"💾 Micro model saved to: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Micro training failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = train_micro_lora()
        
        if success:
            logger.info("🎉 Micro LoRA training test completed!")
            logger.info("💡 This proves the LoRA system works. Scale up gradually for real training.")
        else:
            logger.error("❌ Micro training failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())