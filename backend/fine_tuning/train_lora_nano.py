#!/usr/bin/env python3
"""
LoRA Training Script - Nano Version
Bypasses all custom preprocessing for absolute minimal test
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_nano_lora():
    """Train LoRA with direct HuggingFace trainer - no custom preprocessing"""
    logger.info("🚀 Starting Nano LoRA Training (Direct HF Trainer)")
    
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    
    # Create minimal dataset
    logger.info("📝 Creating nano dataset...")
    texts = [
        "Pergunta: O que é INSIM? Resposta: INSIM é um método de ajuste de histórico.",
        "Pergunta: Como funciona? Resposta: Usa modelos capacitivos-resistivos.",
        "Pergunta: Engenharia de reservatórios? Resposta: Extração de hidrocarbonetos."
    ]
    
    # Load tokenizer and model
    logger.info("🔧 Loading tokenizer and model...")
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("🎮 Model moved to GPU")
    
    # Configure LoRA
    logger.info("⚙️ Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,                    # Very small rank
        lora_alpha=8,           # Small alpha
        lora_dropout=0.05,      # Low dropout
        target_modules=["c_attn"],  # Only attention
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info("✅ LoRA model configured!")
    
    # Tokenize texts directly
    logger.info("📊 Tokenizing texts...")
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,  # Very short
        return_tensors="pt"
    )
    
    # Create simple dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
        "labels": tokenized["input_ids"].tolist()  # Same as input for CLM
    })
    
    logger.info(f"📊 Dataset created with {len(dataset)} examples")
    
    # Training arguments
    logger.info("⚙️ Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./fine_tuning/outputs/nano_model",
        num_train_epochs=1,
        max_steps=2,                    # Only 2 steps!
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_steps=1,
        save_total_limit=1,
        report_to="none",
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Create trainer
    logger.info("🏋️ Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("🚀 Starting nano training...")
    logger.info("⏳ Should complete in seconds...")
    
    try:
        # Train
        trainer.train()
        logger.info("✅ Training completed!")
        
        # Save
        trainer.save_model("./fine_tuning/outputs/nano_lora_model")
        logger.info("💾 Model saved!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = train_nano_lora()
        
        if success:
            logger.info("🎉 Nano LoRA training completed!")
            logger.info("💡 System validated! LoRA fine-tuning works.")
        else:
            logger.error("❌ Nano training failed")
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