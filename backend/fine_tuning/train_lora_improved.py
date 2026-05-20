#!/usr/bin/env python3
"""
Treinamento LoRA melhorado com modelo base adequado
"""

import os
import torch
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ImprovedLoRAConfig:
    """Configuração melhorada para treinamento LoRA"""
    
    # Model
    base_model: str = "microsoft/DialoGPT-medium"  # Modelo maior
    model_max_length: int = 512
    
    # LoRA settings
    lora_r: int = 8  # Rank otimizado
    lora_alpha: int = 16  # Alpha proporcional
    lora_dropout: float = 0.05  # Dropout menor
    target_modules: List[str] = None
    
    # Training
    num_train_epochs: int = 50  # Menos epochs, mais focado
    learning_rate: float = 1e-4  # LR mais alto
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Batch efetivo maior
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Validation and saving
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    save_total_limit: int = 5
    
    # Output
    output_dir: str = "./outputs/improved_lora_model"

class ImprovedLoRATrainer:
    """Trainer LoRA melhorado com configurações otimizadas"""
    
    def __init__(self, config: ImprovedLoRAConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # Dados de treinamento focados
        self.training_data = self._create_focused_training_data()
        
        # Target modules para DialoGPT-medium
        config.target_modules = ["c_attn", "c_proj"]
    
    def _create_focused_training_data(self) -> List[Dict[str, str]]:
        """Criar dados de treinamento focados e limpos"""
        return [
            {
                "instruction": "O que é engenharia de reservatórios?",
                "output": "Engenharia de reservatórios é a disciplina que estuda a caracterização e desenvolvimento de reservatórios de petróleo para maximizar a recuperação de hidrocarbonetos."
            },
            {
                "instruction": "Explique o método INSIM",
                "output": "INSIM é um método de simulação numérica que modela o fluxo entre poços usando modelos capacitivos-resistivos para eficiência computacional."
            },
            {
                "instruction": "Como funciona waterflooding?",
                "output": "Waterflooding é um método de recuperação secundária onde água é injetada para deslocar óleo em direção aos poços produtores."
            },
            {
                "instruction": "O que é simulação de reservatórios?",
                "output": "Simulação de reservatórios usa modelos matemáticos para prever o comportamento de fluidos em meios porosos subterrâneos."
            },
            {
                "instruction": "Defina permeabilidade relativa",
                "output": "Permeabilidade relativa quantifica a capacidade de escoamento de um fluído específico na presença de outros fluidos no meio poroso."
            },
            {
                "instruction": "O que são métodos EOR?",
                "output": "Métodos EOR são técnicas de recuperação avançada como injeção térmica, química ou miscível para extrair petróleo adicional."
            },
            {
                "instruction": "Como funciona INSIM-FT?",
                "output": "INSIM-FT combina eficiência do INSIM com front-tracking para cálculo mais preciso de saturações."
            },
            {
                "instruction": "O que é ajuste de histórico?",
                "output": "Ajuste de histórico calibra modelos de reservatório para reproduzir dados observados de produção e pressão."
            },
            {
                "instruction": "Defina fator de recuperação",
                "output": "Fator de recuperação é a fração do óleo original que pode ser extraída economicamente do reservatório."
            },
            {
                "instruction": "O que é análise nodal?",
                "output": "Análise nodal otimiza sistemas de produção identificando perdas de pressão em cada componente do sistema."
            }
        ]
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info(f"🚀 Setting up improved model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            model_max_length=self.config.model_max_length
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize token embeddings if needed
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("✅ Improved model setup complete")
    
    def prepare_dataset(self):
        """Prepare clean, focused dataset"""
        logger.info("📊 Preparing improved dataset...")
        
        # Format examples cleanly
        formatted_examples = []
        for example in self.training_data:
            text = f"Pergunta: {example['instruction']}\\nResposta: {example['output']}<|endoftext|>"
            formatted_examples.append({"text": text})
        
        # Duplicate for more training data
        formatted_examples = formatted_examples * 20  # 200 examples total
        
        logger.info(f"Created {len(formatted_examples)} training examples")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.model_max_length,
                return_overflowing_tokens=False,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        logger.info(f"Training samples: {len(split_dataset['train'])}")
        logger.info(f"Evaluation samples: {len(split_dataset['test'])}")
        
        return split_dataset["train"], split_dataset["test"]
    
    def train(self):
        """Execute improved training"""
        logger.info("🎯 Starting improved LoRA training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),
            bf16=False,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.config.save_total_limit,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        logger.info("🔥 Beginning improved training...")
        trainer.train()
        
        # Save final model
        final_output_dir = f"{self.config.output_dir}/final_improved_model"
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        logger.info(f"✅ Improved training completed! Final model: {final_output_dir}")
        
        # Test final model
        self._test_final_model(final_output_dir)
    
    def _test_final_model(self, model_path: str):
        """Test the final trained model"""
        logger.info("🧪 Testing final improved model...")
        
        try:
            # Test questions
            test_questions = [
                "O que é engenharia de reservatórios?",
                "Explique INSIM",
                "Como funciona waterflooding?"
            ]
            
            for question in test_questions:
                prompt = f"Pergunta: {question}\\nResposta:"
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
                
                with torch.no_grad():
                    outputs = self.peft_model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response.split("Resposta:")[-1].strip()
                
                logger.info(f"❓ {question}")
                logger.info(f"💬 {answer}")
                logger.info("-" * 50)
        
        except Exception as e:
            logger.error(f"Test failed: {e}")

def main():
    """Main function"""
    config = ImprovedLoRAConfig()
    trainer = ImprovedLoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()