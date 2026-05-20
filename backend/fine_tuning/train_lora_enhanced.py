#!/usr/bin/env python3
"""
Enhanced LoRA Training for IAbel - Standalone Version
Robust training with optimized hyperparameters and Portuguese support
"""

import os
import sys
import json
import torch
import logging
import warnings
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings("ignore")

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced training examples for reservoir engineering
ENHANCED_TRAINING_DATA = [
    {
        "instruction": "O que é engenharia de reservatórios?",
        "output": "A engenharia de reservatórios é a disciplina da engenharia de petróleo responsável por estudar e otimizar a extração de hidrocarbonetos dos reservatórios subterrâneos. Envolve caracterização de reservatórios, modelagem de fluxo de fluidos, análise de dados de produção e implementação de métodos de recuperação avançada para maximizar a recuperação de óleo e gás."
    },
    {
        "instruction": "Explique o método INSIM",
        "output": "INSIM (Interwell Numerical Simulation Model) é um método de simulação numérica interpoços que resolve equações de escoamento bifásico em uma malha de nós e arestas construída a partir da trajetória dos poços. Utiliza modelos capacitivos-resistivos para simular fluxo entre poços, oferecendo eficiência computacional para ajuste de histórico e predição de reservatórios."
    },
    {
        "instruction": "Como funciona waterflooding?",
        "output": "Waterflooding é um método de recuperação secundária onde água é injetada em poços específicos para deslocar óleo em direção aos poços produtores. O processo cria diferencial de pressão que força movimento dos fluidos através do meio poroso. A eficiência depende de permeabilidade relativa, molhabilidade da rocha e razão de mobilidade entre água e óleo."
    },
    {
        "instruction": "O que é simulação de reservatórios?",
        "output": "Simulação de reservatórios é uma técnica computacional que utiliza modelos matemáticos para representar comportamento de fluidos em meios porosos subterrâneos. Os simuladores resolvem equações de fluxo multifásico considerando propriedades da rocha e fluidos para prever produção de hidrocarbonetos e avaliar estratégias de desenvolvimento."
    },
    {
        "instruction": "Defina permeabilidade relativa",
        "output": "Permeabilidade relativa é uma propriedade adimensional que quantifica a capacidade de um fluido específico escoar através de meio poroso na presença de outros fluidos. É expressa como fração da permeabilidade absoluta e varia com saturação dos fluidos. As curvas de permeabilidade relativa são fundamentais para modelagem de fluxo multifásico."
    },
    {
        "instruction": "O que são métodos de recuperação avançada?",
        "output": "Métodos de recuperação avançada (EOR) são técnicas para extrair petróleo adicional após recuperação primária e secundária. Incluem métodos térmicos (vapor, combustão in-situ), químicos (polímeros, surfactantes) e miscíveis (CO2, nitrogênio). Visam alterar propriedades dos fluidos ou rocha para mobilizar óleo residual."
    },
    {
        "instruction": "Como funciona o INSIM-FT?",
        "output": "INSIM-FT é versão aprimorada do INSIM que utiliza front-tracking para calcular saturações. Combina eficiência computacional do INSIM com maior precisão no cálculo de saturação através do rastreamento de frentes. Permite modelar com fidelidade breakthrough de água e evolução de contatos fluido-fluido."
    },
    {
        "instruction": "O que é ajuste de histórico?",
        "output": "Ajuste de histórico é o processo de calibração de modelos de reservatório para reproduzir dados observados de produção, pressão e parâmetros dinâmicos. Envolve modificação sistemática de propriedades do modelo até que respostas simuladas coincidam com dados históricos, aumentando confiabilidade das predições."
    },
    {
        "instruction": "Defina fator de recuperação",
        "output": "Fator de recuperação é a fração do óleo original in-place que pode ser extraída do reservatório. Expressa em porcentagem, típicamente varia de 20-60% dependendo das características do reservatório e métodos de recuperação empregados. É influenciado por propriedades da rocha, fluidos e técnicas de produção."
    },
    {
        "instruction": "O que é análise nodal?",
        "output": "Análise nodal é metodologia para otimização de sistemas de produção de petróleo que divide o sistema em componentes e analisa perdas de pressão em cada segmento. Permite identificar gargalos e otimizar vazões de produção através do balanceamento entre curvas de inflow (reservatório) e outflow (sistema de produção)."
    }
]

@dataclass
class EnhancedLoRAConfig:
    """Enhanced LoRA configuration with optimized hyperparameters"""
    
    # Model settings - Portuguese-friendly model
    base_model: str = "microsoft/DialoGPT-small"  # Smaller, more trainable
    model_max_length: int = 1024
    
    # Enhanced LoRA parameters
    lora_r: int = 32  # Higher rank for better expressiveness
    lora_alpha: int = 64  # Alpha = 2 * rank for proper scaling
    lora_dropout: float = 0.05  # Low dropout for better learning
    target_modules: List[str] = None  # Will be set based on model
    
    # Robust training hyperparameters
    num_train_epochs: int = 50  # More epochs for thorough training
    per_device_train_batch_size: int = 4  # Conservative batch size
    gradient_accumulation_steps: int = 8  # Effective batch = 32
    
    # Optimized learning rate schedule
    learning_rate: float = 2e-5  # Conservative but effective
    warmup_ratio: float = 0.1  # 10% warmup
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    
    # Advanced settings
    max_grad_norm: float = 1.0
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 25
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Output
    output_dir: str = "./outputs/enhanced_lora_model"

class EnhancedLoRATrainer:
    """Enhanced LoRA trainer with robust configuration"""
    
    def __init__(self, config: EnhancedLoRAConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # Set target modules based on model
        if "DialoGPT" in config.base_model:
            config.target_modules = ["c_attn", "c_proj", "c_fc"]
        elif "gpt2" in config.base_model.lower():
            config.target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            config.target_modules = ["query", "key", "value", "dense"]
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info(f"🚀 Loading model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Enable gradient checkpointing for memory efficiency (disabled for stability)
        # self.model.gradient_checkpointing_enable()
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            fan_in_fan_out=True if "DialoGPT" in self.config.base_model else False
        )
        
        # Get PEFT model
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("✅ Model setup complete")
    
    def prepare_dataset(self):
        """Prepare training dataset"""
        logger.info("📊 Preparing dataset...")
        
        # Format training examples
        formatted_examples = []
        for example in ENHANCED_TRAINING_DATA:
            # Use consistent Portuguese format
            text = f"### Pergunta: {example['instruction']} ### Resposta: {example['output']}<|endoftext|>"
            formatted_examples.append({"text": text})
        
        # Duplicate data for more training
        formatted_examples = formatted_examples * 10  # 10x augmentation
        
        logger.info(f"Created {len(formatted_examples)} training examples")
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        # Tokenize function
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # Always pad to max length
                max_length=self.config.model_max_length,
                return_overflowing_tokens=False,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # Apply tokenization
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split into train/eval
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        logger.info(f"Training samples: {len(split_dataset['train'])}")
        logger.info(f"Evaluation samples: {len(split_dataset['test'])}")
        
        return split_dataset["train"], split_dataset["test"]
    
    def train(self):
        """Execute training"""
        logger.info("🎯 Starting enhanced LoRA training...")
        
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
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            fp16=False,  # Disable mixed precision to avoid issues
            dataloader_num_workers=0,  # Disable multiprocessing
            remove_unused_columns=False,
            report_to=[],  # Disable wandb logging
        )
        
        # Data collator with proper padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Improve efficiency
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            ]
        )
        
        # Train
        logger.info("🔥 Starting training...")
        trainer.train()
        
        # Save final model
        final_path = Path(self.config.output_dir) / "final_enhanced_model"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        
        logger.info(f"✅ Training completed! Model saved to: {final_path}")
        
        # Test the model
        self.test_model(str(final_path))
        
        return True
    
    def test_model(self, model_path: str):
        """Test the trained model"""
        logger.info("🧪 Testing trained model...")
        
        try:
            from peft import PeftModel
            
            # Load for testing
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
            
            test_questions = [
                "O que é engenharia de reservatórios?",
                "Explique o método INSIM",
                "Como funciona waterflooding?"
            ]
            
            for question in test_questions:
                prompt = f"### Pergunta: {question} ### Resposta:"
                
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "### Resposta:" in response:
                    answer = response.split("### Resposta:")[-1].strip()
                else:
                    answer = response[len(prompt):].strip()
                
                logger.info(f"❓ {question}")
                logger.info(f"💬 {answer}")
                logger.info("-" * 50)
        
        except Exception as e:
            logger.error(f"Error testing model: {e}")

def main():
    """Main training function"""
    logger.info("🚀 Enhanced LoRA Training for IAbel")
    logger.info("Configuration: 25 epochs, LR=2e-5, LoRA r=32/α=64")
    
    # Create configuration
    config = EnhancedLoRAConfig()
    
    # Create trainer
    trainer = EnhancedLoRATrainer(config)
    
    # Execute training
    try:
        success = trainer.train()
        if success:
            logger.info("🎉 Enhanced LoRA training completed successfully!")
        else:
            logger.error("❌ Training failed")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()