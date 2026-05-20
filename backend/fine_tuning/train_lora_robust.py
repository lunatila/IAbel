#!/usr/bin/env python3
"""
Robust LoRA Fine-tuning for IAbel
Enhanced training with optimized hyperparameters, more epochs, and better model selection
"""

import os
import sys
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RobustLoRAConfig:
    """Enhanced configuration for robust LoRA training"""
    
    # Model selection - using Portuguese-friendly models
    base_model: str = "neuralmind/bert-base-portuguese-cased"  # Portuguese model
    # Alternative models to consider:
    # "microsoft/DialoGPT-small"  # Smaller, might train better
    # "gpt2"  # Classic choice for fine-tuning
    # "pierreguillou/gpt2-small-portuguese"  # Portuguese GPT-2
    
    model_max_length: int = 512  # Reduced for stability
    
    # Enhanced LoRA settings
    lora_r: int = 32  # Increased rank for better expressiveness
    lora_alpha: int = 64  # Higher alpha for stronger adaptation
    lora_dropout: float = 0.05  # Lower dropout for better learning
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "query", "key", "value", "dense"  # For BERT-based models
        # For GPT models: ["c_attn", "c_proj", "c_fc", "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    ])
    
    # Robust training hyperparameters
    num_train_epochs: int = 15  # Much more epochs for thorough training
    per_device_train_batch_size: int = 8  # Larger batch size
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size = 8*4 = 32
    
    # Optimized learning rate schedule
    learning_rate: float = 5e-5  # Conservative but effective
    warmup_ratio: float = 0.1  # 10% warmup
    lr_scheduler_type: str = "cosine"  # Cosine decay for better convergence
    weight_decay: float = 0.01  # Regularization
    
    # Advanced optimization
    optim: str = "adamw_torch"  # Standard AdamW
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01
    
    # Precision and performance
    fp16: bool = True  # Mixed precision for efficiency
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # Logging and monitoring
    logging_steps: int = 25
    report_to: str = "wandb"  # Weights & Biases tracking
    run_name: str = "robust_lora_iabel"
    
    # Data settings
    max_seq_length: int = 512
    train_split_ratio: float = 0.9
    
    # Output settings
    output_dir: str = "./outputs/robust_lora_model"
    
    # Training stability
    seed: int = 42
    data_seed: int = 42

class EnhancedDataProcessor:
    """Enhanced data processor for better training data quality"""
    
    @staticmethod
    def load_pdf_data(pdf_directory: str) -> List[Dict[str, Any]]:
        """Load and process PDF data with enhanced quality"""
        try:
            # Import data processor
            from data_processor import ReservoirEngineeringDataProcessor
            
            processor = ReservoirEngineeringDataProcessor()
            dataset = processor.process_pdfs_for_training(pdf_directory)
            
            logger.info(f"Loaded {len(dataset)} training examples from PDFs")
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading PDF data: {e}")
            return []
    
    @staticmethod
    def create_enhanced_prompts(examples: List[Dict]) -> List[Dict[str, str]]:
        """Create high-quality training prompts with consistent formatting"""
        enhanced_examples = []
        
        # Standard question templates for reservoir engineering
        question_templates = [
            "O que é {topic}?",
            "Explique {topic}",
            "Como funciona {topic}?",
            "Defina {topic}",
            "Qual a importância de {topic}?",
            "Descreva {topic}",
            "Quais são as características de {topic}?",
            "Como aplicar {topic}?",
            "Qual o objetivo de {topic}?"
        ]
        
        for example in examples:
            # Extract key terms for question generation
            content = example.get('output', '')
            
            if len(content) < 50:  # Skip very short content
                continue
                
            # Use original instruction or generate one
            instruction = example.get('instruction', '')
            if not instruction:
                # Try to extract main topic
                words = content.split()[:10]
                topic_candidates = [w for w in words if len(w) > 4 and w.istitle()]
                if topic_candidates:
                    topic = topic_candidates[0]
                    instruction = f"O que é {topic}?"
                else:
                    instruction = "Explique o conceito apresentado"
            
            # Format with consistent template
            formatted_example = {
                "text": f"### Pergunta: {instruction} ### Resposta: {content}<|endoftext|>",
                "instruction": instruction,
                "response": content,
                "source": example.get('source', 'unknown'),
                "quality_score": len(content) * 0.1  # Simple quality metric
            }
            
            enhanced_examples.append(formatted_example)
        
        # Sort by quality and take top examples
        enhanced_examples.sort(key=lambda x: x['quality_score'], reverse=True)
        
        logger.info(f"Created {len(enhanced_examples)} enhanced training examples")
        return enhanced_examples

class RobustLoRATrainer:
    """Enhanced LoRA trainer with robust training configuration"""
    
    def __init__(self, config: RobustLoRAConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with robust configuration"""
        logger.info(f"🚀 Setting up model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )
        
        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate dtype
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Prepare for training
        self.model.gradient_checkpointing_enable()
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            fan_in_fan_out=False  # Set based on model architecture
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("✅ Model and tokenizer setup complete")
    
    def prepare_datasets(self, pdf_directory: str):
        """Prepare training and evaluation datasets"""
        logger.info("📊 Preparing datasets...")
        
        # Load data
        processor = EnhancedDataProcessor()
        raw_data = processor.load_pdf_data(pdf_directory)
        
        if not raw_data:
            logger.error("No training data found!")
            return False
        
        # Enhance data quality
        enhanced_data = processor.create_enhanced_prompts(raw_data)
        
        # Split data
        split_idx = int(len(enhanced_data) * self.config.train_split_ratio)
        train_data = enhanced_data[:split_idx]
        eval_data = enhanced_data[split_idx:]
        
        logger.info(f"Training examples: {len(train_data)}")
        logger.info(f"Evaluation examples: {len(eval_data)}")
        
        # Tokenize datasets
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_overflowing_tokens=False,
            )
            
            # Set labels (for causal LM, labels = input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Create datasets
        self.train_dataset = Dataset.from_list(train_data)
        self.eval_dataset = Dataset.from_list(eval_data)
        
        # Apply tokenization
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        self.eval_dataset = self.eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names
        )
        
        logger.info("✅ Datasets prepared")
        return True
    
    def train(self, pdf_directory: str):
        """Execute robust training"""
        logger.info("🎯 Starting robust LoRA training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        
        if not self.prepare_datasets(pdf_directory):
            logger.error("Failed to prepare datasets")
            return False
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Optimizer settings
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            optim=self.config.optim,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            
            # Evaluation and saving
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Performance
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            
            # Logging
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            
            # Reproducibility
            seed=self.config.seed,
            data_seed=self.config.data_seed,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )
        
        # Start training
        logger.info("🔥 Beginning training...")
        trainer.train()
        
        # Save final model
        final_model_path = Path(self.config.output_dir) / "final_robust_model"
        trainer.save_model(str(final_model_path))
        self.tokenizer.save_pretrained(str(final_model_path))
        
        # Save training config
        config_path = final_model_path / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"✅ Training completed! Model saved to: {final_model_path}")
        
        # Test the model
        self.test_model(str(final_model_path))
        
        return True
    
    def test_model(self, model_path: str):
        """Test the trained model"""
        logger.info("🧪 Testing trained model...")
        
        try:
            from transformers import AutoTokenizer
            from peft import PeftModel
            
            # Load model for testing
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
            
            # Test questions
            test_questions = [
                "O que é engenharia de reservatórios?",
                "Explique o método INSIM",
                "Como funciona waterflooding?",
                "O que é simulação de reservatórios?"
            ]
            
            for question in test_questions:
                prompt = f"### Pergunta: {question} ### Resposta:"
                
                inputs = tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "### Resposta:" in response:
                    answer = response.split("### Resposta:")[-1].strip()
                else:
                    answer = response[len(prompt):].strip()
                
                logger.info(f"❓ {question}")
                logger.info(f"💬 {answer[:200]}...")
                logger.info("-" * 50)
        
        except Exception as e:
            logger.error(f"Error testing model: {e}")

def main():
    """Main training function"""
    
    # Initialize Weights & Biases
    try:
        wandb.init(
            project="iabel-robust-lora",
            name="robust_training_run",
            config={
                "epochs": 15,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "lora_r": 32,
                "lora_alpha": 64
            }
        )
    except:
        logger.warning("Could not initialize wandb")
    
    # Configuration
    config = RobustLoRAConfig()
    
    # PDF directory
    pdf_directory = "/home/lacucaratila/Projetos/IAbel/backend/data/pdfs"
    
    if not Path(pdf_directory).exists():
        logger.error(f"PDF directory not found: {pdf_directory}")
        return
    
    # Create trainer and start training
    trainer = RobustLoRATrainer(config)
    success = trainer.train(pdf_directory)
    
    if success:
        logger.info("🎉 Robust LoRA training completed successfully!")
    else:
        logger.error("❌ Training failed")

if __name__ == "__main__":
    main()