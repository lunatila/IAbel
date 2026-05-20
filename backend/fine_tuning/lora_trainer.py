"""
LoRA Fine-tuning Trainer for IAbel
Parameter-Efficient Fine-Tuning specialized for Reservoir Engineering
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training"""
    
    # Model settings
    base_model: str = "microsoft/DialoGPT-medium"  # Conversational base model
    model_max_length: int = 2048
    
    # LoRA specific settings
    lora_r: int = 16  # Rank of adaptation
    lora_alpha: int = 32  # LoRA scaling parameter
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "c_attn", "c_proj", "c_fc"  # Target modules for DialoGPT
    ])
    
    # Quantization settings (for memory efficiency)
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 100
    max_steps: int = 1000
    learning_rate: float = 2e-4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 25
    optim: str = "paged_adamw_32bit"
    
    # Data settings
    max_seq_length: int = 2048
    
    # Output settings
    output_dir: str = "./lora_outputs"
    save_steps: int = 250
    save_total_limit: int = 3

class ReservoirEngineeringLoRATrainer:
    """LoRA trainer specialized for reservoir engineering domain"""
    
    def __init__(self, config: LoRATrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with LoRA configuration"""
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Configure 4-bit quantization
        if self.config.use_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.use_4bit,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        
        # Move to GPU if available
        if torch.cuda.is_available() and not self.config.use_4bit:
            self.model = self.model.to("cuda")
        
        # Prepare model for k-bit training
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            fan_in_fan_out=True,  # Fix for DialoGPT Conv1D layers to avoid warning
        )
        
        # Get PEFT model
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("Model and tokenizer setup complete")
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset for instruction following training"""
        
        def format_instruction(examples):
            """Format data in instruction-following format"""
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            
            formatted_texts = []
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                # Create conversational format
                if input_text and input_text.strip():
                    prompt = f"### Instrução: {instruction}\n### Contexto: {input_text}\n### Resposta: {output}{self.tokenizer.eos_token}"
                else:
                    prompt = f"### Instrução: {instruction}\n### Resposta: {output}{self.tokenizer.eos_token}"
                
                formatted_texts.append(prompt)
            
            return {"text": formatted_texts}
        
        # Format the dataset
        formatted_dataset = dataset.map(
            format_instruction,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_overflowing_tokens=False,
            )
        
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names
        )
        
        return tokenized_dataset
    
    def create_data_collator(self):
        """Create data collator for language modeling"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Train the LoRA model"""
        logger.info("Starting LoRA training...")
        
        # Setup model if not done
        if self.peft_model is None:
            self.setup_model_and_tokenizer()
        
        # Preprocess datasets
        logger.info("📝 Preprocessing training dataset...")
        train_dataset = self.preprocess_dataset(train_dataset)
        logger.info("✅ Training dataset preprocessed!")
        
        if eval_dataset:
            logger.info("📝 Preprocessing evaluation dataset...")
            eval_dataset = self.preprocess_dataset(eval_dataset)
            logger.info("✅ Evaluation dataset preprocessed!")
        
        # Create data collator
        logger.info("📋 Creating data collator...")
        data_collator = self.create_data_collator()
        logger.info("✅ Data collator created!")
        
        # Training arguments
        logger.info("⚙️ Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            optim=self.config.optim,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=0.001,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_grad_norm=0.3,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            group_by_length=True,
            lr_scheduler_type="cosine",
            report_to="none",  # Disable wandb for now
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.save_steps if eval_dataset else None,
        )
        logger.info("✅ Training arguments configured!")
        
        # Create trainer
        logger.info("🏋️ Creating Hugging Face trainer...")
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        logger.info("✅ Trainer created successfully!")
        
        # Train
        logger.info("🚀 Starting actual training loop...")
        logger.info(f"📊 Will train for {self.config.max_steps} steps with {len(train_dataset)} examples")
        
        trainer.train()
        
        logger.info("🎉 Training loop completed successfully!")
        
        # Save the final model
        trainer.save_model()
        
        logger.info(f"Training complete. Model saved to {self.config.output_dir}")
    
    def save_model(self, save_path: str):
        """Save the trained LoRA adapter"""
        if self.peft_model is None:
            raise ValueError("No model to save. Train the model first.")
        
        self.peft_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_path = Path(save_path) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model and config saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load a trained LoRA adapter"""
        from peft import PeftModel
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            base_model = base_model.to("cuda")
        
        # Load PEFT model
        self.peft_model = PeftModel.from_pretrained(base_model, model_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def generate_response(self, instruction: str, context: str = "") -> str:
        """Generate response using the fine-tuned model"""
        if self.peft_model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Format prompt
        if context:
            prompt = f"### Instrução: {instruction}\n### Contexto: {context}\n### Resposta:"
        else:
            prompt = f"### Instrução: {instruction}\n### Resposta:"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        if "### Resposta:" in response:
            response = response.split("### Resposta:")[-1].strip()
        
        return response

class ReservoirEngineeringTrainingPipeline:
    """Complete training pipeline for reservoir engineering LoRA"""
    
    def __init__(self, 
                 pdf_directory: str,
                 output_directory: str = "./fine_tuning_outputs"):
        self.pdf_directory = pdf_directory
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.datasets_dir = self.output_directory / "datasets"
        self.models_dir = self.output_directory / "models"
        self.datasets_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting complete LoRA training pipeline...")
        
        # Step 1: Process PDFs and create dataset
        logger.info("Step 1: Processing PDFs...")
        from data_processor import AcademicDataProcessor
        
        processor = AcademicDataProcessor(
            chunk_size=1024,
            chunk_overlap=128,
            min_technical_density=0.2  # Lower threshold for more data
        )
        
        dataset = processor.process_pdf_directory(
            self.pdf_directory, 
            str(self.datasets_dir)
        )
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        
        # Step 2: Split dataset
        logger.info("Step 2: Splitting dataset...")
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        logger.info(f"Training examples: {len(train_dataset)}")
        logger.info(f"Evaluation examples: {len(eval_dataset)}")
        
        # Step 3: Configure and train LoRA
        logger.info("Step 3: Setting up LoRA training...")
        config = LoRATrainingConfig(
            output_dir=str(self.models_dir / "lora_checkpoint"),
            num_train_epochs=2,  # Start with fewer epochs
            max_steps=500,  # Limit steps for initial training
            per_device_train_batch_size=2,  # Small batch size for memory
            learning_rate=2e-4,
        )
        
        trainer = ReservoirEngineeringLoRATrainer(config)
        
        # Step 4: Train
        logger.info("Step 4: Training LoRA model...")
        trainer.train(train_dataset, eval_dataset)
        
        # Step 5: Save final model
        final_model_path = self.models_dir / "reservoir_engineering_lora"
        trainer.save_model(str(final_model_path))
        
        logger.info(f"Training complete! Model saved to {final_model_path}")
        
        # Step 6: Test the model
        logger.info("Step 6: Testing the trained model...")
        self._test_model(trainer)
        
        return trainer
    
    def _test_model(self, trainer: ReservoirEngineeringLoRATrainer):
        """Test the trained model with sample questions"""
        test_questions = [
            "O que é INSIM?",
            "Como funciona waterflooding?",
            "Explique o conceito de permeabilidade",
            "What is reservoir simulation?",
            "Describe the INSIM-FT method"
        ]
        
        logger.info("Testing trained model:")
        for question in test_questions:
            try:
                response = trainer.generate_response(question)
                logger.info(f"Q: {question}")
                logger.info(f"A: {response[:200]}...")
                logger.info("-" * 50)
            except Exception as e:
                logger.error(f"Error generating response for '{question}': {e}")

def main():
    """Main training script"""
    # Configure paths
    pdf_directory = "/home/lacucaratila/Projetos/IAbel/backend/data/pdfs"
    output_directory = "/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/outputs"
    
    # Run pipeline
    pipeline = ReservoirEngineeringTrainingPipeline(pdf_directory, output_directory)
    trainer = pipeline.run_complete_pipeline()
    
    print("LoRA training completed successfully!")

if __name__ == "__main__":
    main()