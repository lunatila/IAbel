#!/usr/bin/env python3
"""
LoRA Training Script - Light Version
Processes only smaller PDFs to avoid memory issues
"""

import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_small_pdfs(pdf_directory: str, max_size_mb: float = 0.5):
    """Filter PDFs by size - use only the smallest PDF for ultra-fast testing"""
    pdf_dir = Path(pdf_directory)
    all_pdfs = list(pdf_dir.glob("*.pdf"))
    
    small_pdfs = []
    large_pdfs = []
    
    for pdf_file in all_pdfs:
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        if size_mb <= max_size_mb:
            small_pdfs.append((pdf_file, size_mb))
        else:
            large_pdfs.append((pdf_file, size_mb))
    
    # Sort by size and take only the smallest one for ultra-fast testing
    small_pdfs.sort(key=lambda x: x[1])
    if small_pdfs:
        small_pdfs = [small_pdfs[0]]  # Use only the smallest PDF
    
    logger.info(f"PDF Analysis (Ultra-Fast Mode):")
    logger.info(f"  Tiny PDFs (≤{max_size_mb}MB): {len(small_pdfs)}")
    logger.info(f"  Larger PDFs (>{max_size_mb}MB): {len(large_pdfs)}")
    
    # Show files
    if small_pdfs:
        logger.info("Tiny PDF to process (for fast testing):")
        for pdf, size in small_pdfs:
            logger.info(f"  📄 {pdf.name}: {size:.1f}MB")
    
    if large_pdfs:
        logger.info(f"Skipping {len(large_pdfs)} larger PDFs for fast testing")
    
    return [pdf for pdf, _ in small_pdfs]

def process_pdfs_streaming(pdf_files, processor, max_chunks_per_pdf=20):
    """Process PDFs with streaming to avoid memory issues and show progress"""
    all_examples = []
    
    logger.info(f"🔄 Starting streaming PDF processing...")
    
    # Process each PDF with progress bar
    for pdf_file in tqdm(pdf_files, desc="📄 Processing PDFs", unit="pdf"):
        try:
            logger.info(f"📖 Processing {pdf_file.name}...")
            
            # Extract chunks from PDF
            chunks = processor.extract_text_from_pdf(str(pdf_file))
            
            # Limit chunks to avoid memory issues
            chunks = chunks[:max_chunks_per_pdf]
            
            # Process each chunk individually with progress
            for chunk in tqdm(chunks, desc=f"📝 {pdf_file.name[:20]}...", leave=False, unit="chunk"):
                # Create instruction example from chunk
                chunk_text = getattr(chunk, 'text', getattr(chunk, 'content', ''))
                if chunk_text and chunk_text.strip() and len(chunk_text) > 50:
                    example = {
                        'instruction': f"Explique sobre: {chunk_text[:100]}...",
                        'input': chunk_text[:500],  # Limit input size
                        'output': f"Com base no documento: {chunk_text[:200]}..."
                    }
                    all_examples.append(example)
            
            logger.info(f"✅ Processed {len(chunks)} chunks from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_file.name}: {e}")
            continue
    
    logger.info(f"📊 Total examples created: {len(all_examples)}")
    return all_examples

def create_dataset_lightweight(examples):
    """Create dataset without heavy preprocessing to avoid OOM"""
    from datasets import Dataset
    
    logger.info("📋 Creating lightweight dataset...")
    
    # Simple format - no complex preprocessing
    formatted_examples = []
    
    for example in tqdm(examples, desc="📝 Formatting examples", unit="example"):
        # Simple instruction format
        text = f"### Pergunta: {example['instruction']}\n### Resposta: {example['output']}"
        formatted_examples.append({"text": text})
    
    # Create dataset directly
    dataset = Dataset.from_list(formatted_examples)
    logger.info(f"✅ Dataset created with {len(dataset)} examples")
    
    return dataset

class LightweightLoRATrainer:
    """Lightweight LoRA trainer that avoids heavy preprocessing"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with progress feedback"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        import torch
        
        logger.info("🔧 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("✅ Tokenizer loaded!")
        
        logger.info("🔧 Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.config.base_model)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            logger.info("🎮 Model moved to GPU!")
        
        logger.info("⚙️ Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            fan_in_fan_out=True,  # Fix for DialoGPT Conv1D layers
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        logger.info("✅ LoRA model configured!")
        
    def tokenize_dataset_direct(self, dataset):
        """Direct tokenization without heavy map operations"""
        logger.info("🔤 Starting direct tokenization...")
        
        tokenized_examples = []
        
        for example in tqdm(dataset, desc="🔤 Tokenizing", unit="example"):
            # Tokenize directly
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                padding=False,
                max_length=self.config.max_seq_length,
                return_tensors=None,  # Return lists, not tensors
            )
            
            # Add labels (same as input_ids for causal LM)
            tokens["labels"] = tokens["input_ids"].copy()
            tokenized_examples.append(tokens)
        
        # Convert back to dataset
        from datasets import Dataset
        tokenized_dataset = Dataset.from_list(tokenized_examples)
        
        logger.info(f"✅ Tokenization complete! {len(tokenized_dataset)} examples")
        return tokenized_dataset
        
    def train(self, train_dataset, eval_dataset=None):
        """Train with lightweight approach"""
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        if self.model is None:
            self.setup_model_and_tokenizer()
        
        # Direct tokenization
        logger.info("📝 Processing training dataset...")
        train_dataset = self.tokenize_dataset_direct(train_dataset)
        
        if eval_dataset:
            logger.info("📝 Processing evaluation dataset...")
            eval_dataset = self.tokenize_dataset_direct(eval_dataset)
        
        # Training arguments
        logger.info("⚙️ Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,    # Avoid multiprocessing overhead
            dataloader_pin_memory=False, # Save GPU memory
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        logger.info("🏋️ Creating Hugging Face trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train with progress
        logger.info("🚀 Starting training loop...")
        logger.info(f"📊 Training for {self.config.max_steps} steps")
        
        trainer.train()
        
        logger.info("🎉 Training completed!")
        return trainer
        
    def save_model(self, path):
        """Save the trained model"""
        if self.model:
            self.model.save_pretrained(path)
            logger.info(f"💾 Model saved to: {path}")

def train_with_small_pdfs():
    """Train LoRA model using only smaller PDFs"""
    logger.info("🚀 Starting Ultra-Fast LoRA Training (Single Tiny PDF Test)")
    
    # Import here to avoid issues
    from data_processor import AcademicDataProcessor
    from lora_trainer import LoRATrainingConfig, ReservoirEngineeringLoRATrainer
    
    # Configuration optimized for ultra-fast testing
    processor = AcademicDataProcessor(
        chunk_size=128,           # Very small chunks for speed
        chunk_overlap=32,         # Minimal overlap  
        min_technical_density=0.05 # Very low threshold for speed
    )
    
    # Filter small PDFs - use only the tiniest for ultra-fast testing  
    pdf_directory = "/home/lacucaratila/Projetos/IAbel/backend/data/pdfs"
    small_pdfs = filter_small_pdfs(pdf_directory, max_size_mb=1.0)  # Increased to 1MB for more options
    
    if not small_pdfs:
        logger.error("No small PDFs found for training!")
        return False
    
    # Use streaming processing with progress bars
    examples = process_pdfs_streaming(small_pdfs, processor, max_chunks_per_pdf=15)
    
    if not examples:
        logger.error("No examples created!")
        return False
    
    # Create lightweight dataset (avoiding heavy preprocessing)
    dataset = create_dataset_lightweight(examples)
    
    if len(dataset) < 5:
        logger.error("Too few examples for training!")
        return False
    
    # Split dataset
    logger.info("📊 Splitting dataset...")
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    logger.info(f"📊 Training examples: {len(train_dataset)}")
    logger.info(f"📊 Evaluation examples: {len(eval_dataset)}")
    
    # Configure ultra-fast training for testing
    config = LoRATrainingConfig(
        output_dir="./fine_tuning/outputs/ultra_fast_model",
        num_train_epochs=1,           # Just 1 epoch for speed
        max_steps=15,                 # Increased for better training with real PDFs
        per_device_train_batch_size=1,  # Keep small batch
        gradient_accumulation_steps=1,  # Minimal accumulation for speed
        learning_rate=5e-4,           # Higher LR for faster convergence
        fp16=False,                   # Disable fp16 to save memory
        bf16=False,                   # Disable bf16 too
        use_4bit=False,               # Disable 4-bit to avoid bitsandbytes issues
        save_steps=5,                 # Save frequently for feedback
        logging_steps=2,              # Log very frequently
        max_seq_length=256,           # Shorter sequences to save memory
        lora_r=8,                     # Slightly larger LoRA rank
        lora_alpha=16                 # Corresponding alpha
    )
    
    logger.info(f"📊 Ultra-Fast Training configuration:")
    logger.info(f"  Max steps: {config.max_steps} (ultra-fast mode)")
    logger.info(f"  Epochs: {config.num_train_epochs}")
    logger.info(f"  Batch size: {config.per_device_train_batch_size}")
    logger.info(f"  Accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(f"  4-bit quantization: {config.use_4bit}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    # Create lightweight trainer (avoids heavy preprocessing)
    logger.info("🔧 Creating lightweight LoRA trainer...")
    trainer = LightweightLoRATrainer(config)
    logger.info("✅ Lightweight trainer created!")
    
    # Train
    logger.info("🎯 Starting training process...")
    logger.info("⏳ Training with real PDF data and progress bars...")
    try:
        trainer.train(train_dataset, eval_dataset)
        logger.info("✅ Training completed successfully!")
        
        # Save model
        model_path = "./fine_tuning/outputs/ultra_fast_lora_model"
        trainer.save_model(model_path)
        logger.info(f"💾 Ultra-fast model saved to: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(f"Error details: {traceback.format_exc()}")
        return False

def main():
    """Main function"""
    try:
        success = train_with_small_pdfs()
        
        if success:
            logger.info("🎉 Ultra-Fast LoRA training test completed!")
            logger.info("💡 This was a minimal test. For full training, use more PDFs and higher max_steps")
        else:
            logger.error("❌ Training failed")
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