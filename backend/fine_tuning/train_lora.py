#!/usr/bin/env python3
"""
LoRA Training Script for IAbel
Automated training pipeline for reservoir engineering domain adaptation
"""

import os
import sys
import subprocess
import argparse
import logging
import traceback
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('lora_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_pdfs_streaming(pdf_files, processor, max_chunks_per_pdf=50):
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
                        'input': chunk_text[:800],  # Larger input for production
                        'output': f"Com base no documento: {chunk_text[:400]}..."
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

class ProductionLoRATrainer:
    """Production LoRA trainer with optimized preprocessing"""
    
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
            save_total_limit=3,          # Keep only 3 checkpoints
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

def setup_environment():
    """Setup environment and check dependencies"""
    logger.info("Setting up training environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ required for LoRA training")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("No CUDA GPU detected. Training will be slower on CPU.")
    except ImportError:
        logger.error("PyTorch not installed. Please install requirements first.")
        sys.exit(1)
    
    # Check and install critical packages
    required_packages = {
        'transformers': 'transformers==4.44.2',
        'peft': 'peft==0.7.1', 
        'datasets': 'datasets==2.16.1',
        'accelerate': 'accelerate==0.25.0',
        'PyMuPDF': 'PyMuPDF==1.23.26',
        'pdfplumber': 'pdfplumber==0.10.3'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'PyMuPDF':
                import PyMuPDF
            else:
                __import__(package)
            logger.info(f"✅ {package} available")
        except ImportError:
            missing_packages.append(install_name)
            logger.warning(f"⚠️ {package} missing")
    
    # Auto-install missing packages
    if missing_packages:
        logger.info(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                logger.info(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True, text=True)
                logger.info(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                logger.error(f"Stderr: {e.stderr}")
                logger.error("Please install manually: pip install requirements.txt")
                sys.exit(1)
    
    logger.info("Environment setup complete")

def train_lora_model(pdf_directory: str, 
                    output_directory: str,
                    model_name: str = "microsoft/DialoGPT-medium",
                    epochs: int = 2,
                    batch_size: int = 2,
                    learning_rate: float = 2e-4,
                    max_steps: int = 500):
    """Train LoRA model with optimized approach - no OOM issues"""
    
    logger.info("🚀 Starting Production LoRA Training...")
    logger.info(f"📂 PDF directory: {pdf_directory}")
    logger.info(f"📁 Output directory: {output_directory}")
    logger.info(f"🤖 Base model: {model_name}")
    logger.info(f"⚙️ Config: {epochs} epochs, batch={batch_size}, lr={learning_rate}, steps={max_steps}")
    
    try:
        from data_processor import AcademicDataProcessor
        from lora_trainer import LoRATrainingConfig
        from pathlib import Path
        
        # Setup processor
        processor = AcademicDataProcessor(
            chunk_size=512,           # Production size
            chunk_overlap=64,         
            min_technical_density=0.2  # Higher quality threshold
        )
        
        # Filter PDFs for production (larger than light version)
        logger.info("📊 Analyzing PDFs...")
        pdf_dir = Path(pdf_directory)
        all_pdfs = list(pdf_dir.glob("*.pdf"))
        
        # Use PDFs up to 5MB for production
        production_pdfs = []
        for pdf_file in all_pdfs:
            size_mb = pdf_file.stat().st_size / (1024 * 1024)
            if size_mb <= 5.0:  # 5MB limit for production
                production_pdfs.append(pdf_file)
        
        logger.info(f"📄 Found {len(all_pdfs)} PDFs total")
        logger.info(f"📄 Using {len(production_pdfs)} PDFs (≤5MB) for training")
        
        if not production_pdfs:
            logger.error("❌ No suitable PDFs found for training!")
            return False
        
        # Process PDFs with streaming and progress bars
        examples = process_pdfs_streaming(production_pdfs, processor, max_chunks_per_pdf=50)
        
        if not examples:
            logger.error("❌ No examples created from PDFs!")
            return False
        
        logger.info(f"📊 Created {len(examples)} training examples")
        
        # Create lightweight dataset
        dataset = create_dataset_lightweight(examples)
        
        if len(dataset) < 10:
            logger.error("❌ Too few examples for meaningful training!")
            return False
        
        # Split dataset
        logger.info("📊 Splitting dataset...")
        train_test = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        logger.info(f"📊 Training examples: {len(train_dataset)}")
        logger.info(f"📊 Evaluation examples: {len(eval_dataset)}")
        
        # Production LoRA configuration
        config = LoRATrainingConfig(
            base_model=model_name,
            output_dir=output_directory,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,      # More accumulation for stability
            learning_rate=learning_rate,
            fp16=False,                         # Disable to avoid issues
            bf16=False,                         # Disable to avoid issues  
            use_4bit=False,                     # Disable quantization
            max_seq_length=512,                 # Production sequence length
            lora_r=16,                          # Production LoRA rank
            lora_alpha=32,                      # Production alpha
            lora_dropout=0.1,                   # Production dropout
            save_steps=50,                      # Save checkpoints frequently
            logging_steps=10,                   # Log frequently
            save_total_limit=3,                 # Keep 3 checkpoints
        )
        
        logger.info("📊 Production Training Configuration:")
        logger.info(f"  🎯 Max steps: {config.max_steps}")
        logger.info(f"  📚 Training examples: {len(train_dataset)}")
        logger.info(f"  🔢 Batch size: {config.per_device_train_batch_size}")
        logger.info(f"  📏 LoRA rank: {config.lora_r}")
        logger.info(f"  📐 Sequence length: {config.max_seq_length}")
        
        # Create production trainer
        logger.info("🔧 Creating production LoRA trainer...")
        trainer = ProductionLoRATrainer(config)
        
        # Train with progress feedback
        logger.info("🎯 Starting production training...")
        logger.info("⏳ This will take several minutes with progress feedback...")
        
        trainer.train(train_dataset, eval_dataset)
        
        # Save final model
        final_model_path = f"{output_directory}/final_lora_model"
        trainer.save_model(final_model_path)
        
        logger.info("🎉 Production LoRA training completed successfully!")
        logger.info(f"💾 Final model saved to: {final_model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Production training failed: {str(e)}")
        logger.error(f"🔍 Error details: {traceback.format_exc()}")
        return False

def estimate_training_time(pdf_directory: str, batch_size: int = 2, max_steps: int = 500):
    """Estimate training time based on data size"""
    
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
    
    # Rough estimates based on file size and parameters
    estimated_examples = total_size_mb * 50  # ~50 examples per MB
    estimated_steps = min(estimated_examples // batch_size, max_steps)
    
    # Time estimates (rough)
    gpu_time_per_step = 3  # seconds on average GPU
    cpu_time_per_step = 15  # seconds on CPU
    
    import torch
    if torch.cuda.is_available():
        estimated_time_min = (estimated_steps * gpu_time_per_step) / 60
        device_info = "GPU"
    else:
        estimated_time_min = (estimated_steps * cpu_time_per_step) / 60
        device_info = "CPU"
    
    logger.info(f"Training estimates ({device_info}):")
    logger.info(f"  PDF files: {len(pdf_files)} ({total_size_mb:.1f} MB)")
    logger.info(f"  Estimated examples: {estimated_examples:.0f}")
    logger.info(f"  Training steps: {estimated_steps}")
    logger.info(f"  Estimated time: {estimated_time_min:.1f} minutes")
    
    return estimated_time_min

def validate_paths(pdf_directory: str, output_directory: str):
    """Validate input and output paths"""
    
    pdf_path = Path(pdf_directory)
    if not pdf_path.exists():
        raise ValueError(f"PDF directory does not exist: {pdf_directory}")
    
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_directory}")
    
    logger.info(f"Found {len(pdf_files)} PDF files for training")
    
    # Show file sizes
    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / (1024 * 1024)
        logger.info(f"  {pdf_file.name}: {size_mb:.1f} MB")
    
    # Create output directory
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

def main():
    """Main training script with CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter for IAbel reservoir engineering domain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="/home/lacucaratila/Projetos/IAbel/backend/data/pdfs",
        help="Directory containing PDF files for training"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/outputs",
        help="Output directory for trained models"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Base model name from HuggingFace"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (production default: 3)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size per device (optimized for memory)"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for training (production optimized)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum training steps (production optimized for balance)"
    )
    
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate training time, don't train"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force training even with warnings"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup and validation
        setup_environment()
        validate_paths(args.pdf_dir, args.output_dir)
        
        # Estimate training time
        estimated_time = estimate_training_time(
            args.pdf_dir, 
            args.batch_size, 
            args.max_steps
        )
        
        if args.estimate_only:
            logger.info("Estimation complete. Exiting.")
            return
        
        # Ask for confirmation if training will take long
        if estimated_time > 60 and not args.force:  # More than 1 hour
            response = input(f"\nTraining estimated to take {estimated_time:.1f} minutes. Continue? (y/N): ")
            if response.lower() != 'y':
                logger.info("Training cancelled by user")
                return
        
        # Start training
        logger.info("Starting LoRA training...")
        success = train_lora_model(
            pdf_directory=args.pdf_dir,
            output_directory=args.output_dir,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps
        )
        
        if success:
            logger.info("="*50)
            logger.info("🎉 LoRA training completed successfully!")
            logger.info(f"📁 Model saved to: {args.output_dir}")
            logger.info("🚀 You can now use the hybrid RAG+LoRA system")
            logger.info("="*50)
        else:
            logger.error("❌ Training failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()