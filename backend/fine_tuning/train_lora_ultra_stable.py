#!/usr/bin/env python3
"""
Ultra-Stable LoRA Training for IAbel
Advanced training with NaN/Inf detection, automatic recovery, and robust checkpointing
"""

import os
import sys
import json
import torch
import logging
import warnings
import numpy as np
import shutil
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

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
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultra_stable_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class UltraStableConfig:
    """Ultra-conservative configuration for maximum stability"""
    
    # Model settings
    base_model: str = "microsoft/DialoGPT-small"
    model_max_length: int = 512  # Shorter for stability
    
    # Ultra-conservative LoRA parameters
    lora_r: int = 16  # Lower rank for stability
    lora_alpha: int = 32  # 2x rank
    lora_dropout: float = 0.1  # Higher dropout for regularization
    target_modules: List[str] = None
    
    # Extended training hyperparameters for ~6 hours
    num_train_epochs: int = 100  # Much more epochs for thorough training
    per_device_train_batch_size: int = 2  # Very small batch
    gradient_accumulation_steps: int = 16  # Effective batch = 32
    
    # Optimized learning rate for extended training
    learning_rate: float = 8e-6  # Even lower LR for longer training
    min_learning_rate: float = 5e-7  # Minimum LR
    warmup_ratio: float = 0.15  # Moderate warmup for longer training
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.05  # Strong regularization
    
    # Stability settings
    max_grad_norm: float = 0.5  # Strong gradient clipping
    stability_check_steps: int = 10  # Check stability every N steps
    
    # Checkpoint settings for extended training
    save_steps: int = 100  # Save less frequently for longer training
    eval_steps: int = 100
    save_total_limit: int = 10  # Keep more checkpoints
    
    # Output
    output_dir: str = "./outputs/ultra_stable_lora_extended"
    checkpoint_dir: str = "./outputs/ultra_stable_lora_extended/safe_checkpoints"
    
    # Extended training settings
    early_stopping_patience: int = 15  # More patience for longer training
    min_improvement_threshold: float = 0.001  # Minimum improvement to continue

class StabilityMonitor(TrainerCallback):
    """Advanced callback to monitor training stability and recover from instability"""
    
    def __init__(self, config: UltraStableConfig, tokenizer, model):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.last_stable_checkpoint = None
        self.instability_count = 0
        self.current_lr = config.learning_rate
        self.best_eval_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_start_time = None
        
        # Create safe checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info("🔍 Stability Monitor initialized for extended training")
    
    def check_tensor_health(self, tensor: torch.Tensor, name: str) -> bool:
        """Check if tensor contains NaN or Inf values"""
        if tensor is None:
            return True
        
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan:
            logger.error(f"❌ NaN detected in {name}")
            return False
        if has_inf:
            logger.error(f"❌ Inf detected in {name}")
            return False
        
        return True
    
    def check_model_health(self) -> bool:
        """Comprehensive model health check"""
        try:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if not self.check_tensor_health(param.grad, f"gradient_{name}"):
                        return False
                if not self.check_tensor_health(param.data, f"param_{name}"):
                    return False
            return True
        except Exception as e:
            logger.error(f"❌ Model health check failed: {e}")
            return False
    
    def test_generation_health(self) -> bool:
        """Test if model can generate without errors"""
        try:
            test_prompt = "### Pergunta: Teste ### Resposta:"
            inputs = self.tokenizer(test_prompt, return_tensors="pt", max_length=100, truncation=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,  # Deterministic for testing
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Check if output contains valid tokens
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check for problematic patterns
            if len(output_text.strip()) == 0:
                logger.warning("⚠️ Model generating empty text")
                return False
            
            # Check for excessive repetition of same character
            if any(char * 10 in output_text for char in "!@#$%^&*()"):
                logger.warning("⚠️ Model generating repetitive patterns")
                return False
            
            logger.debug(f"✅ Generation test passed: {output_text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Generation test failed: {e}")
            return False
    
    def save_safe_checkpoint(self, trainer, step: int):
        """Save a validated checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"safe_checkpoint_{step}"
        
        try:
            # Save model state
            trainer.save_model(str(checkpoint_path))
            
            # Save additional info
            checkpoint_info = {
                "step": step,
                "learning_rate": self.current_lr,
                "instability_count": self.instability_count,
                "timestamp": datetime.now().isoformat(),
                "health_status": "stable"
            }
            
            with open(checkpoint_path / "checkpoint_info.json", 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            self.last_stable_checkpoint = str(checkpoint_path)
            logger.info(f"💾 Safe checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save safe checkpoint: {e}")
    
    def load_last_stable_checkpoint(self, trainer):
        """Load the last stable checkpoint"""
        if not self.last_stable_checkpoint:
            logger.error("❌ No stable checkpoint available for recovery")
            return False
        
        try:
            logger.info(f"🔄 Loading stable checkpoint: {self.last_stable_checkpoint}")
            
            # Load model state
            trainer.model.load_adapter(self.last_stable_checkpoint, adapter_name="default", is_trainable=True)
            
            # Reset optimizer state
            trainer.optimizer.zero_grad()
            
            logger.info("✅ Successfully recovered from stable checkpoint")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load stable checkpoint: {e}")
            return False
    
    def handle_instability(self, trainer, logs: Dict[str, float]):
        """Handle detected instability"""
        self.instability_count += 1
        logger.warning(f"⚠️ Instability detected (count: {self.instability_count})")
        
        if self.instability_count <= 3:
            # Try reducing learning rate
            new_lr = self.current_lr * 0.5
            logger.info(f"📉 Reducing learning rate: {self.current_lr:.2e} -> {new_lr:.2e}")
            
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            
            # Load last stable checkpoint if available
            if self.last_stable_checkpoint:
                self.load_last_stable_checkpoint(trainer)
        
        elif self.instability_count <= 5:
            # More aggressive recovery
            logger.warning("🚨 Persistent instability - aggressive recovery")
            
            # Further reduce LR
            new_lr = max(self.config.min_learning_rate, self.current_lr * 0.1)
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.current_lr = new_lr
            
            # Force checkpoint recovery
            if self.last_stable_checkpoint:
                self.load_last_stable_checkpoint(trainer)
        
        else:
            # Too many instabilities - stop training
            logger.error("🛑 Too many instabilities detected - stopping training")
            trainer.control.should_training_stop = True
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check stability at each step"""
        
        # Get trainer safely
        trainer = kwargs.get('trainer')
        if not trainer:
            return control
            
        # Check every N steps
        if state.global_step % self.config.stability_check_steps == 0:
            
            # Check loss for NaN/Inf
            if state.log_history:
                latest_log = state.log_history[-1]
                if 'train_loss' in latest_log:
                    loss = latest_log['train_loss']
                    if np.isnan(loss) or np.isinf(loss):
                        logger.error(f"❌ Invalid loss detected: {loss}")
                        self.handle_instability(trainer, latest_log)
                        return control
            
            # Check model health
            if not self.check_model_health():
                logger.error("❌ Model health check failed")
                self.handle_instability(trainer, {})
                return control
            
            # Periodic generation test
            if state.global_step % (self.config.stability_check_steps * 5) == 0:
                if not self.test_generation_health():
                    logger.error("❌ Generation health check failed")
                    self.handle_instability(trainer, {})
                    return control
        
        return control
    
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize training start time"""
        self.training_start_time = datetime.now()
        logger.info(f"🚀 Extended training started at {self.training_start_time.strftime('%H:%M:%S')}")
        logger.info(f"📊 Target: {args.num_train_epochs} epochs (~6 hours)")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Save safe checkpoint and monitor progress for extended training"""
        
        # Get trainer safely
        trainer = kwargs.get('trainer')
        if not trainer:
            return control
            
        # Calculate progress
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            progress_pct = (state.epoch / args.num_train_epochs) * 100
            estimated_total = elapsed / (state.epoch / args.num_train_epochs) if state.epoch > 0 else elapsed
            remaining = estimated_total - elapsed
            
            logger.info(f"📈 Progress: {progress_pct:.1f}% | Elapsed: {elapsed} | ETA: {remaining}")
        
        # Check eval loss improvement
        if state.log_history:
            eval_logs = [log for log in state.log_history if 'eval_loss' in log]
            if eval_logs:
                current_eval_loss = eval_logs[-1]['eval_loss']
                
                if current_eval_loss < self.best_eval_loss - self.config.min_improvement_threshold:
                    self.best_eval_loss = current_eval_loss
                    self.epochs_without_improvement = 0
                    logger.info(f"📉 New best eval loss: {current_eval_loss:.6f}")
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"📊 No improvement for {self.epochs_without_improvement} epochs (best: {self.best_eval_loss:.6f})")
                
                # Early stopping for extended training
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    logger.info(f"🛑 Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                    control.should_training_stop = True
                    return control
        
        # Health checks and checkpoint saving
        if self.check_model_health() and self.test_generation_health():
            self.save_safe_checkpoint(trainer, state.global_step)
            self.instability_count = 0  # Reset instability count on successful epoch
            logger.info(f"✅ Epoch {state.epoch:.0f} completed successfully")
        else:
            logger.warning(f"⚠️ Epoch {state.epoch:.0f} completed with instability")
        
        return control

class UltraStableTrainer:
    """Ultra-stable LoRA trainer with advanced monitoring"""
    
    def __init__(self, config: UltraStableConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        # Enhanced training data
        self.training_data = self._create_enhanced_training_data()
        
        # Set target modules
        config.target_modules = ["c_attn", "c_proj", "c_fc"]
    
    def _create_enhanced_training_data(self) -> List[Dict[str, str]]:
        """Create high-quality, diverse training data"""
        return [
            {
                "instruction": "O que é engenharia de reservatórios?",
                "output": "A engenharia de reservatórios é uma disciplina da engenharia de petróleo que se dedica ao estudo, caracterização e desenvolvimento de técnicas para maximizar a extração de hidrocarbonetos dos reservatórios subterrâneos. Esta área envolve a análise de propriedades da rocha e dos fluidos, modelagem de fluxo, simulação numérica e implementação de métodos de recuperação primária, secundária e terciária."
            },
            {
                "instruction": "Explique o método INSIM",
                "output": "INSIM (Interwell Numerical Simulation Model) é um método de simulação numérica interpoços que resolve equações de escoamento bifásico em uma malha construída a partir da trajetória dos poços. O método utiliza modelos capacitivos-resistivos para representar o comportamento do reservatório, oferecendo eficiência computacional superior aos simuladores convencionais para ajuste de histórico e predição de desempenho."
            },
            {
                "instruction": "Como funciona waterflooding?",
                "output": "Waterflooding é um método de recuperação secundária onde água é injetada em poços específicos para deslocar óleo em direção aos poços produtores. O processo baseia-se na criação de um diferencial de pressão que força o movimento dos fluidos através do meio poroso. A eficiência depende da permeabilidade relativa, molhabilidade da rocha, razão de mobilidade entre água e óleo, e padrão de injeção utilizado."
            },
            {
                "instruction": "O que é simulação de reservatórios?",
                "output": "A simulação de reservatórios é uma técnica computacional que utiliza modelos matemáticos para representar o comportamento de fluidos em meios porosos subterrâneos. Os simuladores resolvem equações de fluxo multifásico considerando as propriedades da rocha, dos fluidos e as condições de contorno para prever a produção de hidrocarbonetos ao longo do tempo e avaliar diferentes estratégias de desenvolvimento."
            },
            {
                "instruction": "Defina permeabilidade relativa",
                "output": "Permeabilidade relativa é uma propriedade adimensional que quantifica a capacidade de um fluido específico escoar através de um meio poroso na presença de outros fluidos. É expressa como uma fração da permeabilidade absoluta e varia com a saturação dos fluidos presentes. As curvas de permeabilidade relativa são fundamentais para a modelagem de fluxo multifásico em reservatórios de petróleo."
            },
            {
                "instruction": "O que são métodos de recuperação avançada?",
                "output": "Métodos de recuperação avançada (EOR - Enhanced Oil Recovery) são técnicas utilizadas para extrair petróleo adicional após as fases de recuperação primária e secundária. Incluem métodos térmicos como injeção de vapor e combustão in-situ, métodos químicos como injeção de polímeros e surfactantes, e métodos miscíveis como injeção de CO2 e nitrogênio. Estes métodos visam alterar as propriedades dos fluidos ou da rocha para mobilizar óleo residual."
            },
            {
                "instruction": "Como funciona o INSIM-FT?",
                "output": "INSIM-FT é uma versão aprimorada do INSIM que utiliza a técnica de front-tracking para calcular saturações com maior precisão. O método combina a eficiência computacional do INSIM tradicional com maior exatidão no cálculo de saturação através do rastreamento explícito de frentes de saturação. Isso permite modelar com maior fidelidade fenômenos como breakthrough de água e evolução de contatos fluido-fluido em reservatórios heterogêneos."
            },
            {
                "instruction": "O que é ajuste de histórico?",
                "output": "Ajuste de histórico é o processo de calibração de modelos de reservatório para reproduzir dados observados de produção, pressão e outros parâmetros dinâmicos. Este processo envolve a modificação sistemática de propriedades do modelo como permeabilidade, porosidade e transmissibilidades até que as respostas simuladas coincidam adequadamente com os dados históricos, aumentando assim a confiabilidade das predições futuras."
            },
            {
                "instruction": "Defina fator de recuperação",
                "output": "Fator de recuperação é a fração do óleo original in-place que pode ser extraída economicamente do reservatório. Geralmente expresso em porcentagem, varia tipicamente entre 20% e 60% dependendo das características do reservatório, propriedades dos fluidos e métodos de recuperação empregados. É influenciado por fatores como heterogeneidade da rocha, viscosidade do óleo, pressão do reservatório e técnicas de produção utilizadas."
            },
            {
                "instruction": "O que é análise nodal?",
                "output": "Análise nodal é uma metodologia para otimização de sistemas de produção de petróleo que divide o sistema produtivo em componentes e analisa as perdas de pressão em cada segmento. A técnica permite identificar gargalos no sistema e otimizar vazões de produção através do balanceamento entre as curvas de inflow do reservatório e outflow do sistema de produção, incluindo completação, coluna de produção e equipamentos de superfície."
            }
        ]
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with ultra-stable configuration"""
        logger.info(f"🚀 Setting up ultra-stable model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            model_max_length=self.config.model_max_length,
            padding_side="right",
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with float32 for maximum stability
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.float32,  # Always use float32 for stability
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Move to GPU carefully
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model moved to GPU")
        
        # Configure LoRA with conservative settings
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            fan_in_fan_out=True  # For DialoGPT
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("✅ Ultra-stable model setup complete")
    
    def prepare_dataset(self):
        """Prepare training dataset with quality validation"""
        logger.info("📊 Preparing ultra-stable dataset...")
        
        # Format examples with data augmentation
        formatted_examples = []
        for example in self.training_data:
            # Primary format
            text = f"### Pergunta: {example['instruction']} ### Resposta: {example['output']}<|endoftext|>"
            formatted_examples.append({"text": text})
            
            # Add slight variations for robustness
            variations = [
                f"### Pergunta: {example['instruction']}? ### Resposta: {example['output']}<|endoftext|>",
                f"### Pergunta: Explique {example['instruction'].lower()} ### Resposta: {example['output']}<|endoftext|>"
            ]
            
            for var in variations:
                if len(var) <= self.config.model_max_length * 4:  # Reasonable length
                    formatted_examples.append({"text": var})
        
        # Multiply dataset for extended training (more data for longer training)
        formatted_examples = formatted_examples * 8
        
        logger.info(f"Created {len(formatted_examples)} training examples")
        
        # Create and tokenize dataset
        dataset = Dataset.from_list(formatted_examples)
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,  # Explicitly enable truncation
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
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        logger.info(f"Training samples: {len(split_dataset['train'])}")
        logger.info(f"Evaluation samples: {len(split_dataset['test'])}")
        
        return split_dataset["train"], split_dataset["test"]
    
    def train(self):
        """Execute ultra-stable training"""
        logger.info("🎯 Starting ultra-stable LoRA training...")
        
        # Setup
        self.setup_model_and_tokenizer()
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Training arguments with ultra-stable settings
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Ultra-conservative optimization
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Stability settings
            fp16=False,  # Use FP32 for maximum stability
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=[],
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        # Create stability monitor
        stability_monitor = StabilityMonitor(self.config, self.tokenizer, self.peft_model)
        
        # Create trainer with stability monitoring
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[stability_monitor]
        )
        
        # Start training
        logger.info("🔥 Beginning ultra-stable training with monitoring...")
        
        try:
            trainer.train()
            
            # Save final model if stable
            if stability_monitor.check_model_health() and stability_monitor.test_generation_health():
                final_path = Path(self.config.output_dir) / "final_ultra_stable_model"
                trainer.save_model(str(final_path))
                self.tokenizer.save_pretrained(str(final_path))
                
                logger.info(f"✅ Ultra-stable training completed! Final model: {final_path}")
                
                # Test the final model
                self.test_final_model(str(final_path))
                return True
            else:
                logger.error("❌ Training completed but final model is unstable")
                return False
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False
    
    def test_final_model(self, model_path: str):
        """Test the final trained model"""
        logger.info("🧪 Testing final ultra-stable model...")
        
        try:
            from peft import PeftModel
            
            # Load for testing
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float32
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model.eval()
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            test_questions = [
                "O que é engenharia de reservatórios?",
                "Explique o método INSIM",
                "Como funciona waterflooding?"
            ]
            
            for question in test_questions:
                prompt = f"### Pergunta: {question} ### Resposta:"
                
                inputs = tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        repetition_penalty=1.1,
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
            logger.error(f"Error testing final model: {e}")

def main():
    """Main training function"""
    logger.info("🚀 Ultra-Stable LoRA Training for IAbel")
    logger.info("Features: NaN/Inf detection, automatic recovery, robust checkpointing")
    
    # Create configuration
    config = UltraStableConfig()
    
    # Log configuration
    logger.info("📋 Training Configuration:")
    logger.info(f"  - Base Model: {config.base_model}")
    logger.info(f"  - LoRA Rank: {config.lora_r}, Alpha: {config.lora_alpha}")
    logger.info(f"  - Learning Rate: {config.learning_rate}")
    logger.info(f"  - Epochs: {config.num_train_epochs}")
    logger.info(f"  - Gradient Clipping: {config.max_grad_norm}")
    logger.info(f"  - Stability Checks: Every {config.stability_check_steps} steps")
    
    # Create trainer
    trainer = UltraStableTrainer(config)
    
    # Execute training
    try:
        success = trainer.train()
        if success:
            logger.info("🎉 Ultra-stable LoRA training completed successfully!")
        else:
            logger.error("❌ Ultra-stable training failed")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()