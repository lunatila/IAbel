#!/usr/bin/env python3
"""
Simple Robust LoRA Training Script
Direct execution without complex launcher
"""

import os
import sys
import logging
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Execute robust LoRA training"""
    
    logger.info("🚀 Starting Robust LoRA Training for IAbel")
    
    try:
        # Import training modules
        from train_lora_portuguese import run_portuguese_training, PORTUGUESE_CONFIGS
        
        logger.info("📋 Available Models:")
        for name, config in PORTUGUESE_CONFIGS.items():
            logger.info(f"  - {name}: {config['description']}")
        
        # Use Portuguese GPT-2 as default (best for Portuguese)
        model_name = "gpt2_portuguese"
        logger.info(f"🎯 Using model: {model_name}")
        logger.info(f"📄 Base model: {PORTUGUESE_CONFIGS[model_name]['base_model']}")
        
        # Execute training
        logger.info("🔥 Starting training with enhanced configuration:")
        logger.info("  - Epochs: 20")
        logger.info("  - Learning Rate: 3e-5") 
        logger.info("  - LoRA Rank: 24")
        logger.info("  - Batch Size: 6 (effective: 36 with gradient accumulation)")
        logger.info("  - Warmup Ratio: 15%")
        logger.info("  - Scheduler: Cosine with restarts")
        
        success = run_portuguese_training(model_name)
        
        if success:
            logger.info("✅ Robust LoRA training completed successfully!")
            logger.info("📁 Model saved to: ./outputs/portuguese_lora_gpt2_portuguese/")
            return True
        else:
            logger.error("❌ Training failed")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please install required packages: pip install transformers peft datasets torch")
        return False
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)