#!/usr/bin/env python3
"""
Start Robust LoRA Training - Quick launcher script
Executes optimized training with monitoring and progress tracking
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robust_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check training environment and dependencies"""
    logger.info("🔍 Checking training environment...")
    
    checks = {
        "CUDA Available": torch.cuda.is_available() if 'torch' in sys.modules else False,
        "GPU Memory": None,
        "Required Packages": True,
        "PDF Directory": Path("../data/pdfs").exists(),
        "Output Directory": True
    }
    
    try:
        import torch
        checks["CUDA Available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            checks["GPU Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        # Check required packages
        required_packages = ['transformers', 'peft', 'datasets', 'torch']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                checks["Required Packages"] = False
                logger.error(f"Missing package: {package}")
        
    except Exception as e:
        logger.error(f"Environment check error: {e}")
    
    # Create output directory
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Print environment status
    logger.info("📋 Environment Status:")
    for check, status in checks.items():
        status_emoji = "✅" if status else "❌"
        logger.info(f"  {status_emoji} {check}: {status}")
    
    return all([v for v in checks.values() if isinstance(v, bool)])

def create_training_config():
    """Create and save optimized training configuration"""
    
    config = {
        "training_type": "robust_lora",
        "model_configs": {
            "option_1": {
                "name": "Portuguese GPT-2",
                "base_model": "pierreguillou/gpt2-small-portuguese",
                "description": "Best for Portuguese text generation",
                "recommended": True
            },
            "option_2": {
                "name": "DialoGPT Small", 
                "base_model": "microsoft/DialoGPT-small",
                "description": "Lighter model, potentially better training",
                "recommended": True
            },
            "option_3": {
                "name": "Original GPT-2",
                "base_model": "gpt2",
                "description": "Reliable baseline for fine-tuning",
                "recommended": False
            }
        },
        "hyperparameters": {
            "epochs": 20,
            "learning_rate": "3e-5",
            "batch_size": 6,
            "gradient_accumulation": 6,
            "lora_r": 24,
            "lora_alpha": 48,
            "warmup_ratio": 0.15
        },
        "expected_duration": "2-4 hours",
        "hardware_requirements": "GPU with 8GB+ VRAM recommended"
    }
    
    # Save config
    config_file = Path("robust_training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 Training configuration saved to: {config_file}")
    return config

def run_training_command(model_choice: str = "gpt2_portuguese"):
    """Execute the training command"""
    
    logger.info(f"🚀 Starting robust LoRA training with model: {model_choice}")
    logger.info("⏰ Estimated training time: 2-4 hours")
    
    # Prepare command
    if model_choice in ["gpt2_portuguese", "dialogpt_small", "gpt2_multilingual"]:
        script = "train_lora_portuguese.py"
        cmd = [sys.executable, script, "--model", model_choice]
    else:
        script = "train_lora_robust.py"
        cmd = [sys.executable, script]
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
    env["CUDA_LAUNCH_BLOCKING"] = "0"  # Allow async CUDA operations
    
    logger.info(f"💻 Executing: {' '.join(cmd)}")
    
    try:
        # Start training process
        start_time = time.time()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        # Monitor training progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Log training output
                logger.info(f"TRAINING: {output.strip()}")
        
        # Get return code
        return_code = process.poll()
        
        # Calculate duration
        duration = time.time() - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        
        if return_code == 0:
            logger.info(f"✅ Training completed successfully in {hours}h {minutes}m!")
            return True
        else:
            logger.error(f"❌ Training failed with return code: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"Training execution error: {e}")
        return False

def main():
    """Main launcher function"""
    
    print("🎯 IAbel Robust LoRA Training Launcher")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please install missing dependencies.")
        return
    
    # Create config
    config = create_training_config()
    
    # Show model options
    print("\n📋 Available Models:")
    for key, model_config in config["model_configs"].items():
        recommended = "⭐ RECOMMENDED" if model_config["recommended"] else ""
        print(f"  {key}: {model_config['name']} - {model_config['description']} {recommended}")
    
    # Get user choice
    print("\n" + "=" * 50)
    model_choice = input("Choose model (1/2/3) or enter 'auto' for best option: ").strip()
    
    # Map choice to model name
    choice_map = {
        "1": "gpt2_portuguese",
        "2": "dialogpt_small", 
        "3": "gpt2_multilingual",
        "auto": "gpt2_portuguese",
        "": "gpt2_portuguese"  # Default
    }
    
    selected_model = choice_map.get(model_choice, "gpt2_portuguese")
    
    print(f"\n🎯 Selected: {selected_model}")
    print(f"⏰ Expected duration: {config['expected_duration']}")
    print(f"💾 Hardware needs: {config['hardware_requirements']}")
    
    # Confirm start
    confirm = input("\nStart training? [Y/n]: ").strip().lower()
    if confirm not in ['', 'y', 'yes']:
        print("Training cancelled.")
        return
    
    # Run training
    print("\n🚀 Starting robust LoRA training...")
    print("📝 Training logs will be saved to: robust_training.log")
    print("🔄 This may take several hours. Please be patient...")
    
    success = run_training_command(selected_model)
    
    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Check the outputs/ directory for the trained model")
        print("🧪 The model will be automatically tested after training")
    else:
        print("\n❌ Training failed. Check robust_training.log for details")

if __name__ == "__main__":
    # Add current directory to path
    sys.path.append(str(Path(__file__).parent))
    
    try:
        import torch
        main()
    except ImportError:
        print("❌ PyTorch not found. Please install requirements:")
        print("pip install torch transformers peft datasets")
        sys.exit(1)