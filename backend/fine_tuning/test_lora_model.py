#!/usr/bin/env python3
"""
Teste do Modelo LoRA Treinado
Script para validar o modelo fine-tuned e comparar com modelo base
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_lora_model(model_path: str):
    """Load LoRA fine-tuned model"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    logger.info("🔧 Loading LoRA model...")
    
    # Load base model and tokenizer
    base_model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("🎮 Model moved to GPU")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("✅ LoRA model loaded successfully!")
    return model, tokenizer

def load_base_model():
    """Load base model for comparison"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    logger.info("🔧 Loading base model for comparison...")
    
    base_model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("✅ Base model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 100):
    """Generate response from model"""
    
    # Format prompt
    formatted_prompt = f"### Pergunta: {prompt} ### Resposta:"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    if "### Resposta:" in response:
        response = response.split("### Resposta:")[-1].strip()
    
    return response

def test_questions():
    """Test questions about reservoir engineering"""
    return [
        "O que é INSIM?",
        "Como funciona o método INSIM?",
        "O que é engenharia de reservatórios?", 
        "Explique sobre modelos capacitivos-resistivos",
    ]

def compare_models():
    """Compare LoRA model vs base model"""
    
    logger.info("🚀 Starting LoRA Model Testing...")
    
    # Paths
    lora_model_path = "/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/outputs/final_lora_model"
    
    if not os.path.exists(lora_model_path):
        logger.error(f"❌ LoRA model not found at: {lora_model_path}")
        return
    
    # Load models
    logger.info("📥 Loading models...")
    lora_model, lora_tokenizer = load_lora_model(lora_model_path)
    base_model, base_tokenizer = load_base_model()
    
    # Test questions
    questions = test_questions()
    
    print("\n" + "="*80)
    print("🧪 COMPARAÇÃO: MODELO BASE vs MODELO LORA FINE-TUNED")
    print("="*80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 PERGUNTA {i}: {question}")
        print("-" * 60)
        
        # Base model response
        print("🤖 MODELO BASE:")
        try:
            base_response = generate_response(base_model, base_tokenizer, question)
            print(f"   {base_response}")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        
        print()
        
        # LoRA model response  
        print("🎯 MODELO LORA (FINE-TUNED):")
        try:
            lora_response = generate_response(lora_model, lora_tokenizer, question)
            print(f"   {lora_response}")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        
        print("-" * 60)
    
    print("\n" + "="*80)
    print("✅ Teste concluído!")
    print("💡 Analise as diferenças para validar o fine-tuning")
    print("="*80)

def quick_test():
    """Quick test of LoRA model only"""
    
    logger.info("⚡ Quick LoRA Test...")
    
    lora_model_path = "/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/outputs/final_lora_model"
    
    if not os.path.exists(lora_model_path):
        logger.error(f"❌ LoRA model not found at: {lora_model_path}")
        return
    
    # Load LoRA model
    model, tokenizer = load_lora_model(lora_model_path)
    
    # Quick test
    test_question = "O que é INSIM?"
    print(f"\n🧪 TESTE RÁPIDO")
    print(f"📝 Pergunta: {test_question}")
    print("-" * 40)
    
    response = generate_response(model, tokenizer, test_question)
    print(f"🎯 Resposta LoRA: {response}")
    print("-" * 40)
    print("✅ Teste rápido concluído!")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LoRA fine-tuned model")
    parser.add_argument("--quick", action="store_true", help="Quick test only")
    parser.add_argument("--compare", action="store_true", help="Compare base vs LoRA")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.compare:
        compare_models()
    else:
        # Default: quick test
        quick_test()
        print("\n💡 Use --compare para comparação completa")

if __name__ == "__main__":
    main()