#!/usr/bin/env python3
"""
LoRA Model Investigation Script
Analyze repetition issues and test different generation parameters
"""

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add paths
sys.path.append(str(Path(__file__).parent))

def test_lora_generation():
    """Test LoRA model with different parameters to diagnose repetition"""
    
    print("🔍 Investigando modelo LoRA...")
    
    # Paths
    lora_path = "/home/lacucaratila/Projetos/IAbel/backend/fine_tuning/outputs/final_lora_model"
    base_model_name = "microsoft/DialoGPT-medium"
    
    try:
        # Load tokenizer
        print("📝 Carregando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(lora_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        print("🤖 Carregando modelo base...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load LoRA adapter
        print("🎯 Carregando adaptador LoRA...")
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        
        # Test prompts
        test_prompts = [
            "### Instrução: O que é engenharia de reservatórios?\n### Resposta:",
            "### Instrução: Defina simulação de reservatórios\n### Resposta:",
            "### Instrução: Explique waterflooding\n### Resposta:"
        ]
        
        # Test different generation parameters
        param_sets = [
            {
                "name": "Current Settings",
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "do_sample": True
            },
            {
                "name": "Conservative",
                "max_new_tokens": 100,
                "temperature": 0.3,
                "top_p": 0.7,
                "repetition_penalty": 1.3,
                "do_sample": True
            },
            {
                "name": "High Penalty",
                "max_new_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.8,
                "repetition_penalty": 1.5,
                "do_sample": True
            },
            {
                "name": "Deterministic",
                "max_new_tokens": 100,
                "temperature": 0.1,
                "top_p": 0.95,
                "repetition_penalty": 1.2,
                "do_sample": False
            }
        ]
        
        for prompt in test_prompts:
            print(f"\n{'='*60}")
            print(f"🔥 Testando: {prompt.split(':')[1].split('?')[0].strip()}?")
            print('='*60)
            
            for params in param_sets:
                print(f"\n--- {params['name']} ---")
                
                # Tokenize
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        **{k: v for k, v in params.items() if k != "name"}
                    )
                
                # Decode
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract generated part
                if "### Resposta:" in full_response:
                    response = full_response.split("### Resposta:")[-1].strip()
                else:
                    response = full_response[len(prompt):].strip()
                
                print(f"Resposta ({len(response)} chars): {response[:200]}...")
                
                # Analyze repetition
                words = response.split()
                if len(words) > 5:
                    # Check for repeated words
                    word_counts = {}
                    for word in words[:20]:  # First 20 words
                        word_counts[word] = word_counts.get(word, 0) + 1
                    
                    repeated = [w for w, c in word_counts.items() if c > 2]
                    if repeated:
                        print(f"⚠️ Palavras repetidas: {repeated}")
                    else:
                        print("✅ Sem repetições detectadas")
        
        print(f"\n{'='*60}")
        print("📊 ANÁLISE DO MODELO:")
        print(f"- Base Model: {base_model_name}")
        print(f"- LoRA r: 16, alpha: 32, dropout: 0.1")
        print(f"- Target modules: c_proj, c_attn, c_fc")
        print(f"- Tokenizer max_length: 1024")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lora_generation()