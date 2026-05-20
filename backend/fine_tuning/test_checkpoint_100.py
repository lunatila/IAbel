#!/usr/bin/env python3
"""
Teste rápido do checkpoint 100 do treinamento ultra-estável
"""

import os
import sys
import torch
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

def test_checkpoint_100():
    """Test the checkpoint-100 model"""
    
    print("🧪 Testando modelo final ultra-estável...")
    
    checkpoint_path = "./outputs/ultra_stable_lora_extended/final_ultra_stable_model"
    base_model_name = "microsoft/DialoGPT-small"
    
    try:
        # Load tokenizer
        print("1. Carregando tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("   ✅ Tokenizer carregado")
        
        # Load base model
        print("2. Carregando modelo base...")
        from transformers import AutoModelForCausalLM
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,  # Use float32 for stability
            trust_remote_code=True
        )
        print("   ✅ Modelo base carregado")
        
        # Load LoRA adapter
        print("3. Carregando checkpoint LoRA...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()
        
        # Force CPU to avoid CUDA issues during training
        model = model.to("cpu")
        print("   ✅ Checkpoint LoRA carregado (CPU mode)")
        
        # Test questions
        test_questions = [
            "O que é INSIM?",
            "Explique engenharia de reservatórios",
            "Como funciona waterflooding?"
        ]
        
        print("\n🎯 Testando geração de respostas...")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. ❓ {question}")
            
            # Format prompt
            prompt = f"### Pergunta: {question} ### Resposta:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
            inputs = {k: v.cpu() for k, v in inputs.items()}  # Force CPU
            
            # Generate
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if "### Resposta:" in full_response:
                        answer = full_response.split("### Resposta:")[-1].strip()
                    else:
                        answer = full_response[len(prompt):].strip()
                    
                    print(f"   💬 {answer}")
                    
                except Exception as gen_error:
                    print(f"   ❌ Erro na geração: {gen_error}")
        
        print("\n" + "=" * 60)
        print("✅ Teste do modelo final ultra-estável concluído!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    success = test_checkpoint_100()
    exit(0 if success else 1)