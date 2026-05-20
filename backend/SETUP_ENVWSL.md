# 🛠️ Setup Guia: Usando Apenas envWSL

Guia para limpar ambientes duplicados e usar apenas o **envWSL** original.

## 🔍 **Passo 1: Identificar Situação Atual**

```bash
# Verificar se conda está acessível
conda --version

# Se conda não estiver acessível, você precisa configurar PATH
# Localize sua instalação conda:
find /home -name "conda" 2>/dev/null
find /mnt/c -name "conda" 2>/dev/null  # Se conda estiver no Windows

# Adicionar conda ao PATH (se necessário)
echo 'export PATH="/home/username/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 🧹 **Passo 2: Executar Limpeza Automática**

```bash
cd /home/lacucaratila/Projetos/IAbel/backend

# Executar script de limpeza
./cleanup_environments.sh

# Ou manualmente:
bash cleanup_environments.sh
```

## 🔧 **Passo 3: Limpeza Manual (se necessário)**

### **A. Listar Todos os Ambientes**
```bash
conda env list
```

### **B. Identificar envWSL Original**
```bash
# Procurar por envWSL
conda env list | grep envWSL

# Se não encontrar, verificar localizações comuns:
ls -la ~/anaconda3/envs/
ls -la ~/miniconda3/envs/
ls -la /mnt/c/Users/*/Anaconda3/envs/
```

### **C. Remover Ambientes Duplicados**
```bash
# Exemplos de ambientes a remover:
conda env remove -n iabel
conda env remove -n lora_env
conda env remove -n backend
conda env remove -n fine_tuning

# Remover virtual environments locais
rm -rf ./venv ./lora_env ./iabel_env
```

### **D. Limpar Caches**
```bash
conda clean --all -y
pip cache purge
```

## ✅ **Passo 4: Verificar envWSL**

```bash
# Ativar envWSL
conda activate envWSL

# Verificar se está correto
conda info --envs
echo "Ambiente ativo: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Listar packages atuais
pip list
```

## 📦 **Passo 5: Instalar Dependências no envWSL**

### **A. Dependências Básicas**
```bash
# Certificar que está no envWSL
conda activate envWSL

# PDF Processing
pip install PyMuPDF==1.23.26
pip install pdfplumber==0.10.3

# Machine Learning Core
pip install transformers==4.44.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# LoRA & Fine-tuning
pip install peft==0.7.1
pip install accelerate==0.25.0
pip install datasets==2.16.1
pip install evaluate==0.4.1

# Data Processing
pip install pandas==2.2.0
pip install numpy==1.26.4
```

### **B. Verificar Instalação**
```bash
# Testar imports críticos
python -c "
import PyMuPDF; print('✅ PyMuPDF OK')
import pdfplumber; print('✅ pdfplumber OK') 
import transformers; print('✅ transformers OK')
import torch; print('✅ torch OK')
import peft; print('✅ peft OK')
print('🎉 Todas dependências OK!')
"

# Testar GPU (se disponível)
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ GPU não detectada - usando CPU')
"
```

## 🧪 **Passo 6: Testar Setup**

```bash
cd /home/lacucaratila/Projetos/IAbel/backend/fine_tuning

# Testar setup completo
python test_setup.py

# Se tudo estiver OK, testar treinamento
python train_lora.py --estimate-only
```

## 🚨 **Resolução de Problemas Comuns**

### **Problema: conda command not found**
```bash
# Localizar conda
find /home -name conda 2>/dev/null
find /mnt/c -name conda 2>/dev/null

# Adicionar ao PATH
export PATH="/caminho/para/conda/bin:$PATH"
echo 'export PATH="/caminho/para/conda/bin:$PATH"' >> ~/.bashrc
```

### **Problema: envWSL não existe**
```bash
# Criar envWSL
conda create -n envWSL python=3.11 -y
conda activate envWSL
```

### **Problema: Conflitos de packages**
```bash
# Reset completo do envWSL
conda env remove -n envWSL
conda create -n envWSL python=3.11 -y
conda activate envWSL
# Reinstalar dependências
```

### **Problema: CUDA não funciona**
```bash
# Verificar driver NVIDIA
nvidia-smi

# Reinstalar PyTorch com CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ✨ **Resultado Final**

Após seguir este guia, você terá:

✅ **Apenas um ambiente**: envWSL  
✅ **Limpo e organizado**: Sem duplicatas  
✅ **Todas dependências**: Instaladas corretamente  
✅ **GPU funcionando**: Se disponível  
✅ **Pronto para LoRA**: Treinamento habilitado  

## 📝 **Comandos de Uso Diário**

```bash
# Sempre usar estes comandos:
conda activate envWSL
cd /home/lacucaratila/Projetos/IAbel/backend

# Para LoRA training:
python fine_tuning/train_lora.py

# Para testar:
python fine_tuning/test_setup.py

# Para desativar:
conda deactivate
```

---

**💡 Dica**: Sempre execute `conda activate envWSL` antes de trabalhar no projeto IAbel!