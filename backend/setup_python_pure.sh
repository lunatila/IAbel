#!/bin/bash
# Setup Script for IAbel using Pure Python (no conda)
# Creates virtual environment and installs all dependencies

echo "🐍 IAbel Pure Python Setup"
echo "=========================="

PROJECT_DIR="/home/lacucaratila/Projetos/IAbel/backend"
VENV_NAME="envWSL"
VENV_PATH="$PROJECT_DIR/$VENV_NAME"

# Function to check Python version
check_python() {
    echo "🔍 Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 not found!"
        echo "Install with: sudo apt update && sudo apt install python3 python3-pip python3-venv"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✅ Python found: $PYTHON_VERSION"
    
    # Check if version is 3.8+
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        echo "✅ Python version is compatible (3.8+)"
    else
        echo "❌ Python 3.8+ required for LoRA training"
        exit 1
    fi
}

# Function to cleanup old environments
cleanup_old_envs() {
    echo ""
    echo "🧹 Cleaning up old environments..."
    
    # Remove any conda environments if they exist
    if command -v conda &> /dev/null; then
        echo "🗑️ Removing conda environments..."
        conda env remove -n iabel -y 2>/dev/null || true
        conda env remove -n lora_env -y 2>/dev/null || true
        conda env remove -n backend -y 2>/dev/null || true
    fi
    
    # Remove old virtual environments
    OLD_VENVS=("./venv" "./lora_env" "./iabel_env" "./env" "./.venv")
    for venv in "${OLD_VENVS[@]}"; do
        if [ -d "$venv" ]; then
            echo "🗑️ Removing old venv: $venv"
            rm -rf "$venv"
        fi
    done
    
    echo "✅ Cleanup completed"
}

# Function to create virtual environment
create_venv() {
    echo ""
    echo "📦 Creating virtual environment..."
    
    cd "$PROJECT_DIR"
    
    # Remove existing envWSL if it exists
    if [ -d "$VENV_PATH" ]; then
        echo "🗑️ Removing existing envWSL..."
        rm -rf "$VENV_PATH"
    fi
    
    # Create new virtual environment
    echo "🆕 Creating new envWSL..."
    python3 -m venv "$VENV_NAME"
    
    if [ $? -eq 0 ]; then
        echo "✅ Virtual environment created: $VENV_PATH"
    else
        echo "❌ Failed to create virtual environment"
        echo "Try: sudo apt install python3-venv"
        exit 1
    fi
}

# Function to activate and test environment
activate_venv() {
    echo ""
    echo "🔌 Activating virtual environment..."
    
    source "$VENV_PATH/bin/activate"
    
    if [ "$VIRTUAL_ENV" = "$VENV_PATH" ]; then
        echo "✅ Environment activated successfully"
        echo "📍 Virtual env: $VIRTUAL_ENV"
        echo "🐍 Python: $(which python)"
        echo "🐍 Version: $(python --version)"
    else
        echo "❌ Failed to activate environment"
        exit 1
    fi
}

# Function to install dependencies
install_dependencies() {
    echo ""
    echo "📦 Installing dependencies..."
    
    # Upgrade pip first
    echo "⬆️ Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install PDF processing
    echo "📄 Installing PDF processors..."
    pip install PyMuPDF==1.23.26 || echo "⚠️ PyMuPDF failed, will try pdfplumber only"
    pip install pdfplumber==0.10.3
    
    # Install ML core
    echo "🤖 Installing ML libraries..."
    pip install transformers==4.44.2
    pip install datasets==2.16.1
    pip install pandas==2.2.0
    pip install numpy==1.26.4
    
    # Install PyTorch (try CUDA first, fallback to CPU)
    echo "🔥 Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected, installing CUDA version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || \
        pip install torch torchvision torchaudio
    else
        echo "💻 No GPU detected, installing CPU version..."
        pip install torch torchvision torchaudio
    fi
    
    # Install LoRA specific
    echo "🎯 Installing LoRA libraries..."
    pip install peft==0.7.1
    pip install accelerate==0.25.0
    pip install evaluate==0.4.1
    
    # Install optional but useful
    echo "🔧 Installing additional tools..."
    pip install bitsandbytes || echo "⚠️ bitsandbytes skipped (requires CUDA)"
    
    # Install project dependencies
    echo "🏗️ Installing project dependencies..."
    pip install fastapi==0.108.0
    pip install uvicorn[standard]==0.25.0
    pip install redis==5.0.1
    pip install sentence-transformers==3.0.1
    pip install chromadb==0.5.11
    
    echo "✅ All dependencies installed"
}

# Function to test installation
test_installation() {
    echo ""
    echo "🧪 Testing installation..."
    
    # Test critical imports
    python -c "
import sys
print(f'🐍 Python: {sys.version}')

try:
    import PyMuPDF
    print('✅ PyMuPDF: OK')
except ImportError:
    print('⚠️ PyMuPDF: Not available')

try:
    import pdfplumber
    print('✅ pdfplumber: OK')
except ImportError:
    print('❌ pdfplumber: FAILED')
    sys.exit(1)

try:
    import transformers
    print('✅ transformers: OK')
except ImportError:
    print('❌ transformers: FAILED')
    sys.exit(1)

try:
    import torch
    print(f'✅ torch: OK')
    if torch.cuda.is_available():
        print(f'🎮 GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('💻 GPU: Not available (using CPU)')
except ImportError:
    print('❌ torch: FAILED')
    sys.exit(1)

try:
    import peft
    print('✅ peft: OK')
except ImportError:
    print('❌ peft: FAILED')
    sys.exit(1)

print('🎉 All critical imports successful!')
"
    
    if [ $? -eq 0 ]; then
        echo "✅ Installation test passed"
    else
        echo "❌ Installation test failed"
        exit 1
    fi
}

# Function to create activation script
create_activation_script() {
    echo ""
    echo "📝 Creating activation script..."
    
    cat > "$PROJECT_DIR/activate_env.sh" << 'EOF'
#!/bin/bash
# IAbel Environment Activation Script

PROJECT_DIR="/home/lacucaratila/Projetos/IAbel/backend"
VENV_PATH="$PROJECT_DIR/envWSL"

echo "🐍 Activating IAbel environment..."

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    cd "$PROJECT_DIR"
    echo "✅ Environment activated"
    echo "📍 Location: $(pwd)"
    echo "🐍 Python: $(which python)"
    echo ""
    echo "Available commands:"
    echo "  python fine_tuning/test_setup.py     # Test setup"
    echo "  python fine_tuning/train_lora.py     # Train LoRA"
    echo "  python app/main.py                   # Start API"
    echo "  deactivate                           # Exit environment"
else
    echo "❌ Environment not found: $VENV_PATH"
    echo "Run setup_python_pure.sh first"
fi
EOF
    
    chmod +x "$PROJECT_DIR/activate_env.sh"
    echo "✅ Activation script created: activate_env.sh"
}

# Function to show final instructions
show_instructions() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "================================"
    echo ""
    echo "📋 Usage Instructions:"
    echo ""
    echo "1. Activate environment:"
    echo "   source activate_env.sh"
    echo "   # or manually:"
    echo "   source envWSL/bin/activate"
    echo ""
    echo "2. Test setup:"
    echo "   python fine_tuning/test_setup.py"
    echo ""
    echo "3. Train LoRA model:"
    echo "   python fine_tuning/train_lora.py"
    echo ""
    echo "4. Start API server:"
    echo "   python app/main.py"
    echo ""
    echo "5. Deactivate when done:"
    echo "   deactivate"
    echo ""
    echo "💡 Always run 'source activate_env.sh' before working on IAbel!"
}

# Main execution
main() {
    check_python
    cleanup_old_envs
    create_venv
    activate_venv
    install_dependencies
    test_installation
    create_activation_script
    show_instructions
}

# Run main function
main