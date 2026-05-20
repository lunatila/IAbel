#!/bin/bash
# Cleanup Script for IAbel Environments
# Removes duplicate environments and ensures only envWSL is used

echo "🧹 IAbel Environment Cleanup Script"
echo "===================================="

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda not found in PATH"
        echo "💡 If conda is installed, run: source ~/.bashrc"
        echo "💡 Or activate conda manually first"
        exit 1
    fi
    echo "✅ Conda found: $(which conda)"
}

# Function to list current environments
list_environments() {
    echo ""
    echo "📋 Current Conda Environments:"
    echo "------------------------------"
    conda env list
}

# Function to identify envWSL
find_envwsl() {
    echo ""
    echo "🔍 Looking for envWSL..."
    
    if conda env list | grep -q "envWSL"; then
        ENVWSL_PATH=$(conda env list | grep "envWSL" | awk '{print $2}')
        echo "✅ Found envWSL at: $ENVWSL_PATH"
        return 0
    else
        echo "❌ envWSL not found!"
        echo "💡 You may need to create it: conda create -n envWSL python=3.11"
        return 1
    fi
}

# Function to remove duplicate environments
remove_duplicates() {
    echo ""
    echo "🗑️ Removing potential duplicate environments..."
    
    # List of common accidentally created environments
    DUPLICATE_ENVS=("iabel" "lora_env" "iabel_env" "backend" "fine_tuning")
    
    for env in "${DUPLICATE_ENVS[@]}"; do
        if conda env list | grep -q "^$env "; then
            echo "🗑️ Removing duplicate environment: $env"
            conda env remove -n $env -y
        else
            echo "✅ No duplicate environment found: $env"
        fi
    done
}

# Function to clean local virtual environments
clean_local_venvs() {
    echo ""
    echo "🗑️ Cleaning local virtual environments..."
    
    # Check in current directory and subdirectories
    LOCAL_VENVS=("./venv" "./lora_env" "./iabel_env" "./env" "./.venv")
    
    for venv in "${LOCAL_VENVS[@]}"; do
        if [ -d "$venv" ]; then
            echo "🗑️ Removing local venv: $venv"
            rm -rf "$venv"
        else
            echo "✅ No local venv found: $venv"
        fi
    done
}

# Function to clean caches
clean_caches() {
    echo ""
    echo "🧽 Cleaning caches..."
    
    echo "Cleaning conda cache..."
    conda clean --all -y
    
    echo "Cleaning pip cache..."
    if command -v pip &> /dev/null; then
        pip cache purge
    fi
}

# Function to verify envWSL
verify_envwsl() {
    echo ""
    echo "✅ Verifying envWSL..."
    
    # Try to activate envWSL
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate envWSL
    
    if [ "$CONDA_DEFAULT_ENV" = "envWSL" ]; then
        echo "✅ Successfully activated envWSL"
        echo "📍 Environment location: $CONDA_PREFIX"
        echo "🐍 Python location: $(which python)"
        echo "🐍 Python version: $(python --version)"
        
        # List installed packages
        echo ""
        echo "📦 Currently installed packages in envWSL:"
        echo "pip list | head -20"
        pip list | head -20
        
        conda deactivate
    else
        echo "❌ Failed to activate envWSL"
        return 1
    fi
}

# Main execution
main() {
    check_conda
    list_environments
    
    echo ""
    read -p "🤔 Do you want to proceed with cleanup? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find_envwsl
        remove_duplicates
        clean_local_venvs
        clean_caches
        
        echo ""
        echo "🎉 Cleanup completed!"
        echo "===================="
        
        list_environments
        verify_envwsl
        
        echo ""
        echo "📝 Next steps:"
        echo "1. Activate envWSL: conda activate envWSL"
        echo "2. Install dependencies manually"
        echo "3. Run: python fine_tuning/test_setup.py"
        
    else
        echo "🚫 Cleanup cancelled"
    fi
}

# Run main function
main