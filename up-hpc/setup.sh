#!/bin/bash
# Quick setup script for SNN inference on remote cluster
# Run this script once to set up your environment

set -e  # Exit on error

echo "=========================================="
echo "SNN Inference Environment Setup"
echo "=========================================="

# Check if we're on the cluster
if [ -z "$SLURM_SUBMIT_DIR" ]; then
    echo "Note: This doesn't appear to be a SLURM cluster"
    echo "The script will continue, but you may need to adjust settings"
fi

# Create logs directory
echo "Creating logs directory..."
mkdir -p logs
echo "✓ logs/ directory created"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✓ Conda found"
    CONDA_AVAILABLE=true
else
    echo "⚠ Conda not found - you may need to: module load anaconda3"
    CONDA_AVAILABLE=false
fi

# Check Python version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python not found!"
    exit 1
fi

# Offer to create environment
if [ "$CONDA_AVAILABLE" = true ]; then
    read -p "Create a new conda environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Environment name [torch310]: " ENV_NAME
        ENV_NAME=${ENV_NAME:-torch310}
        
        echo "Creating conda environment: $ENV_NAME"
        conda create -n $ENV_NAME python=3.10 -y
        
        echo "Activating environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $ENV_NAME
        
        echo "Installing PyTorch..."
        read -p "Install PyTorch with CUDA support? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        else
            conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        fi
        
        echo "Installing additional packages..."
        pip install -r requirements.txt
        
        echo ""
        echo "✓ Environment setup complete!"
        echo "  To activate: conda activate $ENV_NAME"
    fi
fi

# Check for required files
echo ""
echo "Checking required files..."

FILES_TO_CHECK=(
    "infer.py"
    "spiking_model.py"
    "checkpoint/ckptspiking_model.t7"
)

ALL_PRESENT=true
for file in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing!)"
        ALL_PRESENT=false
    fi
done

if [ "$ALL_PRESENT" = false ]; then
    echo ""
    echo "⚠ Some required files are missing!"
    echo "  Make sure you have:"
    echo "  - infer.py"
    echo "  - spiking_model.py"
    echo "  - checkpoint/ckptspiking_model.t7"
    echo "  - data_setup.py (and other dependencies)"
fi

# Check if data directory exists
echo ""
if [ -d "raw" ] && [ "$(ls -A raw)" ]; then
    echo "✓ Data directory found (raw/)"
else
    echo "⚠ Data directory is empty or missing"
    read -p "Download MNIST dataset? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python -c "from torchvision import datasets; datasets.MNIST('./raw', train=False, download=True)"
        echo "✓ MNIST downloaded"
    fi
fi

# Test imports
echo ""
echo "Testing Python imports..."
python << 'EOF'
import sys

errors = []
try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError:
    print("✗ torch")
    errors.append("torch")

try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")
except ImportError:
    print("✗ torchvision")
    errors.append("torchvision")

try:
    import numpy
    print(f"✓ numpy {numpy.__version__}")
except ImportError:
    print("✗ numpy")
    errors.append("numpy")

try:
    import matplotlib
    print(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("✗ matplotlib")
    errors.append("matplotlib")

if errors:
    print(f"\n⚠ Missing packages: {', '.join(errors)}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✓ All packages installed correctly")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Setup Complete! 🎉"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Review and edit run_infer.sbatch:"
    echo "   - Set correct partition name"
    echo "   - Set correct module loads"
    echo "   - Set correct environment name"
    echo ""
    echo "2. Submit your job:"
    echo "   sbatch run_infer.sbatch"
    echo ""
    echo "3. Monitor your job:"
    echo "   squeue -u \$USER"
    echo "   tail -f logs/infer_<JOB_ID>.out"
    echo ""
else
    echo ""
    echo "⚠ Setup completed with warnings"
    echo "Please install missing packages before running"
fi
