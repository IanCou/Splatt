#!/bin/bash
# Automated Nerfstudio Installation Script
# ============================================================================
# Run this on your GPU server to install nerfstudio and all dependencies
# ============================================================================

set -e

echo "=========================================="
echo "Nerfstudio + GPU Server Setup"
echo "=========================================="
echo ""

# Detect CUDA version
echo "Detecting CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
    echo "✅ Detected CUDA $CUDA_VERSION"
else
    echo "❌ nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

# Determine PyTorch CUDA version
if [[ "$CUDA_VERSION" == 11.* ]]; then
    TORCH_CUDA="cu118"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
elif [[ "$CUDA_VERSION" == 12.0 ]] || [[ "$CUDA_VERSION" == 12.1 ]]; then
    TORCH_CUDA="cu121"
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    TORCH_CUDA="cu124"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi

echo "Will install PyTorch for $TORCH_CUDA"
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "✅ conda found: $(conda --version)"
echo ""

# Create environment
echo "=========================================="
echo "Step 1/7: Creating conda environment 'nerf'"
echo "=========================================="

if conda env list | grep -q "^nerf "; then
    echo "⚠️  Environment 'nerf' already exists."
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n nerf -y
        conda create -n nerf python=3.10 -y
    fi
else
    conda create -n nerf python=3.10 -y
fi

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate nerf

echo "✅ Python version: $(python --version)"
echo ""

# Install PyTorch
echo "=========================================="
echo "Step 2/7: Installing PyTorch with CUDA support"
echo "=========================================="

pip install torch==2.1.0 torchvision==0.16.0 --index-url $TORCH_INDEX

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ PyTorch cannot detect GPU. Check CUDA installation."
    exit 1
fi

echo "✅ PyTorch installed with GPU support"
echo ""

# Install CUDA toolkit (for nvcc)
echo "=========================================="
echo "Step 3/7: Installing CUDA toolkit (nvcc)"
echo "=========================================="

conda install -c nvidia cuda-toolkit cuda-nvcc -y

echo "✅ CUDA toolkit installed"
echo ""

# Install tiny-cuda-nn
echo "=========================================="
echo "Step 4/7: Installing tiny-cuda-nn"
echo "=========================================="

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

echo "✅ tiny-cuda-nn installed"
echo ""

# Install COLMAP
echo "=========================================="
echo "Step 5/7: Installing COLMAP"
echo "=========================================="

conda install -c conda-forge colmap -y

if command -v colmap &> /dev/null; then
    echo "✅ COLMAP installed: $(colmap --version 2>&1 | head -1)"
else
    echo "⚠️  COLMAP not found in PATH, but may still work"
fi

echo ""

# Install nerfstudio
echo "=========================================="
echo "Step 6/7: Installing nerfstudio"
echo "=========================================="

pip install nerfstudio

# Install CLI
ns-install-cli

# Verify
if command -v ns-train &> /dev/null; then
    echo "✅ nerfstudio installed"
    ns-train --help > /dev/null 2>&1 && echo "   CLI tools working"
else
    echo "⚠️  ns-train not found. You may need to add it to PATH manually."
fi

echo ""

# Install additional dependencies
echo "=========================================="
echo "Step 7/7: Installing additional dependencies"
echo "=========================================="

pip install \
    opencv-python-headless \
    plyfile \
    fastapi \
    uvicorn[standard] \
    pydantic \
    requests \
    python-dotenv

echo "✅ Additional dependencies installed"
echo ""

# Final verification
echo "=========================================="
echo "Verification"
echo "=========================================="

python << 'VERIFY_EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'installed')
        print(f"✅ {name}: {version}")
        return True
    except ImportError:
        print(f"❌ {name}: not found")
        return False

all_good = True
all_good &= check_import("torch")
all_good &= check_import("tiny-cuda-nn", "tinycudann")
all_good &= check_import("nerfstudio")
all_good &= check_import("opencv", "cv2")
all_good &= check_import("plyfile")
all_good &= check_import("fastapi")
all_good &= check_import("uvicorn")

# Check PyTorch GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU: not available")
    all_good = False

sys.exit(0 if all_good else 1)
VERIFY_EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Installation Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: conda activate nerf"
    echo "2. Test Nerf4Dgsplat.py with a video"
    echo "3. Set up API server (see README.md)"
    echo ""
    echo "Quick test:"
    echo "  python Nerf4Dgsplat.py --help"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "⚠️  Installation had issues"
    echo "=========================================="
    echo ""
    echo "Some packages failed to install."
    echo "Check the output above and see INSTALLATION.md for troubleshooting."
    exit 1
fi
