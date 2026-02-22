# GPU Server Installation Guide

Complete setup guide for nerfstudio and dependencies on a fresh GPU server.

## Prerequisites Check

Before starting, verify you have:

```bash
# Check CUDA
nvidia-smi
nvcc --version  # Should show CUDA compiler version

# Check conda
conda --version

# Check Python availability
python --version
```

If `nvcc` is not found but `nvidia-smi` works, you may need to install CUDA toolkit (see troubleshooting).

---

## Installation Steps

### Step 1: Create Conda Environment

```bash
# Create a new environment with Python 3.10 (nerfstudio works best with 3.10)
conda create -n nerf python=3.10 -y

# Activate environment
conda activate nerf

# Verify Python version
python --version  # Should show Python 3.10.x
```

### Step 2: Install System Dependencies

These are needed for COLMAP (camera pose estimation):

**For Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev
```

**For CentOS/RHEL (if using):**
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake \
    ninja-build \
    boost-devel \
    eigen3-devel \
    freeimage-devel \
    glog-devel \
    gflags-devel \
    glew-devel \
    qt5-qtbase-devel \
    ffmpeg \
    ffmpeg-devel
```

**If you don't have sudo access:**
Skip this and try Step 3. COLMAP might be installed via conda (see below).

### Step 3: Install PyTorch with CUDA Support

**Check your CUDA version first:**
```bash
nvidia-smi  # Look at top right for CUDA version (e.g., "CUDA Version: 12.1")
```

**Install PyTorch** (adjust for your CUDA version):

For **CUDA 11.8**:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

For **CUDA 12.1**:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

For **CUDA 12.4+** (latest):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Verify PyTorch can see your GPU:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
PyTorch: 2.1.0+cu118
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
```

**If CUDA available is False**, you have a PyTorch/CUDA version mismatch. Reinstall PyTorch with correct CUDA version.

### Step 4: Install TinyCUDA (Required for Gaussian Splatting)

```bash
# Install tiny-cuda-nn (used by nerfstudio for speed)
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

**Common error:** "nvcc not found"

If you get this error:
```bash
# Option 1: Install CUDA toolkit via conda
conda install -c nvidia cuda-toolkit cuda-nvcc -y

# Option 2: Add CUDA to PATH (if installed system-wide)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc to make permanent
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Step 5: Install COLMAP (Camera Pose Estimation)

**Option A: Via conda (easiest if no sudo)**
```bash
conda install -c conda-forge colmap -y
```

**Option B: Build from source (if conda version doesn't work)**
```bash
# Install to local directory (no sudo needed)
cd ~
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build && cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$HOME/.local
ninja install

# Add to PATH
export PATH=$HOME/.local/bin:$PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
```

**Verify COLMAP works:**
```bash
colmap -h
# Should show COLMAP help text
```

### Step 6: Install Nerfstudio

```bash
# Install nerfstudio and all dependencies
pip install nerfstudio

# Verify installation
ns-install-cli
```

**This installs:**
- nerfstudio core
- All supported methods (including splatfacto for Gaussian Splatting)
- CLI tools (`ns-train`, `ns-process-data`, etc.)

**Verify nerfstudio works:**
```bash
ns-train -h
# Should show nerfstudio training options

# Check splatfacto (Gaussian Splatting) is available
ns-train splatfacto -h
```

### Step 7: Install Additional Dependencies

```bash
# For video processing (used by Nerf4Dgsplat.py)
pip install opencv-python-headless

# For PLY file handling
pip install plyfile

# For API server (covered in requirements.txt, but listed here for completeness)
pip install fastapi uvicorn pydantic requests python-dotenv

# Optional: For better progress bars
pip install tqdm rich
```

### Step 8: Verify Complete Installation

Run this comprehensive test:

```bash
python << 'EOF'
import sys
import subprocess

def check(name, cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 or "version" in result.stdout.lower() or "help" in result.stdout.lower():
            print(f"✅ {name}")
            return True
        else:
            print(f"❌ {name}: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

print("Checking installation...\n")

# Python packages
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import tinycudann
    print(f"✅ tiny-cuda-nn")
except ImportError as e:
    print(f"❌ tiny-cuda-nn: {e}")

try:
    import nerfstudio
    print(f"✅ nerfstudio {nerfstudio.__version__}")
except ImportError as e:
    print(f"❌ nerfstudio: {e}")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    import plyfile
    print(f"✅ plyfile")
except ImportError as e:
    print(f"❌ plyfile: {e}")

try:
    import fastapi
    print(f"✅ FastAPI {fastapi.__version__}")
except ImportError as e:
    print(f"❌ FastAPI: {e}")

print("\nSystem tools:")
check("COLMAP", "colmap -h")
check("FFmpeg", "ffmpeg -version")
check("NVCC", "nvcc --version")

print("\nIf all items show ✅, your installation is complete!")
EOF
```

### Step 9: Test with Nerf4Dgsplat.py

```bash
# Upload Nerf4Dgsplat.py to server
cd /workspace/server  # or your chosen directory

# Download a test video
wget https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/tests/data/lego/images/r_0.png -O test_frame.png
# Or use your own short test video

# Run a quick test (if you have a test video)
python Nerf4Dgsplat.py --video test_video.mp4 --iterations 100 --fps 2 --output_dir ./test_output
```

---

## Quick Setup Script

Save this as `setup_nerf.sh` for easy installation:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Nerfstudio Installation Script"
echo "=========================================="

# Create environment
echo "Creating conda environment..."
conda create -n nerf python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate nerf

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch..."
read -p "Enter your CUDA version (11.8, 12.1, or 12.4): " cuda_ver
case $cuda_ver in
    11.8)
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
        ;;
    12.1)
        pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
        ;;
    12.4)
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
        ;;
    *)
        echo "Invalid CUDA version"
        exit 1
        ;;
esac

# Install CUDA toolkit (for nvcc)
echo "Installing CUDA toolkit..."
conda install -c nvidia cuda-toolkit cuda-nvcc -y

# Install tiny-cuda-nn
echo "Installing tiny-cuda-nn..."
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install COLMAP
echo "Installing COLMAP..."
conda install -c conda-forge colmap -y

# Install nerfstudio
echo "Installing nerfstudio..."
pip install nerfstudio
ns-install-cli

# Install other dependencies
echo "Installing additional dependencies..."
pip install opencv-python-headless plyfile
pip install fastapi uvicorn pydantic requests python-dotenv

# Verify
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
colmap -h > /dev/null 2>&1 && echo "✅ COLMAP installed" || echo "❌ COLMAP not found"
ns-train -h > /dev/null 2>&1 && echo "✅ nerfstudio installed" || echo "❌ nerfstudio not found"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "Activate environment with: conda activate nerf"
echo "=========================================="
```

Make it executable and run:
```bash
chmod +x setup_nerf.sh
./setup_nerf.sh
```

---

## Troubleshooting

### "CUDA out of memory"

Your GPU doesn't have enough VRAM. Reduce batch size or iterations:
```bash
python Nerf4Dgsplat.py --video vid.mp4 --iterations 3000  # Lower iterations
```

### "nvcc not found" during tiny-cuda-nn installation

```bash
# Install CUDA toolkit via conda
conda install -c nvidia cuda-toolkit cuda-nvcc -y

# Or add system CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### "COLMAP not found"

```bash
# Try conda installation
conda install -c conda-forge colmap -y

# Verify
which colmap
colmap -h
```

### "ImportError: cannot import name 'packaging'" or similar

```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Reinstall nerfstudio
pip uninstall nerfstudio -y
pip install nerfstudio
```

### PyTorch not detecting GPU

```bash
# Check CUDA version compatibility
nvidia-smi  # Note CUDA version
python -c "import torch; print(torch.version.cuda)"  # Should match

# If mismatch, reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Adjust version
```

### System dependencies missing (no sudo access)

If you can't install system packages, try:
```bash
# Install everything via conda
conda install -c conda-forge colmap ffmpeg cmake ninja -y
```

---

## After Installation

1. **Activate environment** whenever you work:
   ```bash
   conda activate nerf
   ```

2. **Set up GPU server API** (from main README):
   ```bash
   cd /workspace/server
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your settings
   ./start.sh
   ```

3. **Test with a real video**:
   ```bash
   python Nerf4Dgsplat.py --video test.mp4 --iterations 5000 --output_dir ./test_output
   ```

---

## Environment Management

**Activate environment:**
```bash
conda activate nerf
```

**Deactivate:**
```bash
conda deactivate
```

**List all environments:**
```bash
conda env list
```

**Delete environment (if you need to start over):**
```bash
conda deactivate
conda env remove -n nerf -y
```

**Export environment (for backup/sharing):**
```bash
conda activate nerf
conda env export > nerf_environment.yml
```

**Recreate from export:**
```bash
conda env create -f nerf_environment.yml
```

---

## What Gets Installed

| Component | Purpose | Size |
|-----------|---------|------|
| PyTorch + CUDA | GPU deep learning | ~3 GB |
| tiny-cuda-nn | Fast neural networks | ~100 MB |
| nerfstudio | NeRF/Gaussian Splatting | ~500 MB |
| COLMAP | Camera pose estimation | ~50 MB |
| OpenCV | Video processing | ~100 MB |
| FFmpeg | Video encoding/decoding | ~100 MB |

**Total:** ~4-5 GB

---

## Next Steps

✅ Environment set up
✅ Dependencies installed
✅ GPU accessible

Now you can:
1. Test Nerf4Dgsplat.py with a sample video
2. Start the GPU API server (see README.md)
3. Connect your local backend

---

**Questions?** Check the main README.md or troubleshooting sections above.
