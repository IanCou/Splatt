# GPU Server Setup Checklist

Quick checklist for setting up your GPU server from scratch.

## ☐ Step 1: Upload Files to Server

```bash
# On your local machine, upload server files
scp -r server/ user@your-gpu-server:/workspace/

# SSH into server
ssh user@your-gpu-server
cd /workspace/server
```

## ☐ Step 2: Run Automated Setup

```bash
# Make setup script executable
chmod +x setup_nerf.sh

# Run installation (takes 10-20 minutes)
./setup_nerf.sh
```

**What this installs:**
- ✅ Conda environment named `nerf`
- ✅ PyTorch with CUDA support
- ✅ tiny-cuda-nn (Gaussian Splatting acceleration)
- ✅ COLMAP (camera pose estimation)
- ✅ nerfstudio (NeRF/Gaussian Splatting framework)
- ✅ API server dependencies (FastAPI, etc.)

## ☐ Step 3: Test Installation

```bash
# Activate environment
conda activate nerf

# Test PyTorch GPU
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
# Should print: GPU: True

# Test nerfstudio
ns-train --help
# Should show help text

# Test COLMAP
colmap -h
# Should show help text
```

## ☐ Step 4: Test Nerf4Dgsplat.py

```bash
# Download a test video (or use your own)
# Then run:
python Nerf4Dgsplat.py \
  --video test_video.mp4 \
  --iterations 500 \
  --fps 2 \
  --output_dir ./test_output

# Should complete without errors
# Check output:
ls test_output/
# Should see: baked_splat.ply, baked_splat_sh.ply, data_prepared/
```

## ☐ Step 5: Configure API Server

```bash
# Create config file
cp .env.example .env

# Generate secure API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Edit .env and paste the key
nano .env
```

**Required .env settings:**
```bash
GPU_API_KEY=your-generated-key-here
WORKSPACE_ROOT=/workspace/jobs
NERF_PYTHON_PATH=/opt/miniforge3/envs/nerf/bin/python
SERVER_PORT=8080
```

## ☐ Step 6: Start API Server

```bash
# Quick test (foreground)
./start.sh

# Or use screen for persistent session
screen -S gpu-api
./start.sh
# Detach: Ctrl+A, D
```

## ☐ Step 7: Test API Server

```bash
# From another terminal (or your local machine)
curl http://your-server-ip:8080/health

# Expected response:
# {"status":"healthy","active_jobs":0,"gpu_available":true,...}
```

## ☐ Step 8: Submit Test Job

```bash
# First upload a video to Filebin (https://filebin.net)
# Copy the URL, then:

curl -X POST http://your-server-ip:8080/jobs \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "filebin_url": "https://filebin.net/xyz/video.mp4",
    "filename": "test.mp4",
    "iterations": 1000
  }'

# Save the job_id from response
# Check status:
curl -H "X-API-Key: your-key" \
  http://your-server-ip:8080/jobs/JOB_ID_HERE/status
```

## ☐ Step 9: Configure Local Backend

On your local machine, update `backend/.env`:

```bash
GPU_API_URL=http://your-server-ip:8080
GPU_API_KEY=same-key-as-gpu-server
```

## ☐ Step 10: End-to-End Test

1. Upload video to Filebin from frontend
2. Submit to local backend via `/process-video-filebin`
3. Monitor progress via `/tasks/{task_id}`
4. Verify results are downloaded

---

## Troubleshooting

### Setup script fails

**Check CUDA:**
```bash
nvidia-smi
nvcc --version
```

**Check conda:**
```bash
conda --version
conda env list
```

**Manual installation:**
See `INSTALLATION.md` for step-by-step manual setup.

### API server won't start

**Check environment:**
```bash
conda activate nerf
python -c "import fastapi, uvicorn"
```

**Check port:**
```bash
netstat -tuln | grep 8080
# If in use, change SERVER_PORT in .env
```

### Jobs fail immediately

**Check Nerf4Dgsplat.py:**
```bash
python Nerf4Dgsplat.py --help
```

**Check GPU memory:**
```bash
nvidia-smi
# If low memory, reduce iterations or resolution
```

### Can't connect from local backend

**Option 1: SSH Tunnel**
```bash
# On local machine
ssh -N -L 8080:localhost:8080 user@gpu-server &
# Use http://localhost:8080 in backend
```

**Option 2: Firewall**
```bash
# On GPU server
sudo ufw allow 8080/tcp
# Or configure cloud provider firewall rules
```

---

## Quick Commands Reference

**Activate environment:**
```bash
conda activate nerf
```

**Start server (foreground):**
```bash
./start.sh
```

**Start server (background):**
```bash
screen -S gpu-api
./start.sh
# Ctrl+A, D to detach
# screen -r gpu-api to reattach
```

**View server logs:**
```bash
# If using systemd
sudo journalctl -u gpu-api -f

# If using screen
screen -r gpu-api
```

**Test GPU:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Check API status:**
```bash
curl http://localhost:8080/health
```

---

## Files You Should Have

After setup, your `server/` directory should contain:

```
server/
├── Nerf4Dgsplat.py          ✅ Your pipeline script
├── main.py                  ✅ API server
├── models.py                ✅ Pydantic models
├── job_processor.py         ✅ Job execution
├── requirements.txt         ✅ Dependencies
├── .env                     📝 (you create this)
├── .env.example             ✅ Config template
├── setup_nerf.sh            ✅ Installation script
├── start.sh                 ✅ Startup script
├── README.md                ✅ Full docs
├── INSTALLATION.md          ✅ Manual setup guide
├── QUICKSTART.md            ✅ Quick start
└── SETUP_CHECKLIST.md       ✅ This file
```

---

## All Done? ✅

Your GPU server is ready when:

- [x] `conda activate nerf` works
- [x] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [x] `ns-train --help` shows nerfstudio help
- [x] `python Nerf4Dgsplat.py --help` works
- [x] `curl http://localhost:8080/health` returns JSON
- [x] Test job completes successfully
- [x] Local backend can submit jobs via API

**Next:** Proceed to Phase 2 (updating local backend to use GPU API)
