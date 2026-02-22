# GPU Server - Quick Start Guide

Get the GPU API server running in 5 minutes.

## Step 1: Setup (On GPU Server)

```bash
# Navigate to server directory
cd /workspace/server  # or wherever you placed the files

# Install dependencies (in your nerf environment)
source /opt/miniforge3/envs/nerf/bin/activate
pip install -r requirements.txt

# Create configuration
cp .env.example .env
nano .env
```

**Edit `.env` and set:**
```bash
GPU_API_KEY=your-secret-key-here  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
WORKSPACE_ROOT=/workspace/jobs
NERF_PYTHON_PATH=/opt/miniforge3/envs/nerf/bin/python
```

## Step 2: Start Server

```bash
# Option A: Quick start script
./start.sh

# Option B: Direct uvicorn
python main.py

# Option C: Screen session (recommended for remote servers)
screen -S gpu-api
./start.sh
# Detach: Ctrl+A, D
```

## Step 3: Test It Works

```bash
# Health check (from anywhere)
curl http://your-gpu-ip:8080/health

# Expected output:
# {"status":"healthy","active_jobs":0,"queued_jobs":0,"gpu_available":true,...}
```

## Step 4: Test a Job

```bash
# First, upload a test video to Filebin and get the URL
# Then submit a job:

curl -X POST http://your-gpu-ip:8080/jobs \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "filebin_url": "https://filebin.net/xyz/video.mp4",
    "filename": "test.mp4",
    "iterations": 1000,
    "fps": 4
  }'

# Save the job_id from response, then check status:
curl -H "X-API-Key: your-secret-key-here" \
  http://your-gpu-ip:8080/jobs/JOB_ID_HERE/status
```

## Step 5: Configure Local Backend

Update your local backend `.env`:

```bash
GPU_API_URL=http://your-gpu-ip:8080
GPU_API_KEY=your-secret-key-here
```

That's it! Your GPU server is now ready to receive jobs from your local backend.

## Troubleshooting

**Port 8080 already in use:**
```bash
# Change port in .env
SERVER_PORT=8081
```

**Can't connect from local backend:**
```bash
# Option 1: Port forwarding on router
# Option 2: SSH tunnel
ssh -N -L 8080:localhost:8080 user@gpu-server &
# Then use http://localhost:8080 in local backend

# Option 3: Cloudflare Tunnel (best for production)
```

**Jobs fail immediately:**
```bash
# Check Nerf4Dgsplat.py works
python Nerf4Dgsplat.py --help

# Check GPU
nvidia-smi
```

## Next Steps

- Set up systemd service for auto-start (see README.md)
- Configure HTTPS with reverse proxy
- Set up monitoring/alerts
- Test with real video from your workflow

## Files Created

```
server/
├── main.py              ✅ FastAPI app
├── models.py            ✅ Pydantic models
├── job_processor.py     ✅ Job execution
├── Nerf4Dgsplat.py     ✅ (your existing script)
├── requirements.txt     ✅ Dependencies
├── .env.example         ✅ Config template
├── .env                 📝 (you create this)
├── start.sh             ✅ Startup script
├── README.md            ✅ Full documentation
└── QUICKSTART.md        ✅ This file
```

---

**Ready!** Your GPU server is now a modern HTTP API instead of SSH-based orchestration. 🚀
