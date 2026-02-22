# GPU Server - Gaussian Splatting API

FastAPI-based HTTP server for processing Gaussian Splatting jobs on GPU hardware. Replaces SSH-based orchestration with a modern RESTful API.

## Overview

This server provides a background processing pipeline for video-to-Gaussian-Splat conversion:

1. **Receives job requests** via HTTP API with Filebin video URLs
2. **Downloads videos** from Filebin (much faster than SFTP)
3. **Runs Nerf4Dgsplat.py** pipeline (frame extraction, COLMAP, training)
4. **Tracks progress** and provides real-time status updates
5. **Serves results** for download (PLY files, transforms.json)

## Architecture

```
Local Backend → POST /jobs → GPU Server
Local Backend ← GET /jobs/{id}/status ← GPU Server (polling)
Local Backend ← GET /jobs/{id}/results/file.ply ← GPU Server
```

## Prerequisites

- **Python 3.9+** with nerfstudio environment
- **CUDA-compatible GPU** (required for training)
- **Nerf4Dgsplat.py** in the same directory
- **nerfstudio** and dependencies installed

## Installation

### 1. Install Dependencies

```bash
# Activate your nerfstudio environment
source /opt/miniforge3/envs/nerf/bin/activate  # or conda activate nerf

# Install API server dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and set:
# - GPU_API_KEY (generate a secure key!)
# - WORKSPACE_ROOT (where jobs are stored)
# - NERF_PYTHON_PATH (path to python in your nerf environment)
nano .env
```

**Generate secure API key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Verify Installation

```bash
# Test that Nerf4Dgsplat.py is accessible
ls -l Nerf4Dgsplat.py

# Test GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

## Running the Server

### Development (foreground)

```bash
# Activate environment
source /opt/miniforge3/envs/nerf/bin/activate

# Run server
python main.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Production (systemd service)

Create `/etc/systemd/system/gpu-api.service`:

```ini
[Unit]
Description=GPU Gaussian Splatting API
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/workspace/server
Environment="PATH=/opt/miniforge3/envs/nerf/bin"
EnvironmentFile=/workspace/server/.env
ExecStart=/opt/miniforge3/envs/nerf/bin/uvicorn main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable gpu-api
sudo systemctl start gpu-api
sudo systemctl status gpu-api
```

View logs:

```bash
sudo journalctl -u gpu-api -f
```

### Production (screen session)

```bash
# Start in detached screen
screen -S gpu-api
source /opt/miniforge3/envs/nerf/bin/activate
python main.py

# Detach: Ctrl+A, D
# Reattach: screen -r gpu-api
```

## API Documentation

### Authentication

All endpoints (except `/health`) require API key authentication:

```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8080/jobs
```

### Endpoints

#### `GET /health`

Health check (no auth required).

**Response:**
```json
{
  "status": "healthy",
  "active_jobs": 2,
  "queued_jobs": 1,
  "gpu_available": true,
  "gpu_info": {
    "name": "NVIDIA A100",
    "memory_total": 42949672960
  }
}
```

#### `POST /jobs`

Create a new Gaussian Splatting job.

**Request:**
```json
{
  "task_id": "abc-123",
  "filebin_url": "https://filebin.net/xyz/video.mp4",
  "filename": "construction_site.mp4",
  "iterations": 5000,
  "fps": 4,
  "start_sec": 0.0,
  "end_sec": null
}
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Job created and queued for processing"
}
```

#### `GET /jobs/{job_id}/status`

Get current job status.

**Response:**
```json
{
  "job_id": "550e8400-...",
  "task_id": "abc-123",
  "status": "training",
  "progress": 65,
  "message": "Training Gaussian Splat... Step 3250/5000",
  "error": null,
  "created_at": "2024-02-22T10:30:00",
  "updated_at": "2024-02-22T10:35:30",
  "results": null
}
```

**Status values:**
- `queued` - Waiting to start
- `downloading_video` - Downloading from Filebin
- `extracting_frames` - Extracting video frames
- `running_colmap` - Camera pose estimation
- `training` - Training Gaussian Splat model
- `exporting` - Exporting PLY files
- `completed` - Finished successfully
- `failed` - Error occurred

#### `GET /jobs`

List all jobs (with optional filtering).

**Query params:**
- `status` (optional) - Filter by status
- `limit` (default: 100) - Max results

**Response:**
```json
{
  "jobs": [...],
  "total": 42
}
```

#### `GET /jobs/{job_id}/results/{filename}`

Download a result file.

**Filenames:**
- `transforms.json` - Camera transforms
- `baked_splat.ply` - RGB Gaussian Splat
- `baked_splat_sh.ply` - SH Gaussian Splat (higher quality)

**Response:** Binary file download

#### `DELETE /jobs/{job_id}`

Delete a completed/failed job and clean up files.

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

## Testing

### Test with curl

**Health check:**
```bash
curl http://localhost:8080/health
```

**Submit job:**
```bash
curl -X POST http://localhost:8080/jobs \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "filebin_url": "https://filebin.net/abc/video.mp4",
    "filename": "test.mp4",
    "iterations": 1000,
    "fps": 4
  }'
```

**Check status:**
```bash
JOB_ID="550e8400-e29b-41d4-a716-446655440000"
curl -H "X-API-Key: your-api-key" \
  http://localhost:8080/jobs/$JOB_ID/status
```

**Download result:**
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8080/jobs/$JOB_ID/results/baked_splat.ply \
  -o result.ply
```

## Network Configuration

### Option 1: Direct Access (Port Forwarding)

Forward port 8080 on your router/firewall to the GPU server.

Local backend connects to: `http://your-public-ip:8080`

### Option 2: SSH Tunnel (Secure)

Create persistent tunnel from local machine:

```bash
ssh -N -L 8080:localhost:8080 user@gpu-server &
```

Local backend connects to: `http://localhost:8080`

### Option 3: Cloudflare Tunnel (Production)

```bash
# On GPU server
cloudflared tunnel create gpu-api
cloudflared tunnel route dns gpu-api gpu.yourdomain.com
cloudflared tunnel run gpu-api
```

Local backend connects to: `https://gpu.yourdomain.com`

## Monitoring

### Check server status

```bash
curl http://localhost:8080/health | jq
```

### View active jobs

```bash
curl -H "X-API-Key: your-key" \
  http://localhost:8080/jobs?status=training | jq
```

### Watch logs (systemd)

```bash
sudo journalctl -u gpu-api -f --since "10 minutes ago"
```

### Disk usage

```bash
du -sh /workspace/jobs/*
```

## Troubleshooting

### Server won't start

**Check Python environment:**
```bash
which python
python -c "import fastapi, uvicorn"
```

**Check port availability:**
```bash
netstat -tuln | grep 8080
```

### Jobs fail immediately

**Check Nerf4Dgsplat.py:**
```bash
ls -l Nerf4Dgsplat.py
python Nerf4Dgsplat.py --help
```

**Check GPU:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Video download fails

**Test Filebin URL manually:**
```bash
curl -I https://filebin.net/xyz/video.mp4
```

**Check network connectivity:**
```bash
ping filebin.net
```

### Out of disk space

**Clean up old jobs:**
```bash
# Via API
curl -X DELETE -H "X-API-Key: your-key" \
  http://localhost:8080/jobs/{job_id}

# Manually
rm -rf /workspace/jobs/old-job-id
```

## Security Best Practices

1. **Change default API key** - Never use the default key in production
2. **Use HTTPS** - Set up reverse proxy (nginx) or Cloudflare Tunnel
3. **Firewall rules** - Only allow connections from your backend IP
4. **Regular updates** - Keep dependencies updated
5. **Monitor logs** - Set up alerts for errors/unauthorized access

## Performance Tuning

### Concurrent jobs

Currently processes one job at a time. To enable concurrent processing, modify `job_processor.py` to use a job queue (e.g., with `asyncio.Queue`).

### Faster downloads

Filebin is already fast, but you can:
- Increase chunk size in `download_video()`
- Use aria2c for parallel downloads

### Training speed

- Reduce iterations for faster (lower quality) results
- Adjust FPS (fewer frames = faster COLMAP)
- Use smaller video clips

## File Structure

```
server/
├── main.py              # FastAPI application
├── models.py            # Pydantic request/response models
├── job_processor.py     # Background job execution logic
├── Nerf4Dgsplat.py     # Gaussian Splatting pipeline
├── requirements.txt     # Python dependencies
├── .env                 # Configuration (create from .env.example)
├── .env.example         # Example configuration
└── README.md           # This file
```

## Support

For issues:
1. Check logs: `journalctl -u gpu-api -n 100`
2. Test manually: `python Nerf4Dgsplat.py --video test.mp4`
3. Verify API key and network connectivity

## License

Same as parent project (Splatt).
