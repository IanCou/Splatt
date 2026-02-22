"""
GPU Server FastAPI Application.

Provides HTTP API for Gaussian Splatting job management:
- Job submission
- Status monitoring
- Result retrieval
- Health checks

Replaces SSH-based orchestration with RESTful API.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional
import asyncio

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models import (
    JobRequest,
    JobStatus,
    JobCreateResponse,
    JobListResponse,
    HealthCheckResponse,
    ErrorResponse,
)
from job_processor import (
    JOBS,
    WORKSPACE_ROOT,
    create_job,
    get_job_status,
    get_all_jobs,
    process_job,
    cleanup_job,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("GPU_API_KEY", "default-dev-key-CHANGE-IN-PRODUCTION")

if API_KEY == "default-dev-key-CHANGE-IN-PRODUCTION":
    logger.warning("⚠️  Using default API key! Set GPU_API_KEY in .env for production!")

# ──────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GPU Gaussian Splatting Server",
    description="Background processing server for Nerf4Dgsplat pipeline",
    version="1.0.0",
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(..., description="API key for authentication")):
    """Verify API key from request header."""
    if x_api_key != API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Provide valid X-API-Key header."
        )
    return x_api_key


# ──────────────────────────────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.

    Returns server status, active job count, and GPU availability.
    Useful for monitoring and load balancing.
    """
    active_jobs = len([j for j in JOBS.values() if j["status"] not in ["completed", "failed"]])
    queued_jobs = len([j for j in JOBS.values() if j["status"] == "queued"])

    # Check GPU availability
    gpu_available = False
    gpu_info = None

    try:
        import torch
        gpu_available = torch.cuda.is_available()

        if gpu_available:
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(0),
            }
    except Exception as e:
        logger.warning(f"Could not check GPU status: {e}")

    status = "healthy" if gpu_available else "degraded"

    return HealthCheckResponse(
        status=status,
        active_jobs=active_jobs,
        queued_jobs=queued_jobs,
        gpu_available=gpu_available,
        gpu_info=gpu_info,
    )


# ──────────────────────────────────────────────────────────────────────
# Job Management
# ──────────────────────────────────────────────────────────────────────

@app.post("/jobs", response_model=JobCreateResponse, dependencies=[Depends(verify_api_key)])
async def create_job_endpoint(request: JobRequest, background_tasks: BackgroundTasks):
    """
    Create a new Gaussian Splatting job.

    The job is queued and processed in the background. Use GET /jobs/{job_id}/status
    to monitor progress.

    Args:
        request: Job parameters (video URL, iterations, etc.)
        background_tasks: FastAPI background task manager

    Returns:
        Job ID and initial status
    """
    job_id = str(uuid.uuid4())

    logger.info(
        f"Creating job {job_id} | task_id={request.task_id} | "
        f"filebin_url={request.filebin_url} | iterations={request.iterations}"
    )

    # Initialize job in tracking system
    create_job(job_id, request)

    # Start processing in background
    background_tasks.add_task(process_job, job_id, request)

    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} created and queued for processing",
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatus, dependencies=[Depends(verify_api_key)])
async def get_job_status_endpoint(job_id: str):
    """
    Get current status of a job.

    Args:
        job_id: Unique job identifier

    Returns:
        Current job status including progress, message, and results (if completed)
    """
    job = get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(**job)


@app.get("/jobs", response_model=JobListResponse, dependencies=[Depends(verify_api_key)])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
):
    """
    List all jobs, optionally filtered by status.

    Args:
        status: Filter by status (queued, training, completed, failed, etc.)
        limit: Maximum number of jobs to return

    Returns:
        List of jobs matching criteria
    """
    all_jobs = get_all_jobs()

    # Filter by status if provided
    if status:
        filtered = [j for j in all_jobs.values() if j["status"] == status]
    else:
        filtered = list(all_jobs.values())

    # Sort by creation time (newest first)
    sorted_jobs = sorted(filtered, key=lambda j: j["created_at"], reverse=True)

    # Apply limit
    limited = sorted_jobs[:limit]

    return JobListResponse(
        jobs=[JobStatus(**j) for j in limited],
        total=len(limited),
    )


@app.delete("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def delete_job_endpoint(job_id: str):
    """
    Delete a job and clean up its workspace.

    Args:
        job_id: Job to delete

    Returns:
        Confirmation message
    """
    job = get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Don't allow deleting active jobs
    if job["status"] not in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete active job (status: {job['status']}). Wait for completion or failure."
        )

    cleanup_job(job_id)

    return {"message": f"Job {job_id} deleted successfully"}


# ──────────────────────────────────────────────────────────────────────
# Result Download
# ──────────────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}/results/{filename}", dependencies=[Depends(verify_api_key)])
async def download_result_file(job_id: str, filename: str):
    """
    Download a result file from a completed job.

    Args:
        job_id: Job ID
        filename: File to download (e.g., baked_splat.ply, transforms.json)

    Returns:
        File download response
    """
    job = get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Validate filename (security: prevent path traversal)
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Determine file path
    workspace = WORKSPACE_ROOT / job_id / "output"

    # Handle special case: transforms.json is in data_prepared subdirectory
    if filename == "transforms.json":
        file_path = workspace / "data_prepared" / filename
    else:
        file_path = workspace / filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found for job {job_id}"
        )

    logger.info(f"Serving file {filename} for job {job_id}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream",
    )


# ──────────────────────────────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────────────────────────────

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="NotFound",
            detail=str(exc.detail) if hasattr(exc, 'detail') else "Resource not found"
        ).dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.exception("Internal server error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred. Check server logs."
        ).dict()
    )


# ──────────────────────────────────────────────────────────────────────
# Startup / Shutdown
# ──────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    logger.info("=" * 70)
    logger.info("GPU Gaussian Splatting Server starting...")
    logger.info(f"Workspace root: {WORKSPACE_ROOT}")
    logger.info(f"API authentication: {'Enabled' if API_KEY != 'default-dev-key-CHANGE-IN-PRODUCTION' else '⚠️  DEVELOPMENT MODE'}")

    # Create workspace directory if it doesn't exist
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️  No GPU detected. Training will be slow!")
    except ImportError:
        logger.warning("⚠️  PyTorch not imported. Cannot check GPU status.")

    logger.info("Server ready!")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Server shutting down...")

    # Count active jobs
    active = [j for j in JOBS.values() if j["status"] not in ["completed", "failed"]]
    if active:
        logger.warning(f"⚠️  {len(active)} jobs still active at shutdown")


# ──────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Configuration
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8080"))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
