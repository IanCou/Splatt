"""
Job processor for GPU server.

Handles background execution of Gaussian Splatting jobs including:
- Video download from Filebin
- Nerf4Dgsplat pipeline execution
- Progress tracking and status updates
- Result packaging and storage
"""

import os
import sys
import json
import logging
import shutil
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable
import requests

from models import JobRequest, JobStatus, JobResults

logger = logging.getLogger(__name__)

# In-memory job storage
# In production, replace with Redis or database
JOBS: Dict[str, Dict] = {}

# Configuration
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/workspace/jobs"))
PYTHON_PATH = os.getenv("NERF_PYTHON_PATH", "/opt/miniforge3/envs/nerf/bin/python")
NERF4D_SCRIPT = Path(__file__).parent / "Nerf4Dgsplat.py"


def create_job(job_id: str, request: JobRequest) -> Dict:
    """
    Initialize a new job in the tracking system.

    Args:
        job_id: Unique job identifier
        request: Job request parameters

    Returns:
        Initial job status dictionary
    """
    now = datetime.utcnow()

    job = {
        "job_id": job_id,
        "task_id": request.task_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued, waiting to start...",
        "error": None,
        "created_at": now,
        "updated_at": now,
        "results": None,
        "request": request.dict(),  # Store original request
    }

    JOBS[job_id] = job
    logger.info(f"Created job {job_id} for task {request.task_id}")

    return job


def update_job_status(
    job_id: str,
    status: str,
    progress: int,
    message: str,
    error: Optional[str] = None,
    results: Optional[JobResults] = None
):
    """Update job status in tracking system."""
    if job_id not in JOBS:
        logger.warning(f"Attempted to update non-existent job {job_id}")
        return

    JOBS[job_id].update({
        "status": status,
        "progress": progress,
        "message": message,
        "error": error,
        "updated_at": datetime.utcnow(),
    })

    if results:
        JOBS[job_id]["results"] = results.dict() if hasattr(results, 'dict') else results

    logger.info(f"Job {job_id}: {status} ({progress}%) - {message}")


def get_job_status(job_id: str) -> Optional[Dict]:
    """Retrieve current job status."""
    return JOBS.get(job_id)


def get_all_jobs() -> Dict[str, Dict]:
    """Retrieve all jobs."""
    return JOBS


async def process_job(job_id: str, request: JobRequest):
    """
    Main job processing pipeline (runs in background).

    Steps:
    1. Download video from Filebin
    2. Run Nerf4Dgsplat.py
    3. Package results
    4. Notify callback (if provided)
    5. Clean up workspace

    Args:
        job_id: Unique job identifier
        request: Job request parameters
    """
    workspace = WORKSPACE_ROOT / job_id
    workspace.mkdir(parents=True, exist_ok=True)

    video_path = None
    output_dir = workspace / "output"

    try:
        # ──────────────────────────────────────────────────────────────
        # Step 1: Download video from Filebin
        # ──────────────────────────────────────────────────────────────
        update_job_status(
            job_id, "downloading_video", 5,
            f"Downloading video from {request.filebin_url}..."
        )

        video_path = workspace / request.filename
        download_video(request.filebin_url, video_path, job_id)

        logger.info(f"Downloaded video to {video_path} ({video_path.stat().st_size} bytes)")

        # ──────────────────────────────────────────────────────────────
        # Step 2: Run Nerf4Dgsplat pipeline
        # ──────────────────────────────────────────────────────────────
        update_job_status(
            job_id, "extracting_frames", 10,
            "Starting Gaussian Splatting pipeline..."
        )

        run_nerf4dgsplat_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            iterations=request.iterations,
            fps=request.fps,
            start_sec=request.start_sec,
            end_sec=request.end_sec,
            job_id=job_id,
        )

        # ──────────────────────────────────────────────────────────────
        # Step 3: Package results
        # ──────────────────────────────────────────────────────────────
        update_job_status(
            job_id, "exporting", 95,
            "Packaging results..."
        )

        results = package_results(output_dir, job_id)

        # ──────────────────────────────────────────────────────────────
        # Step 4: Mark as completed
        # ──────────────────────────────────────────────────────────────
        update_job_status(
            job_id, "completed", 100,
            "Pipeline completed successfully!",
            results=results
        )

        logger.info(f"Job {job_id} completed successfully")

        # ──────────────────────────────────────────────────────────────
        # Step 5: Notify callback (if provided)
        # ──────────────────────────────────────────────────────────────
        if request.callback_url:
            notify_callback(request.callback_url, job_id, results)

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        update_job_status(
            job_id, "failed", -1,
            f"Pipeline failed: {str(e)}",
            error=str(e)
        )

        # Notify callback of failure
        if request.callback_url:
            notify_callback(request.callback_url, job_id, None, error=str(e))

    finally:
        # Clean up downloaded video (keep results)
        if video_path and video_path.exists():
            video_path.unlink()
            logger.info(f"Cleaned up video file: {video_path}")


def download_video(url: str, dest_path: Path, job_id: str):
    """
    Download video from URL to local path.

    Args:
        url: Video download URL (e.g., Filebin URL)
        dest_path: Destination file path
        job_id: Job ID for progress updates
    """
    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with dest_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Update progress every 1MB
                    if downloaded % (1024 * 1024) == 0 and total_size > 0:
                        pct = int((downloaded / total_size) * 5) + 5  # 5-10%
                        update_job_status(
                            job_id, "downloading_video", pct,
                            f"Downloading video... {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB"
                        )

        logger.info(f"Downloaded {downloaded} bytes from {url}")

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download video: {e}")


def run_nerf4dgsplat_pipeline(
    video_path: Path,
    output_dir: Path,
    iterations: int,
    fps: int,
    start_sec: float,
    end_sec: Optional[float],
    job_id: str,
):
    """
    Execute Nerf4Dgsplat.py pipeline and parse output for progress.

    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        iterations: Number of training iterations
        fps: Frames per second for extraction
        start_sec: Video start time
        end_sec: Video end time (None = full duration)
        job_id: Job ID for progress updates
    """
    # Build command
    cmd = [
        str(PYTHON_PATH),
        str(NERF4D_SCRIPT),
        "--video", str(video_path),
        "--output_dir", str(output_dir),
        "--iterations", str(iterations),
        "--fps", str(fps),
        "--start_sec", str(start_sec),
    ]

    if end_sec is not None:
        cmd.extend(["--end_sec", str(end_sec)])

    logger.info(f"Executing: {' '.join(cmd)}")

    # Execute and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=NERF4D_SCRIPT.parent,
    )

    # Parse output in real-time for progress updates
    step_regex = re.compile(r"[Ss]tep\s+(\d+)[/\s]+(\d+)")

    for line in process.stdout:
        line = line.strip()
        if not line:
            continue

        logger.debug(f"[Nerf4D] {line}")

        # Update status based on output patterns
        if "Extracting Frames" in line or "Extracting frames" in line:
            update_job_status(job_id, "extracting_frames", 15, "Extracting frames from video...")

        elif "Running COLMAP" in line or "COLMAP" in line:
            update_job_status(job_id, "running_colmap", 30, "Running COLMAP pose estimation...")

        elif "Training" in line or "Step" in line or "Iter" in line:
            # Try to parse step numbers
            match = step_regex.search(line)
            if match:
                current_step = int(match.group(1))
                total_steps = int(match.group(2))

                if total_steps > 0:
                    # Map training progress to 30-85%
                    training_pct = (current_step / total_steps)
                    overall_pct = 30 + int(training_pct * 55)

                    update_job_status(
                        job_id, "training", overall_pct,
                        f"Training Gaussian Splat... Step {current_step}/{total_steps}"
                    )
            else:
                # Generic training message
                update_job_status(job_id, "training", 50, "Training Gaussian Splat...")

        elif "Exporting" in line or "Baking" in line or "Export" in line:
            update_job_status(job_id, "exporting", 90, "Exporting PLY files...")

        elif "Pipeline complete" in line or "🎉" in line:
            update_job_status(job_id, "exporting", 95, "Pipeline complete, finalizing...")

    # Wait for process to complete
    exit_code = process.wait()

    if exit_code != 0:
        raise RuntimeError(f"Nerf4Dgsplat.py failed with exit code {exit_code}")

    logger.info(f"Nerf4Dgsplat pipeline completed successfully for job {job_id}")


def package_results(output_dir: Path, job_id: str) -> JobResults:
    """
    Package results for delivery.

    Args:
        output_dir: Directory containing pipeline outputs
        job_id: Job ID (used for constructing URLs)

    Returns:
        JobResults with download URLs
    """
    results = JobResults()

    # Check for transforms.json
    transforms_path = output_dir / "data_prepared" / "transforms.json"
    if transforms_path.exists():
        # Read and include inline (it's typically small)
        try:
            results.transforms_json = json.loads(transforms_path.read_text())
            results.transforms_json_url = f"/jobs/{job_id}/results/transforms.json"
            logger.info(f"Found transforms.json: {transforms_path}")
        except Exception as e:
            logger.warning(f"Could not read transforms.json: {e}")

    # Check for PLY files
    ply_rgb = output_dir / "baked_splat.ply"
    ply_sh = output_dir / "baked_splat_sh.ply"

    if ply_rgb.exists():
        results.ply_rgb_url = f"/jobs/{job_id}/results/baked_splat.ply"
        logger.info(f"Found RGB PLY: {ply_rgb} ({ply_rgb.stat().st_size} bytes)")

    if ply_sh.exists():
        results.ply_sh_url = f"/jobs/{job_id}/results/baked_splat_sh.ply"
        logger.info(f"Found SH PLY: {ply_sh} ({ply_sh.stat().st_size} bytes)")

    return results


def notify_callback(callback_url: str, job_id: str, results: Optional[JobResults] = None, error: Optional[str] = None):
    """
    Notify the local backend via webhook when job completes.

    Args:
        callback_url: URL to POST notification to
        job_id: Job ID
        results: Job results (if successful)
        error: Error message (if failed)
    """
    try:
        payload = {
            "job_id": job_id,
            "status": "completed" if results else "failed",
            "error": error,
        }

        if results:
            payload["results"] = results.dict() if hasattr(results, 'dict') else results

        response = requests.post(callback_url, json=payload, timeout=30)
        response.raise_for_status()

        logger.info(f"Successfully notified callback {callback_url} for job {job_id}")

    except Exception as e:
        logger.warning(f"Failed to notify callback {callback_url}: {e}")


def cleanup_job(job_id: str):
    """
    Clean up job workspace and remove from tracking.

    Args:
        job_id: Job ID to clean up
    """
    # Remove workspace directory
    workspace = WORKSPACE_ROOT / job_id
    if workspace.exists():
        shutil.rmtree(workspace)
        logger.info(f"Removed workspace: {workspace}")

    # Remove from tracking
    if job_id in JOBS:
        del JOBS[job_id]
        logger.info(f"Removed job {job_id} from tracking")
