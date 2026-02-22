"""
Pydantic models for GPU Server API.

Defines request/response schemas for job submission, status tracking,
and result retrieval.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, List
from datetime import datetime


class JobRequest(BaseModel):
    """Request model for creating a new Gaussian Splatting job."""

    task_id: str = Field(
        description="Unique identifier from the local backend (used for correlation)"
    )
    filebin_url: str = Field(
        description="URL to download the video file from (e.g., Filebin URL)"
    )
    filename: str = Field(
        description="Original filename of the video"
    )
    iterations: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Number of Gaussian Splatting training iterations (100-50000)"
    )
    fps: int = Field(
        default=4,
        ge=1,
        le=30,
        description="Frames per second to extract from video for COLMAP processing"
    )
    start_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Start time within the video (seconds)"
    )
    end_sec: Optional[float] = Field(
        default=None,
        description="End time within the video (seconds). If None, uses full video duration"
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL to notify when job completes (optional)"
    )


class JobStatus(BaseModel):
    """Current status of a processing job."""

    job_id: str = Field(description="Unique job identifier")
    task_id: str = Field(description="Original task ID from backend")
    status: Literal[
        "queued",
        "downloading_video",
        "extracting_frames",
        "running_colmap",
        "training",
        "exporting",
        "completed",
        "failed"
    ] = Field(description="Current processing stage")
    progress: int = Field(
        ge=0,
        le=100,
        description="Progress percentage (0-100), or -1 for failed"
    )
    message: str = Field(description="Human-readable status message")
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'"
    )
    created_at: datetime = Field(description="Job creation timestamp")
    updated_at: datetime = Field(description="Last status update timestamp")
    results: Optional["JobResults"] = Field(
        default=None,
        description="Results metadata (only available when status is 'completed')"
    )


class JobResults(BaseModel):
    """Results metadata for a completed job."""

    transforms_json_url: Optional[str] = Field(
        default=None,
        description="Download URL for transforms.json"
    )
    ply_rgb_url: Optional[str] = Field(
        default=None,
        description="Download URL for baked RGB PLY file"
    )
    ply_sh_url: Optional[str] = Field(
        default=None,
        description="Download URL for baked SH PLY file"
    )
    transforms_json: Optional[Dict] = Field(
        default=None,
        description="Inline transforms.json content (if small enough)"
    )


class JobCreateResponse(BaseModel):
    """Response when creating a new job."""

    job_id: str = Field(description="Unique identifier for the created job")
    status: str = Field(description="Initial status (typically 'queued')")
    message: str = Field(description="Confirmation message")


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    active_jobs: int = Field(description="Number of jobs currently processing")
    queued_jobs: int = Field(description="Number of jobs waiting in queue")
    gpu_available: bool = Field(description="Whether GPU is accessible")
    gpu_info: Optional[Dict] = Field(
        default=None,
        description="GPU information (name, memory, etc.)"
    )


class JobListResponse(BaseModel):
    """Response for listing jobs."""

    jobs: List[JobStatus]
    total: int = Field(description="Total number of jobs")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    job_id: Optional[str] = Field(
        default=None,
        description="Job ID if error is job-specific"
    )
