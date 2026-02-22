import base64
import io
import json
import logging
import math
import os
import re
import tempfile
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import cv2
import tempfile
import uuid
from typing import List, Optional, Dict
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
from pydantic import BaseModel, Field
from supabase import create_client, Client

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from app.services.worker_service import run_remote_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas for structured Gemini output
# ---------------------------------------------------------------------------

class DetectionItem(BaseModel):
    object_type: str = Field(
        description="Category of the object: equipment, materials, workers, vehicles, or structures"
    )
    description: str = Field(description="Brief description of the detected object")
    distance_estimate: Optional[float] = Field(
        default=None,
        description=(
            "Estimated distance from camera to the object in meters (as a decimal number). "
            "Use visual cues like object size, detail visibility, and spatial context. "
            "For reference: nearby objects (0-10m), moderate distance (10-20m), far objects (20-100m+). "
            "Provide your best numerical estimate."
        )
    )


class FrameDetections(BaseModel):
    detections: List[DetectionItem] = Field(
        default_factory=list,
        description="All construction objects detected in the image",
    )


class FilebinVideoRequest(BaseModel):
    filebin_url: str = Field(description="Filebin URL where the video is hosted")
    filename: str = Field(description="Original filename of the video")


# ---------------------------------------------------------------------------
# LangChain model singletons (initialised once at startup)
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.1)
structured_llm = llm.with_structured_output(FrameDetections)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)

VIDEOS_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "videos")
# Supabase table + match function for pgvector similarity search.
# Requires the detection_embeddings table and match_detection_embeddings
# function to be created in your Supabase project (see README).
EMBEDDINGS_TABLE = "video_frame_descriptions"
EMBEDDINGS_MATCH_FN = "match_video_frame_descriptions"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
from construction_safety import ConstructionSafetyRAG
from pydantic import BaseModel

from llm_service import get_gemini_service
from models import SceneQueryResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Splatt VLM Backend")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str
    group_id: Optional[str] = None
    descriptors: Optional[List[Dict]] = None  # Array of video descriptor objects

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/video")
async def process_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    if not supabase:
        raise HTTPException(
            status_code=500,
            detail="Supabase is not configured. Please set SUPABASE_URL and SUPABASE_KEY in .env",
        )

    video_id = str(uuid.uuid4())
    temp_video_path = None

    logger.info(
        "Processing video | video_id=%s filename=%s content_type=%s",
        video_id, file.filename, file.content_type,
    )

    try:
        contents = await file.read()
        logger.info("Read %d bytes from upload | video_id=%s", len(contents), video_id)

        # Write to a temp file for frame extraction
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_video_path = tmp.name
        logger.info("Wrote temp file | path=%s", temp_video_path)

        # Analyze video frames with Gemini via LangChain
        logger.info("Starting frame analysis | video_id=%s", video_id)
        detections = await analyze_video(temp_video_path, video_id)
        logger.info("Frame analysis complete | video_id=%s detections=%d", video_id, len(detections))

        # Persist metadata and detections to Supabase
        video_metadata = {
            "id": video_id,
            "filename": file.filename,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "total_detections": len(detections),
        }

        logger.info("Inserting video metadata | video_id=%s", video_id)
        supabase.table("videos").insert(video_metadata).execute()

        if detections:
            logger.info("Inserting %d detections | video_id=%s", len(detections), video_id)
            supabase.table("detections").insert(detections).execute()

            # Store detection embeddings in Supabase for semantic search
            await _store_detection_embeddings(detections, video_id)

        logger.info("Request complete | video_id=%s total_detections=%d", video_id, len(detections))
        return JSONResponse(content={
            "video_id": video_id,
            "filename": file.filename,
            "total_detections": len(detections),
            "detections": detections,
        })

    except Exception as e:
        logger.exception("Failed to process video | video_id=%s error=%s", video_id, e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
            logger.debug("Cleaned up temp file | path=%s", temp_video_path)


TEMP_DIR = Path("./temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# In-memory store for background task statuses
# In production, this would be Redis or Supabase
TASK_STATUSES: Dict[str, Dict] = {}

def update_task_status(task_id: str, status: str, progress: int, result_files: dict = None):
    """Callback for worker_service to update task state."""
    entry = {
        "status": status,
        "progress": progress,
        "updated_at": datetime.utcnow().isoformat()
    }
    if result_files:
        entry["result_files"] = result_files
    TASK_STATUSES[task_id] = entry
    logger.info("Task %s status update: %s (%d%%)", task_id, status, progress)

async def full_pipeline_task(task_id: str, local_path: str, filename: str):
    """Background task for Gemini analysis and then remote Gaussian Splatting (SFTP upload)."""
    try:
        # 0. Initial Status
        update_task_status(task_id, "Initializing...", 0)

        # 1. Gemini Analysis (0-9%)
        update_task_status(task_id, "Analyzing video with Gemini...", 2)
        logger.info("Starting background frame analysis | task_id=%s", task_id)
        detections = await analyze_video(local_path, task_id)

        # Persist metadata and detections to Supabase
        update_task_status(task_id, "Saving detections to Supabase...", 6)
        video_metadata = {
            "id": task_id,
            "filename": filename,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "total_detections": len(detections),
        }

        if supabase:
            supabase.table("videos").insert(video_metadata).execute()
            if detections:
                supabase.table("detections").insert(detections).execute()
        # Embeddings are stored later, after coordinates are backfilled from transforms.json

        logger.info("Gemini analysis complete | task_id=%s", task_id)
        update_task_status(task_id, "Handing off to remote worker...", 9)

        # 2. Remote Splatting (10-100%)
        # run_remote_pipeline is blocking (SSH) so run in thread pool.
        # It reports its own progress via the on_progress callback (10 → 100%).
        # It also cleans up local_path when done.
        logger.info("Starting remote splatting pipeline | task_id=%s", task_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: run_remote_pipeline(task_id, local_path, on_progress=update_task_status)
        )

        # 3. Backfill coordinates + store embeddings now that transforms.json is available.
        # run_remote_pipeline downloads the file to ./results/<task_id>/transforms.json on success.
        transforms_path = Path(f"./results/{task_id}/transforms.json")
        if transforms_path.exists():
            logger.info("Loading transforms.json for coordinate backfill | task_id=%s", task_id)
            with open(transforms_path) as f:
                task_transforms = json.load(f)
            filled_detections = await loop.run_in_executor(
                None,
                lambda: _backfill_coordinates_sync(task_id, task_transforms)
            )
            if filled_detections:
                await _store_detection_embeddings(filled_detections, task_id)
        else:
            logger.warning(
                "transforms.json not found after pipeline, skipping coordinate backfill and embeddings | task_id=%s path=%s",
                task_id, transforms_path
            )

    except Exception as e:
        logger.exception("Background pipeline failed | task_id=%s error=%s", task_id, e)
        update_task_status(task_id, f"Failed: {str(e)}", -1)
        if os.path.exists(local_path):
            os.unlink(local_path)


async def full_pipeline_task_filebin(task_id: str, filebin_url: str, filename: str):
    """Background task for Gemini analysis and then remote Gaussian Splatting (Filebin download)."""
    local_temp_path = None
    try:
        # 0. Initial Status
        update_task_status(task_id, "Initializing...", 0)

        # 1. Download from Filebin for Gemini Analysis (0-5%)
        update_task_status(task_id, "Downloading video from Filebin for analysis...", 1)
        logger.info("Downloading from Filebin | url=%s task_id=%s", filebin_url, task_id)

        # Download to temp file for local Gemini analysis
        local_temp_path = TEMP_DIR / f"{task_id}_temp.mp4"
        import requests
        response = requests.get(filebin_url, stream=True, timeout=600)
        response.raise_for_status()

        with local_temp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Downloaded from Filebin | path=%s", local_temp_path)
        update_task_status(task_id, "Video downloaded", 5)

        # 2. Gemini Analysis (5-9%)
        update_task_status(task_id, "Analyzing video with Gemini...", 6)
        logger.info("Starting background frame analysis | task_id=%s", task_id)
        detections = await analyze_video(str(local_temp_path), task_id)

        # Persist metadata and detections to Supabase
        update_task_status(task_id, "Saving detections to Supabase...", 8)
        video_metadata = {
            "id": task_id,
            "filename": filename,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "total_detections": len(detections),
        }

        if supabase:
            supabase.table("videos").insert(video_metadata).execute()
            if detections:
                supabase.table("detections").insert(detections).execute()
        # Embeddings are stored later, after coordinates are backfilled from transforms.json

        logger.info("Gemini analysis complete | task_id=%s", task_id)

        # Clean up local temp file (we'll use Filebin URL for remote)
        if local_temp_path and local_temp_path.exists():
            local_temp_path.unlink()
            logger.info("Cleaned up local temp file after analysis")

        update_task_status(task_id, "Handing off to remote worker...", 9)

        # 3. Remote Splatting via Filebin URL (10-100%)
        # Worker downloads directly from Filebin (much faster than SFTP!)
        logger.info("Starting remote splatting pipeline with Filebin | task_id=%s", task_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: run_remote_pipeline(task_id, filebin_url=filebin_url, on_progress=update_task_status)
        )

        # 4. Backfill coordinates + store embeddings now that transforms.json is available.
        transforms_path = Path(f"./results/{task_id}/transforms.json")
        if transforms_path.exists():
            logger.info("Loading transforms.json for coordinate backfill | task_id=%s", task_id)
            with open(transforms_path) as f:
                task_transforms = json.load(f)
            filled_detections = await loop.run_in_executor(
                None,
                lambda: _backfill_coordinates_sync(task_id, task_transforms)
            )
            if filled_detections:
                await _store_detection_embeddings(filled_detections, task_id)
        else:
            logger.warning(
                "transforms.json not found after pipeline, skipping coordinate backfill and embeddings | task_id=%s path=%s",
                task_id, transforms_path
            )

    except Exception as e:
        logger.exception("Background pipeline failed | task_id=%s error=%s", task_id, e)
        update_task_status(task_id, f"Failed: {str(e)}", -1)
        if local_temp_path and local_temp_path.exists():
            local_temp_path.unlink()

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a background task."""
    if task_id not in TASK_STATUSES:
        raise HTTPException(status_code=404, detail="Task not found")
    return TASK_STATUSES[task_id]


@app.get("/results/{task_id}/{filename}")
async def get_result_file(task_id: str, filename: str):
    """Serve a result file (PLY or transforms.json) for a completed task."""
    from fastapi.responses import FileResponse

    # Only allow known result filenames to prevent path traversal
    allowed_files = {"baked_splat.ply", "baked_splat_sh.ply", "transforms.json"}
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid filename")

    result_path = Path(f"./results/{task_id}/{filename}")
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    media_type = "application/json" if filename.endswith(".json") else "application/octet-stream"
    return FileResponse(str(result_path), media_type=media_type, filename=filename)

@app.post("/process-video")
async def process_video_pipeline(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Legacy endpoint: Upload video file directly (slower via SFTP)."""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    task_id = str(uuid.uuid4())
    local_path = TEMP_DIR / f"{task_id}_{file.filename}"

    with local_path.open("wb") as buffer:
        import shutil
        contents = await file.read()
        buffer.write(contents)

    # Run the full pipeline (Gemini + SSH orchestration) in the background
    background_tasks.add_task(full_pipeline_task, task_id, str(local_path), file.filename)

    return {
        "task_id": task_id,
        "message": "Processing started. Gemini analysis followed by remote Gaussian Splatting."
    }


@app.post("/process-video-filebin")
async def process_video_filebin(background_tasks: BackgroundTasks, request: FilebinVideoRequest):
    """Fast endpoint: Provide Filebin URL for GPU server to download directly."""
    task_id = str(uuid.uuid4())

    logger.info(
        "Processing video from Filebin | task_id=%s filebin_url=%s filename=%s",
        task_id, request.filebin_url, request.filename
    )

    # Run the full pipeline with Filebin URL (Gemini + SSH orchestration)
    background_tasks.add_task(
        full_pipeline_task_filebin,
        task_id,
        request.filebin_url,
        request.filename
    )

    return {
        "task_id": task_id,
        "message": "Processing started. Video will be downloaded from Filebin for analysis and splatting."
    }


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    prompt: Optional[str] = "Describe this image in detail.",
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}},
        ])
        response = llm.invoke([message])

        return JSONResponse(content={
            "filename": file.filename,
            "analysis": response.content,
        })
    except Exception as e:
        print(f"[ERROR] Failed to process image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")


# In-memory storage for the hackathon
class State:
    groups: List[Dict] = []
    videos: List[Dict] = []
    # Store summarized scene data per group
    scenes: Dict[str, str] = {}
    # Store full video descriptors per video ID
    video_descriptors: Dict[str, Any] = {}

state = State()
safety_rag = ConstructionSafetyRAG()

def extract_keyframes(video_path: str, interval_sec: int = 2) -> List[Image.Image]:
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    interval_frames = int(fps * interval_sec)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval_frames == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            # Resize for Gemini efficiency
            img.thumbnail((800, 800))
            keyframes.append(img)
        frame_count += 1
        
        # Limit to 10 frames per video for the demo
        if len(keyframes) >= 10:
            break
            
    cap.release()
    return keyframes

@app.get("/api/videos/groups")
async def get_groups():
    """
    Get all video groups with embedded video descriptors.

    Returns groups with full descriptor data for each video,
    enabling frontend to store and use descriptor information.
    """
    # Enhance groups with descriptor data
    enhanced_groups = []
    for group in state.groups:
        enhanced_group = group.copy()
        enhanced_videos = []

        for video in group.get("videos", []):
            enhanced_video = video.copy()
            video_id = video.get("id")

            # Add descriptor if available
            if video_id and video_id in state.video_descriptors:
                descriptor = state.video_descriptors[video_id]
                enhanced_video["descriptor"] = descriptor.to_json_dict()
                print(f"[BACKEND] Added descriptor for video {video_id} to response")
            else:
                enhanced_video["descriptor"] = None
                print(f"[BACKEND] No descriptor found for video {video_id}")

            enhanced_videos.append(enhanced_video)

        enhanced_group["videos"] = enhanced_videos
        enhanced_groups.append(enhanced_group)

    print(f"[BACKEND] Returning {len(enhanced_groups)} groups with descriptor data")
    return {"groups": enhanced_groups}

@app.post("/api/videos/groups")
async def upload_video(video: UploadFile = File(...)):
    logger.info(f"Received upload request for video: {video.filename}")
    if not video.content_type.startswith("video/"):
        logger.warning(f"Invalid content type: {video.content_type}")
        raise HTTPException(status_code=400, detail="File must be a video.")

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            logger.info(f"Streaming video to temp file: {tmp.name}")
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Extracting keyframes from: {tmp_path}")
        keyframes = extract_keyframes(tmp_path)
        logger.info(f"Extracted {len(keyframes)} keyframes.")

        # Generate video ID
        video_id = str(uuid.uuid4())

        # Comprehensive analysis with Gemini via LangChain
        logger.info("Sending keyframes to Gemini for comprehensive analysis...")
        gemini = get_gemini_service()

        # Get video duration (approximation based on frames)
        cap_temp = cv2.VideoCapture(tmp_path)
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        frame_count = cap_temp.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = frame_count / fps if fps > 0 else 0
        cap_temp.release()

        # Perform comprehensive analysis
        video_descriptor = gemini.analyze_video_comprehensive(
            keyframes=keyframes,
            video_id=video_id,
            filename=video.filename,
            duration_seconds=duration_seconds
        )
        logger.info("Comprehensive analysis complete.")

        # Print the descriptor summary
        video_descriptor.print_summary()

        # Print JSON representation
        print("\n" + "="*80)
        print("VIDEO DESCRIPTOR - JSON REPRESENTATION")
        print("="*80)
        print(video_descriptor.to_json())
        print("="*80 + "\n")

        # Use overall summary as scene_summary for backward compatibility
        scene_summary = video_descriptor.overall_summary

        # Clean up temp file
        os.unlink(tmp_path)
        logger.info(f"Deleted temp file: {tmp_path}")

        # Mock ML grouping logic
        group_id = f"g{len(state.groups) + 1}"
        print(f"\n{'='*80}")
        print(f"[BACKEND] Creating new group: {group_id}")
        print(f"[BACKEND] Scene summary length: {len(scene_summary)} characters")
        print(f"[BACKEND] Scene summary preview: {scene_summary[:200]}...")
        print(f"{'='*80}\n")

        new_group = {
            "id": group_id,
            "name": f"Project: {video.filename.split('.')[0]}",
            "description": scene_summary[:100] + "...",
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "videoCount": 1,
            "totalDuration": int(duration_seconds),
            "videos": []
        }

        new_video = {
            "id": video_id,
            "filename": video.filename,
            "originalName": video.filename,
            "duration": int(duration_seconds),
            "size": len(content),
            "uploadedAt": datetime.utcnow().isoformat() + "Z",
            "status": "ready",
            "groupId": group_id
        }

        new_group["videos"].append(new_video)
        state.groups.append(new_group)
        state.scenes[group_id] = scene_summary
        state.video_descriptors[video_id] = video_descriptor

        print(f"[BACKEND] Stored scene data in state.scenes['{group_id}']")
        print(f"[BACKEND] Stored video descriptor in state.video_descriptors['{video_id}']")
        print(f"[BACKEND] Total groups in state: {len(state.groups)}")
        print(f"[BACKEND] Total scenes in state: {len(state.scenes)}")
        print(f"[BACKEND] Total descriptors in state: {len(state.video_descriptors)}")
        print(f"[BACKEND] Scene keys: {list(state.scenes.keys())}")

        return {
            "video": new_video,
            "group": {"id": group_id, "name": new_group["name"]},
            "message": "Video processed and 'splat' analysis complete."
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

async def analyze_video(video_path: str, video_id: str) -> List[Dict]:
    """Extract frames from a video and analyze each with Gemini in parallel."""
    SAMPLING_INTERVAL_SECONDS = 10
    MAX_IMAGE_DIMENSION = 1024
    PROMPT = (
        "Analyze this construction site image and identify all construction objects and furniture present. "
        "For each object, provide:\n"
        "1. Its category (equipment, materials, workers, vehicles, or structures)\n"
        "2. A brief description of the object\n"
        "3. Distance estimate from the camera in meters (as a decimal number):\n"
        "   - Use object size relative to the frame as a primary cue\n"
        "   - Consider visible detail level (high detail = closer, low detail = farther)\n"
        "   - Use spatial context and perspective cues\n"
        "   - Typical ranges: 0-10m (large/detailed), 10-20m (moderate), 20-100m+ (small/distant)\n"
        "\n"
        "Provide your best numerical estimate in meters. "
        "Return an empty list if no construction objects are found."
    )

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * SAMPLING_INTERVAL_SECONDS))

    frames_to_analyze = []
    frame_count = 0

    logger.info("Extracting frames | fps=%.2f total_frames=%d", fps, total_frames)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            if max(pil_image.size) > MAX_IMAGE_DIMENSION:
                ratio = MAX_IMAGE_DIMENSION / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            frames_to_analyze.append({
                "image": pil_image,
                "timestamp": frame_count / fps,
                "frame_number": frame_count,
            })

        frame_count += 1

    video.release()
    logger.info("Extracted %d frames for analysis", len(frames_to_analyze))

    def analyze_frame_sync(frame_data: Dict) -> List[Dict]:
        """Analyze a single frame via LangChain structured output (runs in thread pool)."""
        try:
            message = HumanMessage(content=[
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(frame_data["image"])}},
            ])

            result: FrameDetections = structured_llm.invoke([message])

            detections = [
                {
                    "video_id": video_id,
                    "seconds": round(frame_data["timestamp"], 2),
                    "frame_number": frame_data["frame_number"],
                    "object_type": det.object_type,
                    "description": det.description,
                    "distance_estimate": det.distance_estimate,
                    "x": None,
                    "y": None,
                    "z": None,
                    "rx": None,
                    "ry": None,
                    "rz": None,
                }
                for det in result.detections
            ]

            logger.info("✓ Frame %d: %d objects", frame_data["frame_number"], len(detections))
            return detections

        except Exception as e:
            logger.warning("✗ Frame %d error: %s", frame_data["frame_number"], e)
            return []

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [
            loop.run_in_executor(executor, analyze_frame_sync, frame_data)
            for frame_data in frames_to_analyze
        ]
        results = await asyncio.gather(*tasks)

    all_detections = [det for detection_list in results for det in detection_list]
    logger.info("Analysis complete | total_detections=%d", len(all_detections))
    return all_detections


async def _store_detection_embeddings(detections: List[Dict], video_id: str) -> None:
    """Embed each detection description and store in Supabase for semantic search."""
    logger.info("Storing %d detection embeddings | video_id=%s", len(detections), video_id)


    documents = [
        Document(
            page_content=det["description"],
            metadata={
                "video_id": det["video_id"],
                "object_type": det["object_type"],
                "seconds": det["seconds"],
                "frame_number": det["frame_number"],
                "distance_estimate": det["distance_estimate"],
                "x": det["x"],
                "y": det["y"],
                "z": det["z"],
                "rx": det["rx"],
                "ry": det["ry"],
                "rz": det["rz"],
            },
        )
        for det in detections
    ]

    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=EMBEDDINGS_TABLE,
        query_name=EMBEDDINGS_MATCH_FN,
    )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: vector_store.add_documents(documents))
    logger.info("Detection embeddings stored | video_id=%s", video_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_data_url(image: Image.Image) -> str:
    """Convert a PIL image to a base64 data URL for LangChain image messages."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

def get_coordinates_for_frame(frame: int, transforms_data: Dict) -> Dict:
    """Extract position and rotation from a transforms.json dict for a given frame index.

    Parameters
    ----------
    frame : int
        Frame number (e.g. 20 for ``images/frame_00020.jpg``).
    transforms_data : dict
        Parsed contents of the transforms.json produced by the Gaussian Splatting pipeline.

    Returns
    -------
    dict
        ``file_path``, ``position`` (x, y, z) and ``rotation`` (rx, ry, rz in
        degrees) extracted from the frame's 4×4 transform matrix.
    """
    target_path = f"images/frame_{frame:05d}.jpg"
    f = None
    for entry in transforms_data["frames"]:
        if entry["file_path"] == target_path:
            f = entry
            break

    if f is None:
        return {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0}

    m = np.array(f["transform_matrix"])

    # Position (translation column)
    x, y, z = float(m[0, 3]), float(m[1, 3]), float(m[2, 3])

    # Rotation (Euler angles from the 3×3 rotation sub-matrix)
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], sy)
    yaw = math.atan2(r[1, 0], r[0, 0])

    return {
        "file_path": f["file_path"],
        "x": x,
        "y": y,
        "z": z,
        "rx": round(math.degrees(roll), 2),
        "ry": round(math.degrees(pitch), 2),
        "rz": round(math.degrees(yaw), 2),
    }


def _backfill_coordinates_sync(task_id: str, transforms_data: Dict) -> List[Dict]:
    """Fetch all detections for a video from Supabase, update their camera coordinates
    using transforms.json, and return the fully-populated detection dicts so the caller
    can hand them straight to _store_detection_embeddings.
    Called after run_remote_pipeline completes successfully.
    """
    if not supabase:
        logger.warning("Supabase not configured, skipping coordinate backfill | task_id=%s", task_id)
        return []

    response = (
        supabase.table("detections")
        .select("id, video_id, frame_number, object_type, description, seconds, distance_estimate")
        .eq("video_id", task_id)
        .execute()
    )
    detections = response.data or []

    if not detections:
        logger.info("No detections to backfill | task_id=%s", task_id)
        return []

    logger.info("Backfilling coordinates for %d detections | task_id=%s", len(detections), task_id)

    filled: List[Dict] = []
    for det in detections:
        coords = get_coordinates_for_frame(det["frame_number"] + 1, transforms_data)
        supabase.table("detections").update({
            "x": coords["x"],
            "y": coords["y"],
            "z": coords["z"],
            "rx": coords["rx"],
            "ry": coords["ry"],
            "rz": coords["rz"],
        }).eq("id", det["id"]).execute()
        filled.append({
            "video_id": det["video_id"],
            "object_type": det["object_type"],
            "description": det["description"],
            "seconds": det["seconds"],
            "frame_number": det["frame_number"],
            "distance_estimate": det["distance_estimate"],
            "x": coords["x"],
            "y": coords["y"],
            "z": coords["z"],
            "rx": coords["rx"],
            "ry": coords["ry"],
            "rz": coords["rz"],
        })

    logger.info("Coordinate backfill complete | task_id=%s", task_id)
    return filled

class SafetyAnalysisRequest(BaseModel):
    description: str

@app.post("/analyze-safety")
async def analyze_safety(request: SafetyAnalysisRequest):
    """
    Analyzes a construction scene description for safety hazards using RAG.
    """
    try:
        result = safety_rag.evaluate_safety(request.description)
        return {"analysis": result}
    except Exception as e:
        print(f"Safety analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_scene(request: QueryRequest):
    """
    Query scene with optional video descriptor context.

    Accepts:
    - query: Natural language query
    - group_id: Optional group to filter
    - descriptors: Optional array of video descriptors for enhanced context
    """
    print(f"\n{'='*80}")
    print(f"[BACKEND] Query endpoint called - REQUEST RECEIVED")
    print(f"[BACKEND] Query: {request.query}")
    print(f"[BACKEND] Group ID: {request.group_id}")
    print(f"[BACKEND] Descriptors provided: {len(request.descriptors) if request.descriptors else 0}")

    if request.descriptors and len(request.descriptors) > 0:
        print(f"[BACKEND] First descriptor sample:")
        first_desc = request.descriptors[0]
        print(f"  - video_id: {first_desc.get('video_id', 'N/A')}")
        print(f"  - filename: {first_desc.get('filename', 'N/A')}")
        print(f"  - scene_type: {first_desc.get('scene_type', 'N/A')}")
        print(f"  - keys: {list(first_desc.keys())[:10]}")  # First 10 keys

    print(f"[BACKEND] Total scenes available: {len(state.scenes)}")
    print(f"[BACKEND] Scene keys: {list(state.scenes.keys())}")
    print(f"{'='*80}\n")

    if not state.scenes:
        print("[BACKEND] ERROR: No scenes in state!")
        return {"error": "No scenes processed yet. Please upload a video first."}

    # Build context from scenes
    if request.group_id:
        # Use scene for specific group
        context = state.scenes.get(request.group_id, "")
        print(f"[BACKEND] Using context for group: {request.group_id}")
    else:
        # Use all scene context
        context = "\n".join(state.scenes.values())
        print(f"[BACKEND] Using context from all groups")

    # Enhance context with descriptor data if provided
    if request.descriptors and len(request.descriptors) > 0:
        print(f"[BACKEND] Enhancing context with {len(request.descriptors)} video descriptors")
        descriptor_context = "\n\n=== DETAILED VIDEO ANALYSIS ===\n"

        for idx, desc_dict in enumerate(request.descriptors, 1):
            descriptor_context += f"\n--- Video {idx}: {desc_dict.get('filename', 'unknown')} ---\n"
            descriptor_context += f"Scene Type: {desc_dict.get('scene_type', 'unknown')}\n"
            descriptor_context += f"Duration: {desc_dict.get('duration_seconds', 0):.1f}s\n"

            # Add people information
            people = desc_dict.get('people', [])
            if people:
                descriptor_context += f"\nPeople ({len(people)}):\n"
                for person in people:
                    descriptor_context += f"  - {person.get('description', 'unknown person')}\n"
                    descriptor_context += f"    Activities: {', '.join(person.get('activities', []))}\n"

            # Add spatial zones
            zones = desc_dict.get('spatial_zones', [])
            if zones:
                descriptor_context += f"\nSpatial Zones ({len(zones)}):\n"
                for zone in zones:
                    descriptor_context += f"  - {zone.get('zone_name', 'unknown')}: {zone.get('zone_description', '')}\n"

            # Add frame snapshots
            snapshots = desc_dict.get('frame_snapshots', [])
            if snapshots:
                descriptor_context += f"\nKey Moments ({len(snapshots)} snapshots):\n"
                for snapshot in snapshots:
                    timestamp = snapshot.get('timestamp', 0)
                    summary = snapshot.get('scene_summary', '')
                    descriptor_context += f"  - [{timestamp:.1f}s] {summary}\n"

        context = context + descriptor_context
        print(f"[BACKEND] Enhanced context length: {len(context)} characters")

    print(f"[BACKEND] Combined context length: {len(context)} characters")
    print(f"[BACKEND] Context preview: {context[:300]}...")

    try:
        gemini = get_gemini_service()
        print(f"[BACKEND] Sending query to Gemini...")
        response: SceneQueryResponse = gemini.query_scene(request.query, context)

        print(f"\n[BACKEND] Gemini response received:")
        print(f"  - Analysis: {response.analysis[:100]}...")
        print(f"  - Confidence: {response.confidence}")
        print(f"  - Hotspots: {response.hotspots}")
        print(f"  - Location: {response.location}")
        print(f"  - Coordinates: {response.coordinates}")
        print(f"  - Worker: {response.worker}")
        print(f"  - Worker Role: {response.workerRole}")

        # Convert Pydantic model to dict for JSON response
        result = response.model_dump()
        print(f"[BACKEND] Returning response with {len(result.get('hotspots', []))} hotspots")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Query error: {e}")
        print(f"\n[BACKEND] ====== ERROR DURING QUERY ======")
        print(f"[BACKEND] Error type: {type(e).__name__}")
        print(f"[BACKEND] Error message: {str(e)}")
        import traceback
        print(f"[BACKEND] Full traceback:")
        print(traceback.format_exc())
        print(f"[BACKEND] ===================================\n")

        # Return error with proper status code
        return JSONResponse(
            status_code=500,
            content={
                "analysis": f"Error processing query: {str(e)}",
                "confidence": "Low",
                "hotspots": [],
                "location": None,
                "coordinates": None,
                "worker": None,
                "workerRole": None,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
