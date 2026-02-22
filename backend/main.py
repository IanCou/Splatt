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
from fastapi.responses import JSONResponse
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


class SearchRequest(BaseModel):
    query: str = Field(description="Natural language search query")
    limit: int = Field(default=10, description="Max results to return")


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

app = FastAPI(title="Gemini VLM Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/search")
async def semantic_search(request: SearchRequest):
    """Search detections using pgvector semantic similarity.

    Calls the Supabase RPC function directly instead of going through
    LangChain's SupabaseVectorStore, which is incompatible with newer
    versions of supabase-py / postgrest-py.
    """
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    try:
        # 1. Embed the query text
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None, lambda: embeddings.embed_query(request.query)
        )

        # 2. Call the match function via Supabase RPC
        rpc_response = (
            supabase.rpc(
                EMBEDDINGS_MATCH_FN,
                {
                    "query_embedding": query_embedding,
                    "match_count": request.limit,
                },
            ).execute()
        )

        rows = rpc_response.data or []

        hits = []
        for row in rows:
            meta = row.get("metadata") or {}
            hits.append({
                "description": row.get("content", ""),
                "score": round(float(row.get("similarity", 0)), 4),
                "videoId": meta.get("video_id"),
                "objectType": meta.get("object_type"),
                "seconds": meta.get("seconds"),
                "frameNumber": meta.get("frame_number"),
                "distanceEstimate": meta.get("distance_estimate"),
                "x": meta.get("x"),
                "y": meta.get("y"),
                "z": meta.get("z"),
            })

        return {"query": request.query, "results": hits}

    except Exception as e:
        logger.exception("Search failed | query=%s error=%s", request.query, e)
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/object-positions")
async def get_object_positions(video_id: Optional[str] = None):
    """Get all detected objects with their calculated 3D positions.

    Parameters
    ----------
    video_id : str, optional
        Filter by specific video ID. If not provided, returns all objects.

    Returns
    -------
    dict
        List of objects with positions and metadata
    """
    if not supabase:
        raise HTTPException(
            status_code=500,
            detail="Supabase is not configured."
        )

    try:
        query = supabase.table("detections").select("*")

        if video_id:
            query = query.eq("video_id", video_id)

        result = query.execute()

        logger.info("Fetched %d object positions | video_id=%s", len(result.data), video_id or "all")

        return JSONResponse(content={
            "objects": result.data,
            "count": len(result.data),
        })

    except Exception as e:
        logger.exception("Failed to fetch object positions | error=%s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

# FPS at which the splatting pipeline extracts frames (--fps flag in worker_service.py)
SPLAT_FPS = 5
# FPS at which Gemini analyzes frames (can be lower than SPLAT_FPS to save API calls)
GEMINI_FPS = 1

async def analyze_video(video_path: str, video_id: str) -> List[Dict]:
    """Extract frames from a video and analyze each with Gemini in parallel.

    Frames are sampled at GEMINI_FPS (1 fps) but the ``frame_number`` stored
    is the splatting-pipeline-equivalent sequential index (as if extracted at
    SPLAT_FPS).  This way, during backfill ``frame_number + 1`` still maps
    correctly to ``frame_{N:05d}.jpg`` in transforms.json.
    """
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
    source_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Interval (in source frames) between Gemini-analyzed frames.
    # e.g. 30fps source / 1 GEMINI_FPS = every 30th source frame.
    gemini_interval = max(1, round(source_fps / GEMINI_FPS))

    # Interval (in source frames) between splatting pipeline frames.
    # e.g. 30fps source / 5 SPLAT_FPS = every 6th source frame.
    splat_interval = max(1, round(source_fps / SPLAT_FPS))

    frames_to_analyze = []
    source_frame_idx = 0

    logger.info(
        "Extracting frames | source_fps=%.2f total_source_frames=%d "
        "gemini_fps=%d (every %d source frames) splat_fps=%d",
        source_fps, total_frames, GEMINI_FPS, gemini_interval, SPLAT_FPS,
    )

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if source_frame_idx % gemini_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            if max(pil_image.size) > MAX_IMAGE_DIMENSION:
                ratio = MAX_IMAGE_DIMENSION / max(pil_image.size)
                new_size = (int(pil_image.size[0] * ratio), int(pil_image.size[1] * ratio))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

            # Compute the equivalent splatting frame index for this source frame.
            # This is which sequential frame the splatting pipeline would assign.
            splat_frame_idx = source_frame_idx // splat_interval

            frames_to_analyze.append({
                "image": pil_image,
                "timestamp": round(source_frame_idx / source_fps, 2),
                "frame_number": splat_frame_idx,
            })

        source_frame_idx += 1

    video.release()
    logger.info("Extracted %d frames for Gemini analysis (at %d fps)", len(frames_to_analyze), GEMINI_FPS)

    def analyze_frame_sync(frame_data: Dict) -> List[Dict]:
        """Analyze a single frame via LangChain structured output (runs in thread pool)."""
        try:
            message = HumanMessage(content=[
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(frame_data["image"])}},
            ])

            result: FrameDetections = structured_llm.invoke([message])

            if result is None or not hasattr(result, "detections"):
                logger.warning("⚠ Frame %d: Gemini returned empty/invalid response, skipping", frame_data["frame_number"])
                return []

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
                    "obj_x": None,
                    "obj_y": None,
                    "obj_z": None,
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
                "obj_x": det.get("obj_x"),
                "obj_y": det.get("obj_y"),
                "obj_z": det.get("obj_z"),
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


def calculate_object_position(camera_x: float, camera_y: float, camera_z: float,
                             camera_rx: float, camera_ry: float, camera_rz: float,
                             distance: float) -> Tuple[float, float, float]:
    """Calculate object position from camera pose and distance estimate.

    Parameters
    ----------
    camera_x, camera_y, camera_z : float
        Camera position in world coordinates (meters)
    camera_rx, camera_ry, camera_rz : float
        Camera rotation in degrees (roll, pitch, yaw)
    distance : float
        Distance from camera to object (meters)

    Returns
    -------
    tuple
        (obj_x, obj_y, obj_z) - Object position in world coordinates
    """
    if distance is None or distance == 0:
        # If no distance estimate, return camera position
        return (camera_x, camera_y, camera_z)

    # Convert degrees to radians
    roll = math.radians(camera_rx)
    pitch = math.radians(camera_ry)
    yaw = math.radians(camera_rz)

    # Camera forward direction in local space is typically [0, 0, -1]
    # We need to rotate this by the camera's rotation to get world-space direction
    # Using yaw (horizontal rotation) and pitch (vertical tilt)
    # Direction vector: [sin(yaw)*cos(pitch), sin(pitch), -cos(yaw)*cos(pitch)]
    dir_x = math.sin(yaw) * math.cos(pitch)
    dir_y = -math.sin(pitch)  # Negative because Y might be up in your coordinate system
    dir_z = -math.cos(yaw) * math.cos(pitch)

    # Calculate object position: camera position + direction * distance
    obj_x = camera_x + dir_x * distance
    obj_y = camera_y + dir_y * distance
    obj_z = camera_z + dir_z * distance

    return (obj_x, obj_y, obj_z)


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

        # Calculate object position from camera pose + distance
        obj_x, obj_y, obj_z = calculate_object_position(
            coords["x"], coords["y"], coords["z"],
            coords["rx"], coords["ry"], coords["rz"],
            det.get("distance_estimate", 0)
        )

        supabase.table("detections").update({
            "x": coords["x"],
            "y": coords["y"],
            "z": coords["z"],
            "rx": coords["rx"],
            "ry": coords["ry"],
            "rz": coords["rz"],
            "obj_x": round(obj_x, 3),
            "obj_y": round(obj_y, 3),
            "obj_z": round(obj_z, 3),
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
            "obj_x": round(obj_x, 3),
            "obj_y": round(obj_y, 3),
            "obj_z": round(obj_z, 3),
        })

    logger.info("Coordinate backfill complete | task_id=%s", task_id)
    return filled


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)