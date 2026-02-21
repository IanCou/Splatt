import base64
import io
import logging
import os
import tempfile
from typing import Optional, List, Dict
from datetime import datetime
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
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


class FrameDetections(BaseModel):
    detections: List[DetectionItem] = Field(
        default_factory=list,
        description="All construction objects detected in the image",
    )


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


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

async def analyze_video(video_path: str, video_id: str) -> List[Dict]:
    """Extract frames from a video and analyze each with Gemini in parallel."""
    SAMPLING_INTERVAL_SECONDS = 10
    MAX_IMAGE_DIMENSION = 1024
    PROMPT = (
        "Analyze this construction site image and identify all construction objects present. "
        "For each object, provide its category and a brief description. "
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
