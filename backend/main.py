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


# In-memory storage for the hackathon
class State:
    groups: List[Dict] = []
    videos: List[Dict] = []
    # Store summarized scene data per group
    scenes: Dict[str, str] = {}
    # Store full video descriptors per video ID
    video_descriptors: Dict[str, Any] = {}

state = State()

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
    print(f"[BACKEND] Query endpoint called")
    print(f"[BACKEND] Query: {request.query}")
    print(f"[BACKEND] Group ID: {request.group_id}")
    print(f"[BACKEND] Descriptors provided: {len(request.descriptors) if request.descriptors else 0}")
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
        print(f"[BACKEND] ERROR during query: {e}")
        import traceback
        print(f"[BACKEND] Traceback:\n{traceback.format_exc()}")
        # Fallback with all required fields
        return JSONResponse(content={
            "analysis": f"Error processing query: {str(e)}",
            "confidence": "Low",
            "hotspots": [],
            "location": None,
            "coordinates": None,
            "worker": None,
            "workerRole": None
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
