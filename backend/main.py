import logging
import os
import cv2
import tempfile
import uuid
from typing import List, Optional, Dict
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

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

# In-memory storage for the hackathon
class State:
    groups: List[Dict] = []
    videos: List[Dict] = []
    # Store summarized scene data per group
    scenes: Dict[str, str] = {} 

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
    return {"groups": state.groups}

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

        # Analyze scene with Gemini via LangChain
        logger.info("Sending keyframes to Gemini for analysis...")
        gemini = get_gemini_service()
        scene_summary = gemini.analyze_construction_scene(keyframes)
        logger.info("Analysis complete.")

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
            "totalDuration": 0, # Mock
            "videos": []
        }

        video_id = str(uuid.uuid4())
        new_video = {
            "id": video_id,
            "filename": video.filename,
            "originalName": video.filename,
            "duration": 0, # Mock
            "size": len(content),
            "uploadedAt": datetime.utcnow().isoformat() + "Z",
            "status": "ready",
            "groupId": group_id
        }

        new_group["videos"].append(new_video)
        state.groups.append(new_group)
        state.scenes[group_id] = scene_summary

        print(f"[BACKEND] Stored scene data in state.scenes['{group_id}']")
        print(f"[BACKEND] Total groups in state: {len(state.groups)}")
        print(f"[BACKEND] Total scenes in state: {len(state.scenes)}")
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
async def query_scene(query: str = Form(...), group_id: Optional[str] = Form(None)):
    print(f"\n{'='*80}")
    print(f"[BACKEND] Query endpoint called")
    print(f"[BACKEND] Query: {query}")
    print(f"[BACKEND] Group ID: {group_id}")
    print(f"[BACKEND] Total scenes available: {len(state.scenes)}")
    print(f"[BACKEND] Scene keys: {list(state.scenes.keys())}")
    print(f"{'='*80}\n")

    if not state.scenes:
        print("[BACKEND] ERROR: No scenes in state!")
        return {"error": "No scenes processed yet. Please upload a video first."}

    # Use all scene context if no group specified
    context = "\n".join(state.scenes.values())
    print(f"[BACKEND] Combined context length: {len(context)} characters")
    print(f"[BACKEND] Context preview: {context[:300]}...")

    try:
        gemini = get_gemini_service()
        print(f"[BACKEND] Sending query to Gemini...")
        response: SceneQueryResponse = gemini.query_scene(query, context)

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
