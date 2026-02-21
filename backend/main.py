import logging
import io
import os
import cv2
import tempfile
import uuid
import base64
import json
from typing import List, Optional, Dict
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3-flash-preview')

app = FastAPI(title="Splatt VLM Backend")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directories exist
DATA_DIR = "data"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
THUMBS_DIR = os.path.join(DATA_DIR, "thumbnails")
KEYFRAMES_DIR = os.path.join(DATA_DIR, "keyframes")
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)
os.makedirs(KEYFRAMES_DIR, exist_ok=True)

# Mount static files
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# In-memory storage for the hackathon
class State:
    groups: List[Dict] = []
    videos: List[Dict] = []
    # Store summarized scene data per group
    scenes: Dict[str, str] = {} 
    # Store hotspots per group
    hotspots: Dict[str, List[Dict]] = {}
    # Store keyframes per group
    keyframes: Dict[str, List[str]] = {}

state = State()

def generate_thumbnail(video_path: str, thumb_path: str):
    logger.info(f"Generating thumbnail for {video_path}")
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(thumb_path, frame)
        logger.info(f"Successfully saved thumbnail to {thumb_path}")
    else:
        logger.error(f"Failed to read frame from {video_path} for thumbnail.")
    cap.release()

def extract_keyframes(video_path: str, video_id: str, interval_sec: int = 2) -> List[Dict]:
    keyframes_data = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    interval_frames = int(fps * interval_sec)
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval_frames == 0:
            # Save frame
            frame_filename = f"{video_id}_k{extracted_count}.jpg"
            frame_path = os.path.join(KEYFRAMES_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            # Convert BGR to RGB for PIL for AI analysis
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((800, 800))
            
            keyframes_data.append({
                "path": f"/data/keyframes/{frame_filename}",
                "image": img
            })
            extracted_count += 1
        frame_count += 1
        
        # Limit to 10 frames per video for the demo
        if extracted_count >= 10:
            break
            
    cap.release()
    return keyframes_data

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
        video_id = str(uuid.uuid4())
        video_ext = video.filename.split('.')[-1]
        video_filename = f"{video_id}.{video_ext}"
        video_path = os.path.join(VIDEOS_DIR, video_filename)
        
        # Save video
        content = await video.read()
        with open(video_path, "wb") as f:
            f.write(content)
        
        # Generate thumbnail
        thumb_filename = f"{video_id}.jpg"
        thumb_path = os.path.join(THUMBS_DIR, thumb_filename)
        generate_thumbnail(video_path, thumb_path)
        
        logger.info(f"Saved video to {video_path} and thumbnail to {thumb_path}")

        logger.info(f"Extracting keyframes from: {video_path}")
        keyframes_data = extract_keyframes(video_path, video_id)
        pil_images = [k["image"] for k in keyframes_data]
        keyframe_urls = [f"http://localhost:8000{k['path']}" for k in keyframes_data]
        logger.info(f"Extracted {len(pil_images)} keyframes.")

        # Get real video duration
        cap_dur = cv2.VideoCapture(video_path)
        _fps = cap_dur.get(cv2.CAP_PROP_FPS) or 30
        _frames = cap_dur.get(cv2.CAP_PROP_FRAME_COUNT)
        real_duration = int(_frames / _fps) if _frames > 0 else 0
        cap_dur.release()
        logger.info(f"Video duration: {real_duration}s")
        
        # Analyze scene context with Gemini
        analysis_prompt = """
        This is a sequence of frames from a construction site 'helmet cam'. 
        
        Task:
        1. Analyze the scene and provide a concise summary (under 100 characters) for a 3D scene index.
        2. Identify specific 'hotspots' (objects of interest) visible across these frames.
        3. For each hotspot, provide:
           - A unique ID (e.g. h1, h2)
           - A descriptive label (e.g. "Concrete Mixer", "Lumber Stack")
           - Normalized coordinates (x, y) from 0-100 indicating where it is in the general scene represented by these frames.
           - Type: "material", "equipment", "worker", or "event".
        
        Respond ONLY in JSON format:
        {
          "summary": "Concise summary here",
          "hotspots": [
            {"id": "h1", "label": "Label", "x": 50, "y": 50, "type": "material"},
            ...
          ]
        }
        """
        
        logger.info("Sending keyframes to Gemini for dynamic scene analysis...")
        response = model.generate_content([analysis_prompt] + pil_images, generation_config={"response_mime_type": "application/json"})
        
        ai_data = json.loads(response.text)
        scene_summary = ai_data.get("summary", "Analysis complete.")
        hotspots = ai_data.get("hotspots", [])
        
        logger.info(f"Analysis complete. Found {len(hotspots)} hotspots.")

        # ML grouping logic
        group_id = f"g{len(state.groups) + 1}"
        
        base_url = "http://localhost:8000"
        thumb_url = f"{base_url}/data/thumbnails/{thumb_filename}"
        video_url = f"{base_url}/data/videos/{video_filename}"

        new_video = {
            "id": video_id,
            "filename": video.filename,
            "originalName": video.filename,
            "duration": real_duration,
            "size": len(content),
            "uploadedAt": datetime.utcnow().isoformat() + "Z",
            "status": "ready",
            "groupId": group_id,
            "thumbnail": thumb_url,
            "url": video_url
        }

        new_group = {
            "id": group_id,
            "name": f"Project: {video.filename.split('.')[0]}",
            "description": scene_summary,
            "createdAt": datetime.utcnow().isoformat() + "Z",
            "videoCount": 1,
            "totalDuration": 0, # Mock
            "videos": [new_video],
            "thumbnail": thumb_url,
            "hotspots": hotspots,
            "keyframes": keyframe_urls
        }
        
        state.groups.append(new_group)
        state.scenes[group_id] = scene_summary
        state.hotspots[group_id] = hotspots
        state.keyframes[group_id] = keyframe_urls

        return {
            "video": new_video,
            "group": new_group,
            "message": "Video processed and dynamic scene analysis complete."
        }

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_scene(query: str = Form(...), group_id: Optional[str] = Form(None)):
    if not state.scenes:
        return {"error": "No scenes processed yet. Please upload a video first."}
    
    # Use all scene context if no group specified
    context_lines = []
    for gid, summary in state.scenes.items():
        hs_text = ", ".join([h["label"] for h in state.hotspots.get(gid, [])])
        context_lines.append(f"Group {gid}: {summary}. Visible items: {hs_text}")
    
    context = "\n".join(context_lines)
    
    prompt = f"""
    Based on the following construction site scene analysis:
    {context}
    
    User Query: {query}
    
    Respond in JSON format with:
    1. 'analysis': A detailed natural language answer.
    2. 'location': Specific site location if mentioned (e.g. "South wall").
    3. 'coordinates': Mock coordinates if possible (e.g. "X:10, Y:20").
    4. 'confidence': 'High', 'Medium', or 'Low'.
    5. 'hotspots': A list of matching hotspot IDs from the context if any.
    6. 'worker': The name of a worker if seen or inferred.
    7. 'workerRole': Their role.
    8. 'relatedGroupIds': List of group IDs relevant to this query.
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return JSONResponse(content=json.loads(response.text))
    except Exception as e:
        logger.error(f"Query Error: {e}")
        return JSONResponse(content={
            "analysis": f"Gemini Analysis: {response.text}",
            "confidence": "Medium",
            "hotspots": []
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
