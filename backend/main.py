import logging
import io
import os
import cv2
import tempfile
import uuid
import base64
from typing import List, Optional, Dict
from datetime import datetime

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
        # Extract context
        keyframes = extract_keyframes(tmp_path)
        logger.info(f"Extracted {len(keyframes)} keyframes.")
        
        # Analyze scene context with Gemini
        analysis_prompt = """
        This is a sequence of frames from a construction site 'helmet cam'. 
        Analyze the scene and describe:
        1. Major objects/materials (e.g. lumber, concrete)
        2. Equipment present
        3. Activities being performed
        4. Spatial relationships
        
        Format as a concise summary for a 3D scene index.
        """
        
        logger.info("Sending keyframes to Gemini for analysis...")
        response = model.generate_content([analysis_prompt] + keyframes)
        scene_summary = response.text
        logger.info("Analysis complete.")

        # Clean up temp file
        os.unlink(tmp_path)
        logger.info(f"Deleted temp file: {tmp_path}")

        # Mock ML grouping logic
        group_id = f"g{len(state.groups) + 1}"
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
    if not state.scenes:
        return {"error": "No scenes processed yet. Please upload a video first."}
    
    # Use all scene context if no group specified
    context = "\n".join(state.scenes.values())
    
    prompt = f"""
    Based on the following construction site scene analysis:
    {context}
    
    User Query: {query}
    
    Respond in JSON format with:
    1. 'analysis': A detailed natural language answer.
    2. 'location': Specific site location if mentioned (e.g. "South wall").
    3. 'coordinates': Mock coordinates if possible (e.g. "X:10, Y:20").
    4. 'confidence': 'High', 'Medium', or 'Low'.
    5. 'hotspots': A list of matching hotspot IDs (h1, h2, h3, h4, h5, h6) if applicable.
    6. 'worker': The name of a worker if seen or inferred.
    7. 'workerRole': Their role.
    """
    
    try:
        # Use generation_config to encourage JSON if supported, or just parse
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        import json
        return JSONResponse(content=json.loads(response.text))
    except Exception as e:
        print(f"Query Error: {e}")
        # Fallback if Gemini fails to produce valid JSON
        return JSONResponse(content={
            "analysis": f"Gemini Analysis: {response.text}",
            "confidence": "Medium",
            "hotspots": []
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
