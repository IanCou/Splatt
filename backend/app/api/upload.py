import os
import uuid
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from datetime import datetime
import cv2
from PIL import Image

from ..services.gemini_service import analyze_keyframes
from ..services.vector_store import vector_store
from ..services.compute_bridge import compute_bridge
from ..models.schemas import VideoUploadResponse, ProjectGroup, Hotspot

router = APIRouter()

# Data directories
DATA_DIR = "data"
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
THUMBS_DIR = os.path.join(DATA_DIR, "thumbnails")
KEYFRAMES_DIR = os.path.join(DATA_DIR, "keyframes")

for d in [VIDEOS_DIR, THUMBS_DIR, KEYFRAMES_DIR]:
    os.makedirs(d, exist_ok=True)

def generate_thumbnail(video_path: str, thumb_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(thumb_path, frame)
    cap.release()

def extract_keyframes(video_path: str, video_id: str, interval_sec: int = 2):
    keyframes_data = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    interval_frames = int(fps * interval_sec)
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval_frames == 0:
            frame_filename = f"{video_id}_k{extracted_count}.jpg"
            frame_path = os.path.join(KEYFRAMES_DIR, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((800, 800))
            
            keyframes_data.append({
                "path": f"/data/keyframes/{frame_filename}",
                "image": img
            })
            extracted_count += 1
            if extracted_count >= 10:  # limit for demo
                break
        frame_count += 1
    cap.release()
    return keyframes_data

@router.post("/upload", response_model=dict)
async def upload_video(video: UploadFile = File(...)):
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    video_id = str(uuid.uuid4())
    video_ext = video.filename.split('.')[-1]
    video_filename = f"{video_id}.{video_ext}"
    video_path = os.path.join(VIDEOS_DIR, video_filename)
    
    # 1. Save locally
    content = await video.read()
    with open(video_path, "wb") as f:
        f.write(content)
        
    # 2. Keyframes & Thumbnails
    thumb_path = os.path.join(THUMBS_DIR, f"{video_id}.jpg")
    generate_thumbnail(video_path, thumb_path)
    keyframes_data = extract_keyframes(video_path, video_id)
    pil_images = [k["image"] for k in keyframes_data]
    
    # 3. Trigger Gemini Analysis
    ai_data = analyze_keyframes(pil_images)
    
    # 4. Trigger Vast.ai Training Pipeline
    # Using background tasks would be better in production, doing synchronously/mocked for demo
    compute_bridge.train_splat(video_path, f"{video_id}_output")
    
    # 5. Semantic Sync (Save to ChromaDB)
    hotspots = ai_data.get("hotspots", [])
    for hs in hotspots:
        vec_meta = {"type": hs.get("type", "object"), "x": hs.get("x", 0.0), "y": hs.get("y", 0.0), "z": hs.get("z", 0.0)}
        vector_store.add_hotspot(hs.get("id", str(uuid.uuid4())), hs.get("label", "Unknown"), vec_meta)
        
    return {
        "status": "success",
        "video_id": video_id,
        "summary": ai_data.get("summary", "Analysis complete."),
        "hotspots_found": len(hotspots)
    }
