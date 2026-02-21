import io
import os
import time
import shutil
import tempfile
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Fallback to OPENAI_API_KEY if they haven't updated yet, but warning is better
    api_key = os.getenv("OPENAI_API_KEY") 

genai.configure(api_key=api_key)

app = FastAPI(title="Gemini VLM Backend")

@app.post("/process")
async def process_image(file: UploadFile = File(...), prompt: Optional[str] = "Describe this image in detail."):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content
        response = model.generate_content([prompt, image])

        return JSONResponse(content={
            "filename": file.filename,
            "analysis": response.text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_video")
async def process_video(file: UploadFile = File(...), prompt: Optional[str] = "Find specific textual details in this video."):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    # Create a temporary file to store the upload
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Upload to Gemini File API
        print(f"Uploading {file.filename} to Gemini...")
        video_file = genai.upload_file(path=tmp_path, display_name=file.filename)
        print(f"File uploaded: {video_file.uri}")

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            print("Waiting for video to be processed...")
            time.sleep(5)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Video processing failed in Gemini.")

        # Initialize Gemini model (using 1.5-flash for speed/efficiency)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content with reasoning
        print(f"Generating content for prompt: {prompt}")
        response = model.generate_content([prompt, video_file], request_options={"timeout": 600})

        # Cleanup: Delete the file from Gemini
        genai.delete_file(video_file.name)

        return JSONResponse(content={
            "filename": file.filename,
            "analysis": response.text
        })

    except Exception as e:
        print(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup: Delete local temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    # When running manually, ensure to use the venv's uvicorn or just run this script
    uvicorn.run(app, host="0.0.0.0", port=8000)
