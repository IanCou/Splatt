import io
import os
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

if __name__ == "__main__":
    import uvicorn
    # When running manually, ensure to use the venv's uvicorn or just run this script
    uvicorn.run(app, host="0.0.0.0", port=8000)
