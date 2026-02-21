import io
import os
import struct
from typing import Optional

import torch
import numpy as np
from google import genai
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY") 

client = genai.Client(api_key=api_key) if api_key else None

app = FastAPI(title="Gemini VLM Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_image(file: UploadFile = File(...), prompt: Optional[str] = "Describe this image in detail."):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Initialize Gemini client and Generate content
        if not client:
            raise HTTPException(status_code=500, detail="Gemini API Key not configured.")
            
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=[prompt, image]
        )

        return JSONResponse(content={
            "filename": file.filename,
            "analysis": response.text
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def convert_pt_to_splat(model_path: str) -> io.BytesIO:
    """
    Converts a Gaussian4DModel .pt file to the .splat format expected by Three.js renderers.
    The .splat format is a sequence of 32-byte structs:
    pos[3] (float32), scale[3] (float32), color[4] (uint8), rot[4] (uint8)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    
    # Extract parameters.
    xyz = state_dict['xyz'].detach().cpu().numpy().astype(np.float32)           # (N, 3)
    scales = state_dict['scales'].detach().cpu().numpy().astype(np.float32)     # (N, 3)
    colors = state_dict['colors'].detach().cpu().numpy()                       # (N, 3)
    opacities = state_dict['opacities'].detach().cpu().numpy().reshape(-1, 1) # (N, 1)

    num_gaussians = xyz.shape[0]
    
    # Process colors and opacities into RGBA uint8
    rgba_u8 = np.concatenate([
        (np.clip(colors, 0, 1) * 255).astype(np.uint8),
        (np.clip(opacities, 0, 1) * 255).astype(np.uint8)
    ], axis=1) # (N, 4)

    # Standard identity rotation (0,0,0,1) mapped to uint8 (128,128,128,255)
    rot_u8 = np.tile(np.array([128, 128, 128, 255], dtype=np.uint8), (num_gaussians, 1)) # (N, 4)

    # Combine all into a single structured array for fast export
    # Each row: 3x4 (pos) + 3x4 (scale) + 4x1 (color) + 4x1 (rot) = 12 + 12 + 4 + 4 = 32 bytes
    buffer = io.BytesIO()
    for i in range(num_gaussians):
        buffer.write(xyz[i].tobytes())
        buffer.write(scales[i].tobytes())
        buffer.write(rgba_u8[i].tobytes())
        buffer.write(rot_u8[i].tobytes())

    buffer.seek(0)
    return buffer

@app.get("/load-splat/{full_path:path}")
async def load_splat(full_path: str):
    """
    Loads a .pt or .splat file and returns it as a .splat binary stream.
    """
    try:
        # Clean up the path: strip common garbage characters
        clean_path = full_path.strip().strip("'").strip('"').rstrip(":")
        # Normalize slashes for the OS
        clean_path = os.path.normpath(clean_path)
        
        # Determine the target .pt or .splat file
        if clean_path.lower().endswith(".splat"):
            if os.path.exists(clean_path):
                pt_path = None
                splat_path = clean_path
            else:
                pt_path = clean_path.rsplit(".splat", 1)[0] + ".pt"
                splat_path = None
        else:
            if not clean_path.lower().endswith(".pt"):
                pt_path = clean_path + ".pt"
            else:
                pt_path = clean_path
            splat_path = None

        print(f"Request: {full_path} -> Resolved PT: {pt_path}, SPLAT: {splat_path}")
        
        if splat_path and os.path.exists(splat_path):
            return FileResponse(
                splat_path, 
                media_type="application/octet-stream", 
                filename=os.path.basename(splat_path)
            )
        
        if pt_path and os.path.exists(pt_path):
            print(f"Converting PT to Splat: {pt_path}")
            splat_buffer = convert_pt_to_splat(pt_path)
            # For dynamic content, use Response with explicit content and size
            content = splat_buffer.getvalue()
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename={os.path.basename(pt_path).replace('.pt', '.splat')}",
                    "Content-Length": str(len(content)),
                    "Access-Control-Expose-Headers": "Content-Disposition, Content-Length"
                }
            )
            
        # File not found
        error_file = pt_path if pt_path else splat_path
        print(f"ERROR: File not found: {error_file}")
        raise HTTPException(status_code=404, detail=f"File not found: {error_file}")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Exception in load_splat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # When running manually, ensure to use the venv's uvicorn or just run this script
    uvicorn.run(app, host="0.0.0.0", port=8000)
