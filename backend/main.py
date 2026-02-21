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

def convert_ply_to_splat(ply_path: str) -> io.BytesIO:
    """
    Converts a binary GSplat PLY file to the .splat format.
    Handles the specific layout: x,y,z, nx,ny,nz, r,g,b, opacity, scale_0,1,2, rot_0,1,2,3
    """
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    with open(ply_path, 'rb') as f:
        header = ""
        while "end_header" not in header:
            header += f.readline().decode('ascii', errors='ignore')
        
        # After end_header there's usually a newline
        data_start = f.tell()
        
    # Standard Nerf4Dgsplat layout: 
    # floats: x,y,z (12), nx,ny,nz (12)
    # uchars: r,g,b (3)
    # floats: opacity (4), scale[3] (12), rot[4] (16)
    # Total row size: 59 bytes
    
    # Read vertex count from header
    num_vertices = 0
    for line in header.split('\n'):
        if line.startswith('element vertex'):
            num_vertices = int(line.split()[-1])
            break
            
    if num_vertices == 0:
        raise ValueError("Could not find vertex count in PLY header")

    # Read binary data
    # We use numpy to read the structured data efficiently
    dt = np.dtype([
        ('pos', 'f4', 3),
        ('normals', 'f4', 3),
        ('color', 'u1', 3),
        ('opacity', 'f4'),
        ('scale', 'f4', 3),
        ('rot', 'f4', 4)
    ])
    
    data = np.fromfile(ply_path, dtype=dt, count=num_vertices, offset=data_start)
    
    # Pack into .splat format (32 bytes per vertex)
    # pos[3] (f4), scale[3] (f4), rgba[4] (u1), rot[4] (u1)
    
    # 1. Normalize rotation to unit quaternions and map to 0-255
    # Standard: (q + 1) / 2 * 255
    rot = data['rot']
    rot_norm = np.linalg.norm(rot, axis=1, keepdims=True)
    rot_unit = rot / (rot_norm + 1e-8)
    rot_u8 = ((rot_unit + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    
    # 2. Process color and opacity
    # Opacity in PLY is often sigmoid'd if it's from some trainers, 
    # but here we'll assume it's 0-1 unless we see values outside.
    opacity = np.clip(data['opacity'], 0, 1) * 255
    rgba_u8 = np.concatenate([
        data['color'], 
        opacity.reshape(-1, 1).astype(np.uint8)
    ], axis=1)
    
    buffer = io.BytesIO()
    # Batch write everything for speed
    for i in range(num_vertices):
        buffer.write(data['pos'][i].tobytes())
        buffer.write(data['scale'][i].tobytes())
        buffer.write(rgba_u8[i].tobytes())
        buffer.write(rot_u8[i].tobytes())
        
    buffer.seek(0)
    return buffer

@app.api_route("/load-splat/{full_path:path}", methods=["GET", "HEAD"])
async def load_splat(full_path: str):
    """
    Loads a .pt, .ply or .splat file and returns it as a .splat binary stream.
    """
    try:
        # Clean up the path
        clean_path = full_path.strip().strip("'").strip('"').rstrip(":")
        clean_path = os.path.normpath(clean_path)
        
        # Resolve file type
        target_path = clean_path
        
        # If not found, try common variations
        if not os.path.exists(target_path):
            # Case 1: Path was foo.ply but frontend sent foo.splat
            if target_path.lower().endswith(".splat"):
                base = target_path.rsplit(".", 1)[0]
                for ext in [".ply", ".pt", ""]:
                    if os.path.exists(base + ext):
                        target_path = base + ext
                        break
            
            # Case 2: Path was relative and needs to be checked in current dir
            if not os.path.exists(target_path):
                # Try just the filename in current dir if it's a relative-looking path
                basename = os.path.basename(clean_path)
                for ext in ["", ".ply", ".pt", ".splat"]:
                    test_path = basename if not basename.lower().endswith(ext.lower()) else basename
                    if ext and not basename.lower().endswith(ext.lower()):
                         test_path += ext
                    if os.path.exists(test_path):
                        target_path = test_path
                        break

        if not os.path.exists(target_path):
            print(f"ERROR: File not found: {target_path} (Original: {full_path})")
            raise HTTPException(status_code=404, detail=f"File not found: {target_path}")

        ext = os.path.splitext(target_path)[1].lower()
        print(f"Request: {full_path} -> Resolved: {target_path} (Ext: {ext})")
        
        # Handle based on extension
        if ext == ".splat":
            return FileResponse(
                target_path, 
                media_type="application/octet-stream", 
                filename=os.path.basename(target_path)
            )
        
        elif ext == ".pt":
            print(f"Converting PT to Splat: {target_path}")
            splat_buffer = convert_pt_to_splat(target_path)
        
        elif ext == ".ply":
            print(f"Converting PLY to Splat: {target_path}")
            splat_buffer = convert_ply_to_splat(target_path)
            
        else:
            # Fallback for bare files that might be PLY or PT internally
            # For now just treat as PLY if it exists and has no ext
            print(f"Attempting to treat unknown format as PLY: {target_path}")
            splat_buffer = convert_ply_to_splat(target_path)

        content = splat_buffer.getvalue()
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(target_path).rsplit('.', 1)[0]}.splat",
                "Content-Length": str(len(content)),
                "Access-Control-Expose-Headers": "Content-Disposition, Content-Length"
            }
        )

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
