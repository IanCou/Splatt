import os
import json
import logging
from google import genai
from typing import List, Dict

logger = logging.getLogger(__name__)

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
client = None
if api_key:
    client = genai.Client(api_key=api_key)
    # Using gemini-3-flash-preview as requested
    model_name = 'gemini-3.0-flash'
else:
    logger.warning("GOOGLE_API_KEY not found. Gemini Analysis will be mocked.")

def analyze_keyframes(pil_images: list) -> Dict:
    """
    Sends keyframes to Gemini to extract JSON list of objects, timestamps, and spatial descriptions.
    """
    if not client:
        # Mock response if API key missing
        return {
            "summary": "Mock analysis: Frame shows scaffolding and workers.",
            "hotspots": [
                {"id": "h1", "label": "Scaffolding", "x": 0.5, "y": 0.4, "z": 0.0, "type": "equipment"},
                {"id": "h2", "label": "Excavator", "x": 0.8, "y": 0.2, "z": 0.0, "type": "equipment"}
            ]
        }

    analysis_prompt = """
    This is a sequence of frames from a construction site 'helmet cam'. 
    
    Task:
    1. Analyze the scene and provide a concise summary (under 100 characters) for a 3D scene index.
    2. Identify specific 'hotspots' (objects of interest) visible across these frames.
    3. For each hotspot, provide:
        - A unique ID (e.g. h1, h2)
        - A descriptive label (e.g. "Concrete Mixer", "Lumber Stack")
        - Normalized 3D mock coordinates (x, y, z) from 0.0 to 1.0 indicating where it is in the general scene. (We will sync this via transforms later)
        - Type: "material", "equipment", "worker", or "event".
    
    Respond ONLY in JSON format:
    {
        "summary": "Concise summary here",
        "hotspots": [
        {"id": "h1", "label": "Label", "x": 0.5, "y": 0.5, "z": 0.5, "type": "material"},
        ...
        ]
    }
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[analysis_prompt] + pil_images, 
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        data = json.loads(response.text)
        return data
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {
            "summary": f"Error during analysis: {e}",
            "hotspots": []
        }
