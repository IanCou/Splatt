from google import genai
import os
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..models.schemas import QueryRequest, QueryResponse
from ..services.vector_store import vector_store

router = APIRouter()

@router.post("/search", response_model=QueryResponse)
async def semantic_search(request: QueryRequest):
    """
    Takes a natural language query, searches the vector store for matches,
    and uses Gemini to formulate a structured AI response.
    """
    # 1. Search ChromaDB
    chroma_results = vector_store.search_hotspots(request.query, n_results=3)
    
    # Construct context from vector search
    context_lines = []
    hotspot_ids = []
    if chroma_results and chroma_results.get("documents"):
        for i, docs in enumerate(chroma_results["documents"]):
            for j, doc in enumerate(docs):
                meta = chroma_results["metadatas"][i][j]
                h_id = chroma_results["ids"][i][j]
                hotspot_ids.append(h_id)
                x, y, z = meta.get("x", 0.0), meta.get("y", 0.0), meta.get("z", 0.0)
                context_lines.append(f"- {doc} at coords (X:{x}, Y:{y}, Z:{z}) [ID: {h_id}]")
    
    context = "\n".join(context_lines) if context_lines else "No relevant hotspots found in database."

    # 2. Use Gemini to format the natural response
    prompt = f"""
    The user is asking: "{request.query}"
    
    Here is the data from our 3D spatial database (ChromaDB):
    {context}
    
    Formulate a structured response in JSON. Do not invent coordinates. If no data is found, admit it.
    Format your response EXACTLY as follows:
    {{
        "analysis": "Natural language answer combining user query and context.",
        "location": "General description e.g. 'Near the crane'",
        "coordinates": "If available, e.g. '0.5,0.4,0.0'. Otherwise null",
        "confidence": "High/Medium/Low",
        "worker": "Name if applicable, else null",
        "workerRole": "Role if applicable, else null"
    }}
    """
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-3.0-flash',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            )
            ai_res = json.loads(response.text)
            
            return QueryResponse(
                analysis=ai_res.get("analysis", "I couldn't process this request."),
                location=ai_res.get("location"),
                coordinates=ai_res.get("coordinates"),
                confidence=ai_res.get("confidence", "Low"),
                hotspots=hotspot_ids
            )
        except Exception as e:
            pass # Fall back to basic response below

    # Mock response if Gemini fails or no API key is provided
    return QueryResponse(
        analysis=f"Fallback search logic processing: {request.query}. Top matches: {context}",
        confidence="Medium",
        hotspots=hotspot_ids
    )
