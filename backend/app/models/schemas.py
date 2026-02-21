from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

# --- Pydantic Schemas ---

class Hotspot(BaseModel):
    id: str
    label: str
    x: float
    y: float
    type: str

class VideoUploadResponse(BaseModel):
    id: str
    filename: str
    groupId: str
    url: str

class ProjectGroup(BaseModel):
    id: str
    name: str
    description: str
    createdAt: datetime
    videoCount: int
    totalDuration: int
    videos: List[Dict]
    thumbnail: str
    hotspots: List[Hotspot]
    keyframes: List[str]

class QueryRequest(BaseModel):
    query: str
    group_id: Optional[str] = None

class QueryResponse(BaseModel):
    analysis: str
    location: Optional[str] = None
    coordinates: Optional[str] = None
    confidence: str
    hotspots: List[str] = []
    worker: Optional[str] = None
    workerRole: Optional[str] = None
    relatedGroupIds: List[str] = []
