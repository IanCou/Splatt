from pydantic import BaseModel, Field
from typing import List, Optional


class SceneQueryResponse(BaseModel):
    """
    Structured response for construction site scene queries.

    This schema enforces type safety and validation for AI-generated
    responses to natural language queries about construction scenes.
    """
    analysis: str = Field(
        description="Detailed natural language answer to the user's query"
    )
    location: Optional[str] = Field(
        default=None,
        description="Specific site location if mentioned (e.g. 'South wall', 'Second floor')"
    )
    coordinates: Optional[str] = Field(
        default=None,
        description="Mock 3D coordinates if determinable (e.g. 'X:10, Y:20, Z:5')"
    )
    confidence: str = Field(
        description="Confidence level of the analysis: High, Medium, or Low"
    )
    hotspots: List[str] = Field(
        default_factory=list,
        description="List of matching hotspot IDs (e.g., ['h1', 'h3', 'h5'])"
    )
    worker: Optional[str] = Field(
        default=None,
        description="Name of worker if visible or inferable from context"
    )
    workerRole: Optional[str] = Field(
        default=None,
        description="Role/trade of the worker (e.g., 'Electrician', 'Carpenter')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis": "The lumber pile is located near the south wall, approximately 15 feet from the main entrance.",
                "location": "South wall",
                "coordinates": "X:15, Y:3, Z:0",
                "confidence": "High",
                "hotspots": ["h2", "h5"],
                "worker": "John Smith",
                "workerRole": "Carpenter"
            }
        }
