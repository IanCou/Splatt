"""
Pydantic models for video descriptor structured output with Gemini.

These models define the schema for Gemini's JSON responses when analyzing videos.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple


# ============================================================================
# PERSON TRACKING MODELS
# ============================================================================
class PersonAppearanceModel(BaseModel):
    """Pydantic model for person tracking"""
    person_id: Optional[str] = Field(None, description="Name or identifier if visible")
    role_inferred: Optional[str] = Field(None, description="Role: student, instructor, worker, etc.")
    description: str = Field(..., description="Physical appearance description")
    activities: List[str] = Field(default_factory=list, description="Activities performed")
    interactions: List[str] = Field(default_factory=list, description="Interactions with others/objects")
    location_description: str = Field(..., description="Location in scene")
    timestamp_start: float = Field(..., description="Start time in seconds")
    timestamp_end: float = Field(..., description="End time in seconds")
    notable_behaviors: List[str] = Field(default_factory=list, description="Notable behaviors")
    objects_used: List[str] = Field(default_factory=list, description="Objects interacted with")
    confidence: str = Field("Medium", description="Confidence level: High/Medium/Low")


# ============================================================================
# SPATIAL ZONE MODELS
# ============================================================================
class SpatialZoneModel(BaseModel):
    """Pydantic model for spatial zones"""
    zone_name: str = Field(..., description="Name of the zone")
    zone_description: str = Field(..., description="Detailed description")
    zone_type: str = Field(..., description="Type: work_area, seating, storage, etc.")
    importance: str = Field(..., description="Importance: primary, secondary, background")
    people_in_zone: List[str] = Field(default_factory=list, description="People present")
    objects_in_zone: List[str] = Field(default_factory=list, description="Notable objects")
    activities_in_zone: List[str] = Field(default_factory=list, description="Activities occurring")
    accessibility: str = Field("open", description="Accessibility: open, crowded, blocked")
    state_changes: List[str] = Field(default_factory=list, description="Notable changes over time")


# ============================================================================
# SCENE CHARACTERISTICS MODELS
# ============================================================================
class SceneCharacteristicsModel(BaseModel):
    """Pydantic model for scene characteristics"""
    setting_type: str = Field(..., description="indoor, outdoor, or mixed")
    location_description: str = Field(..., description="Type of location")
    space_characteristics: str = Field(..., description="Space description")
    lighting_type: str = Field(..., description="natural, artificial, mixed, dim, bright")
    lighting_quality: str = Field(..., description="Quality of lighting")
    lighting_changes: Optional[str] = Field(None, description="Lighting changes over time")
    visibility: str = Field(..., description="clear, partially obscured, etc.")
    camera_quality: str = Field(..., description="stable, shaky, zooming, panning")
    view_angle: str = Field(..., description="wide angle, close-up, overhead, eye-level")
    activity_level: str = Field(..., description="busy, calm, chaotic, orderly")
    noise_level_inferred: Optional[str] = Field(None, description="quiet, moderate, loud")
    overall_mood: str = Field(..., description="focused, relaxed, energetic, tense")
    weather: Optional[str] = Field(None, description="Weather if outdoor/visible")
    temperature_inferred: Optional[str] = Field(None, description="Temperature clues")
    time_of_day: Optional[str] = Field(None, description="morning, afternoon, evening, night")
    season_inferred: Optional[str] = Field(None, description="Season based on clues")
    distinctive_features: List[str] = Field(default_factory=list, description="Unique features")
    background_elements: List[str] = Field(default_factory=list, description="Consistent background items")
    environmental_impact: Optional[str] = Field(None, description="How environment affects activity")


# ============================================================================
# FRAME SNAPSHOT MODELS
# ============================================================================
class FrameSnapshotModel(BaseModel):
    """Pydantic model for comprehensive frame snapshots"""
    timestamp: float = Field(..., description="Timestamp in seconds")

    # Objects
    furniture: List[str] = Field(default_factory=list)
    equipment: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    personal_items: List[str] = Field(default_factory=list)
    decorative_items: List[str] = Field(default_factory=list)
    other_objects: List[str] = Field(default_factory=list)
    object_states: Dict[str, str] = Field(default_factory=dict)
    object_locations: Dict[str, str] = Field(default_factory=dict)
    recently_moved: List[str] = Field(default_factory=list)
    in_use: List[str] = Field(default_factory=list)
    newly_appeared: List[str] = Field(default_factory=list)
    disappeared: List[str] = Field(default_factory=list)
    organization_level: str = Field(..., description="neat, organized, cluttered, chaotic")
    focal_objects: List[str] = Field(default_factory=list)

    # People
    people_visible: List[str] = Field(default_factory=list, description="Person IDs visible")
    people_descriptions: List[str] = Field(default_factory=list, description="Activity descriptors")
    people_locations: List[str] = Field(default_factory=list, description="Person locations")
    primary_activity_by_person: Dict[str, str] = Field(default_factory=dict)

    # Zones
    zones_visible: List[str] = Field(default_factory=list, description="Visible zone names")
    zone_descriptions: List[str] = Field(default_factory=list, description="Zone states")
    primary_zone: Optional[str] = Field(None, description="Main focus zone")

    # Scene state
    lighting_state: str = Field(..., description="Current lighting")
    visibility_state: str = Field(..., description="Current visibility")
    activity_level: str = Field(..., description="Current activity level")
    camera_state: str = Field(..., description="Camera state")
    scene_summary: str = Field(..., description="One-sentence frame summary")


# ============================================================================
# COMPLETE VIDEO ANALYSIS MODEL
# ============================================================================
class VideoAnalysisModel(BaseModel):
    """
    Complete video analysis response from Gemini.

    This is the top-level structured output that Gemini will return.
    """
    # High-level summary
    overall_summary: str = Field(..., description="Comprehensive video summary")
    scene_type: str = Field(..., description="educational, work, social, etc.")
    primary_activities: List[str] = Field(..., description="Main activities in video")
    key_locations: List[str] = Field(..., description="Key locations/areas")

    # Feature 1: People
    people: List[PersonAppearanceModel] = Field(default_factory=list)
    num_people_estimate: Optional[int] = Field(None, description="Total people count")

    # Feature 2: Spatial zones
    spatial_zones: List[SpatialZoneModel] = Field(default_factory=list)
    spatial_organization: Optional[str] = Field(None, description="Overall spatial layout")

    # Feature 3: Scene characteristics
    scene_characteristics: SceneCharacteristicsModel

    # Feature 4: Frame snapshots (3-5 key moments)
    frame_snapshots: List[FrameSnapshotModel] = Field(default_factory=list)

    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Descriptive tags")
    notable_moments: List[str] = Field(default_factory=list, description="Key moments with timestamps")
    anomalies_detected: List[str] = Field(default_factory=list, description="Unusual observations")
