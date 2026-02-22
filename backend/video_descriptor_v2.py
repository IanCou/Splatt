"""
Video Descriptor System for General Scene Analysis

Generalized metadata structures for analyzing any video footage using Gemini AI.
Optimized for tracking people, spatial understanding, scene characteristics, and objects.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import json


# ============================================================================
# PERSON TRACKING
# ============================================================================
@dataclass
class PersonAppearance:
    """
    Track individuals appearing in the video.

    General-purpose person tracking for any scenario (students, workers,
    performers, etc.). Tracks what they're doing, where they are, and
    any notable details.
    """
    # Identification
    person_id: Optional[str]  # Name, number, or identifier if visible
    role_inferred: Optional[str]  # e.g., "student", "instructor", "worker", "participant"
    description: str  # Physical appearance: "person in red shirt", "tall individual with glasses"

    # Activity tracking
    activities: List[str]  # What they're doing: ["writing notes", "pointing at board", "assembling parts"]
    interactions: List[str]  # Who/what they interact with: ["talking to person in blue", "using laptop"]

    # Location and timing
    location_description: str  # Where in the scene: "left side of frame", "near whiteboard", "center table"
    timestamp_range: tuple[float, float]  # (start_sec, end_sec) when visible

    # Observations
    notable_behaviors: List[str]  # Interesting or unusual behaviors
    objects_used: List[str]  # Objects they interact with
    confidence: str  # High/Medium/Low


# ============================================================================
# SPATIAL UNDERSTANDING
# ============================================================================
@dataclass
class SpatialZone:
    """
    Map different areas/zones in the scene.

    Divides the video frame into logical spatial regions and tracks
    what happens in each zone over time.
    """
    zone_name: str  # e.g., "Front of classroom", "Left workbench", "Background area"
    zone_description: str  # Detailed description of what defines this zone

    # Temporal presence
    visible_timeranges: List[tuple[float, float]]  # When this zone is visible

    # Contents
    people_in_zone: List[str]  # Person IDs present in this zone
    objects_in_zone: List[str]  # Notable objects in this zone
    activities_in_zone: List[str]  # What's happening here

    # Characteristics
    zone_type: str  # "work_area", "seating", "storage", "transition", "focal_point"
    importance: str  # "primary", "secondary", "background"
    accessibility: str  # "open", "crowded", "blocked"

    # Changes over time
    state_changes: List[str]  # Notable changes: "became crowded at 2:00", "cleared out after 3:30"


# ============================================================================
# SCENE CHARACTERISTICS
# ============================================================================
@dataclass
class SceneCharacteristics:
    """
    Comprehensive understanding of the scene's environment and setting.

    Captures the overall feel, atmosphere, and physical characteristics
    of the scene. Goes beyond just weather to understand the complete
    setting.
    """
    # Physical Environment
    setting_type: str  # "indoor", "outdoor", "mixed"
    location_description: str  # "classroom", "workshop", "park", "office", "lab"
    space_characteristics: str  # "spacious", "cramped", "open-plan", "divided into sections"

    # Lighting
    lighting_type: str  # "natural", "artificial", "mixed", "dim", "bright"
    lighting_quality: str  # "even", "dramatic shadows", "backlit", "overhead fluorescent"
    lighting_changes: Optional[str]  # Changes over video: "dimmed at 1:30", "sunset at end"

    # Visibility and Clarity
    visibility: str  # "clear", "partially obscured", "dusty/hazy", "foggy"
    camera_quality: str  # "stable", "shaky", "zooming", "panning"
    view_angle: str  # "wide angle", "close-up", "overhead", "eye-level"

    # Atmosphere
    activity_level: str  # "busy", "calm", "chaotic", "orderly"
    noise_level_inferred: Optional[str]  # "quiet", "moderate", "loud" (inferred from visual cues)
    overall_mood: str  # "focused", "relaxed", "energetic", "tense"

    # Weather (if outdoor/visible)
    weather: Optional[str]  # "sunny", "overcast", "rainy", "snowy" - None if indoor
    temperature_inferred: Optional[str]  # Clues like clothing, condensation

    # Time context
    time_of_day: Optional[str]  # "morning", "afternoon", "evening", "night"
    season_inferred: Optional[str]  # Based on clothing, decorations, foliage

    # Notable features
    distinctive_features: List[str]  # Unique aspects: "large whiteboard with diagrams", "plants in background"
    background_elements: List[str]  # Consistent background items

    # Impact on activity
    environmental_impact: Optional[str]  # How setting affects the activity


# ============================================================================
# COMPREHENSIVE FRAME SNAPSHOTS
# ============================================================================
@dataclass
class FrameSnapshot:
    """
    Comprehensive snapshot of the entire scene at a specific timestamp.

    Captures everything visible at this moment: objects, people, zones,
    and scene characteristics. This creates a complete picture of the
    scene state at key moments throughout the video.
    """
    timestamp: float  # When this snapshot was taken

    # ==== OBJECTS ====
    # Categorized objects
    furniture: List[str]  # Tables, chairs, desks, shelves
    equipment: List[str]  # Computers, projectors, machines, tools
    materials: List[str]  # Papers, books, supplies, raw materials
    personal_items: List[str]  # Bags, water bottles, phones, clothing items
    decorative_items: List[str]  # Posters, plants, artwork
    other_objects: List[str]  # Anything else notable

    # Object states and interactions
    object_states: Dict[str, str]  # {"laptop": "open and in use", "door": "closed"}
    object_locations: Dict[str, str]  # {"textbook": "on center table", "backpack": "hanging on chair"}

    # Activity indicators
    recently_moved: List[str]  # Objects that appear to have moved since last snapshot
    in_use: List[str]  # Objects currently being used
    newly_appeared: List[str]  # Objects that weren't visible before
    disappeared: List[str]  # Objects that were visible before but aren't now

    # Scene organization
    organization_level: str  # "neat", "organized", "cluttered", "chaotic"
    focal_objects: List[str]  # Most important/prominent objects in frame

    # ==== PEOPLE ====
    # People visible at this timestamp
    people_visible: List[str]  # Person IDs or descriptions: ["Student A", "person in blue hoodie"]
    people_descriptions: List[str]  # Quick descriptors: ["typing on laptop", "writing notes", "standing"]
    people_locations: List[str]  # Where each person is: ["center table", "left side", "background"]
    primary_activity_by_person: Dict[str, str]  # {"Student A": "explaining", "Student B": "listening"}

    # ==== SPATIAL ZONES ====
    # Zones visible at this timestamp
    zones_visible: List[str]  # Zone names: ["Center Table", "Whiteboard Area", "Doorway"]
    zone_descriptions: List[str]  # Brief zone states: ["Center Table: active with 3 people", "Whiteboard: empty"]
    primary_zone: Optional[str]  # The main focus zone at this moment

    # ==== SCENE STATE ====
    # Scene characteristics at this moment
    lighting_state: str  # "bright natural light", "dimming", "artificial overhead"
    visibility_state: str  # "clear", "partially obscured", "dusty"
    activity_level: str  # "high activity", "calm", "moderate"
    camera_state: str  # "stable", "panning left", "zooming in"

    # Overall scene summary
    scene_summary: str  # One-sentence summary of what's happening at this moment
    # Example: "Three students collaborating at center table while one writes on whiteboard"


@dataclass
class ObjectTimeline:
    """
    Track how a specific object appears, moves, and is used throughout the video.
    """
    object_name: str
    object_description: str  # Detailed description

    # Temporal tracking
    first_appearance: float  # First time seen
    last_appearance: float  # Last time seen
    total_visible_duration: float  # Total seconds visible

    # Location tracking
    locations: List[tuple[float, str]]  # [(timestamp, location_description), ...]
    movement_pattern: Optional[str]  # "stationary", "moved around", "passed between people"

    # Usage tracking
    interactions: List[tuple[float, str, str]]  # [(timestamp, person, action), ...]
    # Example: [(45.2, "person in blue", "picked up"), (67.8, "person in blue", "wrote with")]

    # State changes
    state_changes: List[tuple[float, str]]  # [(timestamp, state_description), ...]
    # Example: [(30.0, "opened"), (120.5, "closed")]


# ============================================================================
# MAIN VIDEO DESCRIPTOR CLASS
# ============================================================================
@dataclass
class VideoDescriptor:
    """
    Comprehensive metadata for general-purpose video analysis.

    Works for any type of video: students in class, people working,
    events, activities, etc. Focuses on understanding people, space,
    scene, and objects.
    """
    # Core metadata
    video_id: str
    filename: str
    duration_seconds: float
    upload_timestamp: datetime
    analysis_timestamp: datetime

    # High-level summary (always included)
    overall_summary: str
    primary_activities: List[str]
    key_locations: List[str]
    scene_type: str  # "educational", "work", "social", "performance", "experiment", etc.

    # FEATURE 1: Person Tracking
    people: List[PersonAppearance] = field(default_factory=list)
    num_people_estimate: Optional[int] = None  # Total count if many people

    # FEATURE 2: Spatial Understanding
    spatial_zones: List[SpatialZone] = field(default_factory=list)
    spatial_organization: Optional[str] = None  # Overall spatial layout description

    # FEATURE 3: Scene Characteristics
    scene_characteristics: Optional[SceneCharacteristics] = None

    # FEATURE 4: Comprehensive Frame Snapshots (includes objects, people, zones, scene)
    frame_snapshots: List[FrameSnapshot] = field(default_factory=list)
    object_timelines: List[ObjectTimeline] = field(default_factory=list)

    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notable_moments: List[tuple[float, str]] = field(default_factory=list)  # [(timestamp, description)]
    anomalies_detected: List[str] = field(default_factory=list)

    gemini_model_used: str = "gemini-2.5-flash"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (summary version)"""
        return {
            "video_id": self.video_id,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "overall_summary": self.overall_summary,
            "scene_type": self.scene_type,
            "primary_activities": self.primary_activities,
            "key_locations": self.key_locations,
            "num_people": len(self.people),
            "num_spatial_zones": len(self.spatial_zones),
            "num_frame_snapshots": len(self.frame_snapshots),
            "num_object_timelines": len(self.object_timelines),
            "tags": self.tags,
            "notable_moments_count": len(self.notable_moments),
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Convert to comprehensive JSON-serializable dictionary.

        Includes all detailed information about people, zones, scene,
        frame snapshots, and object timelines.
        """
        return {
            # Core metadata
            "video_id": self.video_id,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "gemini_model_used": self.gemini_model_used,

            # High-level summary
            "overall_summary": self.overall_summary,
            "scene_type": self.scene_type,
            "primary_activities": self.primary_activities,
            "key_locations": self.key_locations,

            # People tracking
            "people": [
                {
                    "person_id": p.person_id,
                    "role_inferred": p.role_inferred,
                    "description": p.description,
                    "activities": p.activities,
                    "interactions": p.interactions,
                    "location_description": p.location_description,
                    "timestamp_range": {
                        "start": p.timestamp_range[0],
                        "end": p.timestamp_range[1]
                    },
                    "notable_behaviors": p.notable_behaviors,
                    "objects_used": p.objects_used,
                    "confidence": p.confidence
                }
                for p in self.people
            ],
            "num_people_estimate": self.num_people_estimate,

            # Spatial zones
            "spatial_zones": [
                {
                    "zone_name": z.zone_name,
                    "zone_description": z.zone_description,
                    "zone_type": z.zone_type,
                    "importance": z.importance,
                    "visible_timeranges": [
                        {"start": tr[0], "end": tr[1]}
                        for tr in z.visible_timeranges
                    ],
                    "people_in_zone": z.people_in_zone,
                    "objects_in_zone": z.objects_in_zone,
                    "activities_in_zone": z.activities_in_zone,
                    "accessibility": z.accessibility,
                    "state_changes": z.state_changes
                }
                for z in self.spatial_zones
            ],
            "spatial_organization": self.spatial_organization,

            # Scene characteristics
            "scene_characteristics": {
                "setting_type": self.scene_characteristics.setting_type,
                "location_description": self.scene_characteristics.location_description,
                "space_characteristics": self.scene_characteristics.space_characteristics,
                "lighting": {
                    "type": self.scene_characteristics.lighting_type,
                    "quality": self.scene_characteristics.lighting_quality,
                    "changes": self.scene_characteristics.lighting_changes
                },
                "visibility": self.scene_characteristics.visibility,
                "camera": {
                    "quality": self.scene_characteristics.camera_quality,
                    "view_angle": self.scene_characteristics.view_angle
                },
                "activity_level": self.scene_characteristics.activity_level,
                "noise_level_inferred": self.scene_characteristics.noise_level_inferred,
                "overall_mood": self.scene_characteristics.overall_mood,
                "weather": self.scene_characteristics.weather,
                "temperature_inferred": self.scene_characteristics.temperature_inferred,
                "time_of_day": self.scene_characteristics.time_of_day,
                "season_inferred": self.scene_characteristics.season_inferred,
                "distinctive_features": self.scene_characteristics.distinctive_features,
                "background_elements": self.scene_characteristics.background_elements,
                "environmental_impact": self.scene_characteristics.environmental_impact
            } if self.scene_characteristics else None,

            # Frame snapshots
            "frame_snapshots": [
                {
                    "timestamp": fs.timestamp,
                    "scene_summary": fs.scene_summary,
                    "objects": {
                        "furniture": fs.furniture,
                        "equipment": fs.equipment,
                        "materials": fs.materials,
                        "personal_items": fs.personal_items,
                        "decorative_items": fs.decorative_items,
                        "other_objects": fs.other_objects,
                        "object_states": fs.object_states,
                        "object_locations": fs.object_locations,
                        "recently_moved": fs.recently_moved,
                        "in_use": fs.in_use,
                        "newly_appeared": fs.newly_appeared,
                        "disappeared": fs.disappeared,
                        "organization_level": fs.organization_level,
                        "focal_objects": fs.focal_objects
                    },
                    "people": {
                        "visible": fs.people_visible,
                        "descriptions": fs.people_descriptions,
                        "locations": fs.people_locations,
                        "primary_activity_by_person": fs.primary_activity_by_person
                    },
                    "zones": {
                        "visible": fs.zones_visible,
                        "descriptions": fs.zone_descriptions,
                        "primary_zone": fs.primary_zone
                    },
                    "scene_state": {
                        "lighting": fs.lighting_state,
                        "visibility": fs.visibility_state,
                        "activity_level": fs.activity_level,
                        "camera": fs.camera_state
                    }
                }
                for fs in self.frame_snapshots
            ],

            # Object timelines
            "object_timelines": [
                {
                    "object_name": ot.object_name,
                    "object_description": ot.object_description,
                    "first_appearance": ot.first_appearance,
                    "last_appearance": ot.last_appearance,
                    "total_visible_duration": ot.total_visible_duration,
                    "locations": [
                        {"timestamp": loc[0], "location": loc[1]}
                        for loc in ot.locations
                    ],
                    "movement_pattern": ot.movement_pattern,
                    "interactions": [
                        {
                            "timestamp": inter[0],
                            "person": inter[1],
                            "action": inter[2]
                        }
                        for inter in ot.interactions
                    ],
                    "state_changes": [
                        {"timestamp": sc[0], "state": sc[1]}
                        for sc in ot.state_changes
                    ]
                }
                for ot in self.object_timelines
            ],

            # Additional metadata
            "tags": self.tags,
            "notable_moments": [
                {"timestamp": nm[0], "description": nm[1]}
                for nm in self.notable_moments
            ],
            "anomalies_detected": self.anomalies_detected
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to formatted JSON string.

        Args:
            indent: Number of spaces for indentation (default: 2)

        Returns:
            Pretty-printed JSON string
        """
        return json.dumps(self.to_json_dict(), indent=indent, ensure_ascii=False)

    def print_json(self):
        """Print the video descriptor as formatted JSON"""
        print("\n" + "="*80)
        print("VIDEO DESCRIPTOR JSON")
        print("="*80)
        print(self.to_json())
        print("="*80 + "\n")

    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "="*80)
        print(f"VIDEO DESCRIPTOR: {self.filename}")
        print("="*80)
        print(f"Video ID: {self.video_id}")
        print(f"Duration: {self.duration_seconds:.1f} seconds ({self.duration_seconds/60:.1f} minutes)")
        print(f"Scene Type: {self.scene_type}")
        print(f"Analyzed: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nOVERALL SUMMARY:")
        print(f"  {self.overall_summary}")

        print(f"\nPRIMARY ACTIVITIES:")
        for activity in self.primary_activities:
            print(f"  - {activity}")

        print(f"\nKEY LOCATIONS:")
        for location in self.key_locations:
            print(f"  - {location}")

        # FEATURE 1: People
        if self.people:
            print(f"\n{'='*80}")
            print(f"PEOPLE TRACKED ({len(self.people)}):")
            print("="*80)
            for person in self.people:
                print(f"\n  Person: {person.description}")
                if person.person_id:
                    print(f"    ID: {person.person_id}")
                if person.role_inferred:
                    print(f"    Role: {person.role_inferred}")
                print(f"    Location: {person.location_description}")
                print(f"    Visible: {person.timestamp_range[0]:.1f}s - {person.timestamp_range[1]:.1f}s")
                print(f"    Activities:")
                for activity in person.activities:
                    print(f"      - {activity}")
                if person.interactions:
                    print(f"    Interactions:")
                    for interaction in person.interactions:
                        print(f"      - {interaction}")
                if person.objects_used:
                    print(f"    Objects Used: {', '.join(person.objects_used)}")

        # FEATURE 2: Spatial Zones
        if self.spatial_zones:
            print(f"\n{'='*80}")
            print(f"SPATIAL ZONES ({len(self.spatial_zones)}):")
            print("="*80)
            for zone in self.spatial_zones:
                print(f"\n  Zone: {zone.zone_name}")
                print(f"    Description: {zone.zone_description}")
                print(f"    Type: {zone.zone_type} | Importance: {zone.importance}")
                if zone.people_in_zone:
                    print(f"    People: {', '.join(zone.people_in_zone)}")
                if zone.activities_in_zone:
                    print(f"    Activities:")
                    for activity in zone.activities_in_zone:
                        print(f"      - {activity}")
                if zone.objects_in_zone:
                    print(f"    Key Objects: {', '.join(zone.objects_in_zone[:5])}")

        # FEATURE 3: Scene Characteristics
        if self.scene_characteristics:
            print(f"\n{'='*80}")
            print(f"SCENE CHARACTERISTICS:")
            print("="*80)
            sc = self.scene_characteristics
            print(f"  Setting: {sc.setting_type} - {sc.location_description}")
            print(f"  Space: {sc.space_characteristics}")
            print(f"  Lighting: {sc.lighting_type} ({sc.lighting_quality})")
            print(f"  Visibility: {sc.visibility}")
            print(f"  Camera: {sc.camera_quality} - {sc.view_angle}")
            print(f"  Activity Level: {sc.activity_level}")
            print(f"  Mood: {sc.overall_mood}")
            if sc.time_of_day:
                print(f"  Time of Day: {sc.time_of_day}")
            if sc.weather:
                print(f"  Weather: {sc.weather}")
            if sc.distinctive_features:
                print(f"  Distinctive Features:")
                for feature in sc.distinctive_features:
                    print(f"    - {feature}")

        # FEATURE 4: Comprehensive Frame Snapshots
        if self.frame_snapshots:
            print(f"\n{'='*80}")
            print(f"FRAME SNAPSHOTS ({len(self.frame_snapshots)}):")
            print("="*80)
            for i, snapshot in enumerate(self.frame_snapshots[:3], 1):  # Show first 3
                print(f"\n  Snapshot {i} @ {snapshot.timestamp:.1f}s:")
                print(f"    Summary: {snapshot.scene_summary}")

                # People in this frame
                if snapshot.people_visible:
                    print(f"    People ({len(snapshot.people_visible)}):")
                    for person, location, activity in zip(
                        snapshot.people_visible,
                        snapshot.people_locations,
                        [snapshot.primary_activity_by_person.get(p, "present") for p in snapshot.people_visible]
                    ):
                        print(f"      - {person} at {location}: {activity}")

                # Zones in this frame
                if snapshot.zones_visible:
                    print(f"    Zones: {', '.join(snapshot.zones_visible)}")
                    if snapshot.primary_zone:
                        print(f"      Primary focus: {snapshot.primary_zone}")

                # Scene state
                print(f"    Scene State:")
                print(f"      Lighting: {snapshot.lighting_state}")
                print(f"      Activity: {snapshot.activity_level}")
                print(f"      Organization: {snapshot.organization_level}")

                # Objects
                print(f"    Objects:")
                if snapshot.furniture:
                    print(f"      Furniture: {', '.join(snapshot.furniture)}")
                if snapshot.equipment:
                    print(f"      Equipment: {', '.join(snapshot.equipment)}")
                if snapshot.focal_objects:
                    print(f"      Focal: {', '.join(snapshot.focal_objects)}")
                if snapshot.in_use:
                    print(f"      In Use: {', '.join(snapshot.in_use)}")

        if self.object_timelines:
            print(f"\n  Object Timelines ({len(self.object_timelines)}):")
            for timeline in self.object_timelines[:5]:  # Show first 5
                duration = timeline.total_visible_duration
                print(f"    - {timeline.object_name}: visible for {duration:.1f}s")
                if timeline.movement_pattern:
                    print(f"      Movement: {timeline.movement_pattern}")

        # Notable moments
        if self.notable_moments:
            print(f"\n{'='*80}")
            print(f"NOTABLE MOMENTS ({len(self.notable_moments)}):")
            print("="*80)
            for timestamp, description in self.notable_moments:
                print(f"  [{timestamp:.1f}s] {description}")

        # Tags
        if self.tags:
            print(f"\nTAGS: {', '.join(self.tags)}")

        print("="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
def create_sample_descriptor():
    """Create a sample descriptor for a classroom/study session"""
    descriptor = VideoDescriptor(
        video_id="vid_67890",
        filename="study_group_session.mp4",
        duration_seconds=240.0,
        upload_timestamp=datetime.now(),
        analysis_timestamp=datetime.now(),
        overall_summary="Study group session with three students working on a project at a table. Students are discussing, taking notes, and referencing a laptop. Room is well-lit with natural light from windows. Whiteboard visible in background with diagrams.",
        scene_type="educational",
        primary_activities=["Collaborative studying", "Note-taking", "Discussion", "Using laptop"],
        key_locations=["Center table", "Whiteboard area", "Window side"],

        # People tracking
        people=[
            PersonAppearance(
                person_id="Student A",
                role_inferred="group leader",
                description="person in blue hoodie, wearing glasses",
                activities=["Pointing at laptop screen", "Explaining concepts", "Writing notes"],
                interactions=["Talking to Student B", "Showing laptop to others"],
                location_description="center of table, facing camera",
                timestamp_range=(0.0, 240.0),
                notable_behaviors=["Frequently gesturing while explaining"],
                objects_used=["laptop", "notebook", "pen"],
                confidence="High"
            ),
            PersonAppearance(
                person_id="Student B",
                role_inferred="participant",
                description="person in red shirt with long hair",
                activities=["Listening", "Taking notes", "Asking questions"],
                interactions=["Discussing with Student A", "Looking at shared laptop"],
                location_description="left side of table",
                timestamp_range=(0.0, 180.0),
                notable_behaviors=["Left scene at 3:00 mark"],
                objects_used=["notebook", "pen", "water bottle"],
                confidence="High"
            ),
        ],

        # Spatial zones
        spatial_zones=[
            SpatialZone(
                zone_name="Center Table",
                zone_description="Main work area with three students seated",
                visible_timeranges=[(0.0, 240.0)],
                people_in_zone=["Student A", "Student B", "Student C"],
                objects_in_zone=["laptop", "notebooks", "pens", "textbooks", "water bottles"],
                activities_in_zone=["Studying", "Discussion", "Note-taking"],
                zone_type="work_area",
                importance="primary",
                accessibility="crowded",
                state_changes=["Student B left at 3:00, became less crowded"]
            ),
            SpatialZone(
                zone_name="Whiteboard Area",
                zone_description="Background area with whiteboard containing diagrams",
                visible_timeranges=[(0.0, 240.0)],
                people_in_zone=[],
                objects_in_zone=["whiteboard", "diagrams", "markers on tray"],
                activities_in_zone=[],
                zone_type="background",
                importance="secondary",
                accessibility="open",
                state_changes=[]
            ),
        ],

        # Scene characteristics
        scene_characteristics=SceneCharacteristics(
            setting_type="indoor",
            location_description="study room or classroom",
            space_characteristics="medium-sized room, well-organized",
            lighting_type="natural",
            lighting_quality="bright, even lighting from large windows",
            lighting_changes="slightly dimmed near end as sun moved",
            visibility="clear",
            camera_quality="stable, fixed position",
            view_angle="eye-level, wide angle capturing full table",
            activity_level="moderate",
            noise_level_inferred="moderate (active discussion)",
            overall_mood="focused and collaborative",
            weather=None,
            temperature_inferred="comfortable (light clothing)",
            time_of_day="afternoon",
            season_inferred="spring or fall (moderate clothing)",
            distinctive_features=[
                "Large whiteboard with complex diagrams",
                "Windows with natural light on right side",
                "Wooden table in center"
            ],
            background_elements=["Whiteboard", "Window blinds", "Door in background"],
            environmental_impact="Favorable - good lighting and comfortable space supports studying"
        ),

        # Comprehensive frame snapshots
        frame_snapshots=[
            FrameSnapshot(
                timestamp=30.0,
                # Objects
                furniture=["table", "three chairs"],
                equipment=["laptop"],
                materials=["three notebooks", "textbook", "loose papers"],
                personal_items=["two water bottles", "backpack on floor"],
                decorative_items=["plant on window sill"],
                other_objects=["pens (multiple)", "highlighters"],
                object_states={"laptop": "open, screen visible", "notebooks": "open with writing"},
                object_locations={
                    "laptop": "center of table",
                    "textbook": "right side of table",
                    "backpack": "on floor near chair"
                },
                recently_moved=[],
                in_use=["laptop", "notebooks", "pens"],
                newly_appeared=[],
                disappeared=[],
                organization_level="organized",
                focal_objects=["laptop", "notebooks"],
                # People
                people_visible=["Student A", "Student B", "Student C"],
                people_descriptions=["typing on laptop", "writing in notebook", "reading textbook"],
                people_locations=["center of table", "left side", "right side"],
                primary_activity_by_person={
                    "Student A": "explaining while typing",
                    "Student B": "taking notes",
                    "Student C": "reading"
                },
                # Zones
                zones_visible=["Center Table", "Whiteboard Area", "Window Side"],
                zone_descriptions=[
                    "Center Table: active collaboration with 3 students",
                    "Whiteboard Area: visible but not in use",
                    "Window Side: natural light source"
                ],
                primary_zone="Center Table",
                # Scene state
                lighting_state="bright natural light from windows",
                visibility_state="clear, high visibility",
                activity_level="moderate - focused studying",
                camera_state="stable, fixed position",
                scene_summary="Three students actively collaborating at center table with laptop and notes"
            ),
            FrameSnapshot(
                timestamp=120.0,
                # Objects
                furniture=["table", "three chairs"],
                equipment=["laptop"],
                materials=["three notebooks", "textbook", "loose papers"],
                personal_items=["water bottles"],
                decorative_items=["plant on window sill"],
                other_objects=["pens (multiple)"],
                object_states={"laptop": "open, all students viewing screen", "notebooks": "some closed"},
                object_locations={"laptop": "center of table, turned toward Student B"},
                recently_moved=["laptop"],
                in_use=["laptop"],
                newly_appeared=[],
                disappeared=["backpack"],
                organization_level="organized",
                focal_objects=["laptop"],
                # People
                people_visible=["Student A", "Student B", "Student C"],
                people_descriptions=["pointing at screen", "leaning in to view", "leaning in to view"],
                people_locations=["center", "left leaning in", "right leaning in"],
                primary_activity_by_person={
                    "Student A": "presenting on laptop",
                    "Student B": "viewing laptop",
                    "Student C": "viewing laptop"
                },
                # Zones
                zones_visible=["Center Table", "Whiteboard Area"],
                zone_descriptions=[
                    "Center Table: all students focused on laptop",
                    "Whiteboard Area: background, not active"
                ],
                primary_zone="Center Table",
                # Scene state
                lighting_state="natural light, slightly dimmed",
                visibility_state="clear",
                activity_level="high - intense focus moment",
                camera_state="stable",
                scene_summary="All three students lean in to view laptop screen together - key collaborative moment"
            ),
        ],

        # Object timelines
        object_timelines=[
            ObjectTimeline(
                object_name="Laptop",
                object_description="Silver laptop, appears to be displaying shared document",
                first_appearance=0.0,
                last_appearance=240.0,
                total_visible_duration=240.0,
                locations=[(0.0, "center of table")],
                movement_pattern="stationary",
                interactions=[
                    (15.0, "Student A", "typing on"),
                    (45.0, "Student A", "pointing at screen"),
                    (120.0, "Student B", "looking at screen")
                ],
                state_changes=[(0.0, "open"), (240.0, "still open")]
            ),
        ],

        # Notable moments
        notable_moments=[
            (45.0, "Student A explains concept by pointing at laptop"),
            (90.0, "All students lean in to look at laptop together"),
            (180.0, "Student B leaves the table"),
        ],

        tags=["study_session", "collaboration", "indoor", "afternoon", "small_group"],
        anomalies_detected=[]
    )

    return descriptor


if __name__ == "__main__":
    print("\nVIDEO DESCRIPTOR SYSTEM V2 - GENERAL PURPOSE")
    print("="*80)
    print("\nFocused on 4 core features:")
    print("  1. Person Tracking - Who is in the scene and what are they doing?")
    print("  2. Spatial Understanding - How is the space organized?")
    print("  3. Scene Characteristics - What is the setting, lighting, atmosphere?")
    print("  4. Frame Snapshots - Comprehensive snapshots at key moments capturing:")
    print("     - All visible objects and their states")
    print("     - People present and their activities")
    print("     - Active zones and spatial layout")
    print("     - Scene state (lighting, activity level, etc.)")
    print("\nGeneralized for any video type (students, work, events, activities)")
    print("="*80)

    descriptor = create_sample_descriptor()
    descriptor.print_summary()

    print("\n" + "="*80)
    print("JSON REPRESENTATION")
    print("="*80)
    print("\nGenerating JSON output...")
    descriptor.print_json()

    print("\nReady to use with Gemini AI analysis!")
