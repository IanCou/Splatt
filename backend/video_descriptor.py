"""
Video Descriptor System for Construction Site Analysis

This module defines comprehensive metadata structures for analyzing
construction site helmet cam footage using Gemini AI.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# ANALYSIS OPTION 1: Object & Material Tracking
# ============================================================================
class MaterialType(Enum):
    LUMBER = "lumber"
    CONCRETE = "concrete"
    REBAR = "rebar"
    DRYWALL = "drywall"
    PIPING = "piping"
    ELECTRICAL = "electrical"
    INSULATION = "insulation"
    OTHER = "other"


@dataclass
class MaterialInstance:
    """Track specific materials visible in video"""
    material_type: MaterialType
    quantity_estimate: str  # e.g., "40 2x4 boards", "~20 cubic yards"
    condition: str  # e.g., "good", "damaged", "weathered"
    location_description: str  # e.g., "south wall, near loading zone"
    timestamp_first_seen: float  # seconds into video
    confidence: str  # High/Medium/Low


@dataclass
class EquipmentInstance:
    """Track equipment visible in video"""
    equipment_name: str  # e.g., "Concrete Mixer", "Circular Saw"
    equipment_id: Optional[str]  # If visible on equipment
    status: str  # "in_use", "idle", "maintenance"
    operator: Optional[str]  # Worker using equipment
    location_description: str
    timestamp_first_seen: float
    confidence: str


# ============================================================================
# ANALYSIS OPTION 2: Personnel & Safety Tracking
# ============================================================================
@dataclass
class WorkerAppearance:
    """Track worker appearances and safety compliance"""
    worker_id: Optional[str]  # Name if visible on vest/helmet
    role_inferred: str  # e.g., "Foreman", "Carpenter", "Laborer"
    ppe_status: Dict[str, bool]  # {"hard_hat": True, "safety_vest": True, "gloves": False}
    activities: List[str]  # e.g., ["installing headers", "measuring"]
    location_description: str
    timestamp_range: tuple[float, float]  # (start_sec, end_sec)
    safety_concerns: List[str]  # e.g., ["working at height without harness"]


# ============================================================================
# ANALYSIS OPTION 3: Activity & Event Timeline
# ============================================================================
class EventType(Enum):
    DELIVERY = "delivery"
    INSTALLATION = "installation"
    INSPECTION = "inspection"
    POUR = "pour"
    MEASUREMENT = "measurement"
    DEMOLITION = "demolition"
    CLEANUP = "cleanup"
    BREAK = "break"
    SAFETY_INCIDENT = "safety_incident"
    OTHER = "other"


@dataclass
class VideoEvent:
    """Discrete events that occur in the video"""
    event_type: EventType
    description: str  # Natural language description
    participants: List[str]  # Workers involved
    equipment_used: List[str]
    materials_involved: List[str]
    timestamp_start: float
    duration_seconds: Optional[float]
    outcome: Optional[str]  # e.g., "completed", "in_progress", "issue_found"
    related_area: str  # e.g., "south wall second floor"


# ============================================================================
# ANALYSIS OPTION 4: Spatial Mapping
# ============================================================================
@dataclass
class SpatialZone:
    """Map different zones/areas in the construction site"""
    zone_name: str  # e.g., "South Wall", "Loading Zone", "Tool Storage"
    zone_type: str  # e.g., "work_area", "storage", "access_route"
    visible_in_timeranges: List[tuple[float, float]]  # When this zone is visible
    objects_in_zone: List[str]
    activities_in_zone: List[str]
    accessibility: str  # e.g., "clear", "obstructed", "hazardous"


# ============================================================================
# ANALYSIS OPTION 5: Progress & Quality Assessment
# ============================================================================
@dataclass
class ProgressMarker:
    """Track construction progress indicators"""
    task_name: str  # e.g., "Second floor framing"
    completion_estimate: int  # 0-100 percentage
    quality_observations: List[str]  # e.g., ["joints properly aligned", "minor gap in corner"]
    blockers: List[str]  # Issues preventing progress
    milestone_achieved: Optional[str]  # e.g., "Frame inspection passed"
    timestamp: float


# ============================================================================
# ANALYSIS OPTION 6: Weather & Environmental Conditions
# ============================================================================
@dataclass
class EnvironmentalConditions:
    """Track weather and site conditions"""
    weather: str  # e.g., "sunny", "overcast", "light rain"
    lighting: str  # e.g., "natural daylight", "dusk", "artificial"
    temperature_estimate: Optional[str]  # If inferable, e.g., "hot (workers sweating)"
    visibility: str  # e.g., "clear", "dusty", "foggy"
    work_impact: Optional[str]  # e.g., "favorable", "slowed by rain", "stopped due to darkness"


# ============================================================================
# ANALYSIS OPTION 7: Communication & Coordination
# ============================================================================
@dataclass
class CommunicationEvent:
    """Track visible communication/coordination"""
    participants: List[str]
    communication_type: str  # e.g., "discussion", "instruction", "radio_call", "reviewing_plans"
    topic_inferred: Optional[str]  # e.g., "discussing measurements"
    timestamp: float
    duration_seconds: Optional[float]


# ============================================================================
# ANALYSIS OPTION 8: Inventory & Resource Tracking
# ============================================================================
@dataclass
class InventorySnapshot:
    """Snapshot of materials/tools visible at specific times"""
    timestamp: float
    visible_materials: Dict[str, str]  # {material: quantity_estimate}
    visible_tools: List[str]
    deliveries_observed: List[str]
    usage_activity: List[str]  # e.g., ["lumber being cut", "concrete being mixed"]


# ============================================================================
# MAIN VIDEO DESCRIPTOR CLASS
# ============================================================================
@dataclass
class VideoDescriptor:
    """
    Comprehensive metadata for a construction site video.

    Contains all analysis results from Gemini AI across multiple dimensions.
    Each field is optional to allow incremental/modular analysis.
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

    # OPTION 1: Object & Material Tracking
    materials: List[MaterialInstance] = field(default_factory=list)
    equipment: List[EquipmentInstance] = field(default_factory=list)

    # OPTION 2: Personnel & Safety
    workers: List[WorkerAppearance] = field(default_factory=list)
    safety_score: Optional[int] = None  # 0-100
    safety_violations: List[str] = field(default_factory=list)

    # OPTION 3: Activity & Event Timeline
    events: List[VideoEvent] = field(default_factory=list)

    # OPTION 4: Spatial Mapping
    spatial_zones: List[SpatialZone] = field(default_factory=list)

    # OPTION 5: Progress & Quality
    progress_markers: List[ProgressMarker] = field(default_factory=list)

    # OPTION 6: Environmental Conditions
    environmental_conditions: Optional[EnvironmentalConditions] = None

    # OPTION 7: Communication & Coordination
    communication_events: List[CommunicationEvent] = field(default_factory=list)

    # OPTION 8: Inventory & Resource Tracking
    inventory_snapshots: List[InventorySnapshot] = field(default_factory=list)

    # Additional metadata
    tags: List[str] = field(default_factory=list)  # User-defined or AI-generated tags
    anomalies_detected: List[str] = field(default_factory=list)  # Unusual observations
    gemini_model_used: str = "gemini-2.5-flash"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "video_id": self.video_id,
            "filename": self.filename,
            "duration_seconds": self.duration_seconds,
            "upload_timestamp": self.upload_timestamp.isoformat(),
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "overall_summary": self.overall_summary,
            "primary_activities": self.primary_activities,
            "key_locations": self.key_locations,
            "materials_count": len(self.materials),
            "equipment_count": len(self.equipment),
            "workers_count": len(self.workers),
            "events_count": len(self.events),
            "spatial_zones_count": len(self.spatial_zones),
            "progress_markers_count": len(self.progress_markers),
            "safety_score": self.safety_score,
            "safety_violations_count": len(self.safety_violations),
            "tags": self.tags,
            "anomalies_detected": self.anomalies_detected,
        }

    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "="*80)
        print(f"VIDEO DESCRIPTOR: {self.filename}")
        print("="*80)
        print(f"Video ID: {self.video_id}")
        print(f"Duration: {self.duration_seconds:.1f} seconds")
        print(f"Analyzed: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOVERALL SUMMARY:")
        print(f"  {self.overall_summary}")
        print(f"\nPRIMARY ACTIVITIES:")
        for activity in self.primary_activities:
            print(f"  - {activity}")
        print(f"\nKEY LOCATIONS:")
        for location in self.key_locations:
            print(f"  - {location}")

        if self.materials:
            print(f"\nMATERIALS DETECTED ({len(self.materials)}):")
            for mat in self.materials[:3]:  # Show first 3
                print(f"  - {mat.quantity_estimate} at {mat.location_description} (@{mat.timestamp_first_seen:.1f}s)")

        if self.equipment:
            print(f"\nEQUIPMENT DETECTED ({len(self.equipment)}):")
            for eq in self.equipment[:3]:
                print(f"  - {eq.equipment_name} ({eq.status}) at {eq.location_description}")

        if self.workers:
            print(f"\nWORKERS OBSERVED ({len(self.workers)}):")
            for worker in self.workers[:3]:
                print(f"  - {worker.role_inferred}: {', '.join(worker.activities)}")
                if worker.safety_concerns:
                    print(f"    [!] Safety: {', '.join(worker.safety_concerns)}")

        if self.safety_score is not None:
            print(f"\nSAFETY SCORE: {self.safety_score}/100")
            if self.safety_violations:
                print(f"  Violations: {', '.join(self.safety_violations)}")

        if self.events:
            print(f"\nEVENTS TIMELINE ({len(self.events)}):")
            for event in self.events[:5]:
                print(f"  - [{event.timestamp_start:.1f}s] {event.event_type.value}: {event.description}")

        if self.progress_markers:
            print(f"\nPROGRESS TRACKING ({len(self.progress_markers)}):")
            for marker in self.progress_markers:
                print(f"  - {marker.task_name}: {marker.completion_estimate}%")

        if self.environmental_conditions:
            env = self.environmental_conditions
            print(f"\nENVIRONMENTAL CONDITIONS:")
            print(f"  Weather: {env.weather} | Lighting: {env.lighting}")
            if env.work_impact:
                print(f"  Impact: {env.work_impact}")

        if self.anomalies_detected:
            print(f"\n[!] ANOMALIES DETECTED:")
            for anomaly in self.anomalies_detected:
                print(f"  - {anomaly}")

        if self.tags:
            print(f"\nTAGS: {', '.join(self.tags)}")

        print("="*80 + "\n")


# ============================================================================
# EXAMPLE USAGE & DEMO
# ============================================================================
def create_sample_descriptor():
    """Create a sample descriptor to demonstrate the structure"""
    descriptor = VideoDescriptor(
        video_id="vid_12345",
        filename="site_footage_feb_19_morning.mp4",
        duration_seconds=180.5,
        upload_timestamp=datetime.now(),
        analysis_timestamp=datetime.now(),
        overall_summary="Morning footage showing framing crew installing second-floor headers on south wall. Concrete mixer visible in background. Three workers observed, all wearing proper PPE. Delivery truck arrives with rebar at 2:30 mark.",
        primary_activities=["Framing installation", "Material delivery", "Equipment operation"],
        key_locations=["South wall second floor", "Loading zone", "Central work area"],

        # Sample material tracking
        materials=[
            MaterialInstance(
                material_type=MaterialType.LUMBER,
                quantity_estimate="40 2x4 boards",
                condition="good",
                location_description="south wall staging area",
                timestamp_first_seen=15.2,
                confidence="High"
            ),
            MaterialInstance(
                material_type=MaterialType.REBAR,
                quantity_estimate="200 pieces #5 rebar",
                condition="new",
                location_description="north gate delivery area",
                timestamp_first_seen=150.8,
                confidence="High"
            ),
        ],

        # Sample equipment tracking
        equipment=[
            EquipmentInstance(
                equipment_name="Concrete Mixer",
                equipment_id="CM-08",
                status="in_use",
                operator="Unknown worker",
                location_description="central work area near foundation",
                timestamp_first_seen=8.5,
                confidence="High"
            ),
        ],

        # Sample worker tracking
        workers=[
            WorkerAppearance(
                worker_id="Mike T.",
                role_inferred="Carpenter",
                ppe_status={"hard_hat": True, "safety_vest": True, "gloves": True, "safety_harness": True},
                activities=["Installing headers", "Measuring", "Communicating with crew"],
                location_description="south wall second floor",
                timestamp_range=(45.0, 120.0),
                safety_concerns=[]
            ),
        ],

        # Sample events
        events=[
            VideoEvent(
                event_type=EventType.DELIVERY,
                description="Rebar delivery truck arrives at north gate",
                participants=["David P. (superintendent)", "Delivery driver"],
                equipment_used=["Forklift"],
                materials_involved=["#5 rebar (200 pieces)"],
                timestamp_start=150.0,
                duration_seconds=25.0,
                outcome="completed",
                related_area="north gate"
            ),
        ],

        # Sample safety score
        safety_score=95,
        safety_violations=[],

        # Sample environmental conditions
        environmental_conditions=EnvironmentalConditions(
            weather="sunny",
            lighting="natural daylight",
            temperature_estimate="mild (comfortable working conditions)",
            visibility="clear",
            work_impact="favorable"
        ),

        # Sample tags
        tags=["framing", "delivery", "south_wall", "morning_shift", "high_activity"],

        anomalies_detected=[]
    )

    return descriptor


if __name__ == "__main__":
    print("\n" + "VIDEO DESCRIPTOR SYSTEM - DEMO" + "\n")
    print("This demonstrates the comprehensive metadata structure for construction videos.")
    print("Below are 8 analysis options you can choose from:\n")

    print("OPTION 1: Object & Material Tracking")
    print("   - Track lumber, concrete, rebar, tools, equipment")
    print("   - Quantity estimates, condition assessment, location")
    print("   - Example: '40 2x4 boards near south wall, good condition'\n")

    print("OPTION 2: Personnel & Safety Tracking")
    print("   - Worker identification, role inference")
    print("   - PPE compliance (hard hats, vests, harnesses)")
    print("   - Safety violations and concerns")
    print("   - Example: 'Carpenter Mike - all PPE worn, working at height safely'\n")

    print("OPTION 3: Activity & Event Timeline")
    print("   - Discrete events: deliveries, installations, inspections")
    print("   - Participants, duration, outcome")
    print("   - Example: '[2:30] Delivery: 200 rebar pieces at north gate'\n")

    print("OPTION 4: Spatial Mapping")
    print("   - Map different zones (south wall, loading zone, storage)")
    print("   - Track what's in each zone and when it's visible")
    print("   - Accessibility and hazards")
    print("   - Example: 'South wall: framing work, 3 workers, clear access'\n")

    print("OPTION 5: Progress & Quality Assessment")
    print("   - Task completion percentages")
    print("   - Quality observations and issues")
    print("   - Milestone tracking")
    print("   - Example: 'Second floor framing: 65% complete, minor gap in corner'\n")

    print("OPTION 6: Weather & Environmental Conditions")
    print("   - Weather, lighting, visibility")
    print("   - Impact on work activities")
    print("   - Example: 'Sunny, clear visibility, favorable conditions'\n")

    print("OPTION 7: Communication & Coordination")
    print("   - Track discussions, instructions, plan reviews")
    print("   - Participants and topics")
    print("   - Example: '[1:45] Foreman and crew discussing measurements'\n")

    print("OPTION 8: Inventory & Resource Tracking")
    print("   - Periodic snapshots of visible materials/tools")
    print("   - Track usage and consumption")
    print("   - Example: 'Lumber: 40 boards -> 25 boards (15 used)'\n")

    print("="*80)
    print("\nGenerating sample descriptor...\n")

    descriptor = create_sample_descriptor()
    descriptor.print_summary()

    print("\nWhich options are most compelling for your use case?")
    print("Let me know and I'll implement Gemini analysis for those features!")
