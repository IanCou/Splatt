import os
import logging
import base64
from io import BytesIO
from typing import List
from PIL import Image

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from models import SceneQueryResponse
from video_descriptor_models import VideoAnalysisModel
from video_descriptor_v2 import (
    VideoDescriptor, PersonAppearance, SpatialZone,
    SceneCharacteristics, FrameSnapshot, ObjectTimeline
)
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def image_to_base64_url(img: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URL.

    Args:
        img: PIL Image object

    Returns:
        Base64-encoded data URL string (e.g., "data:image/png;base64,...")
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return f"data:image/png;base64,{img_base64}"


class GeminiService:
    """
    Service layer for all Gemini AI operations in Splatt.

    Encapsulates LangChain interactions and provides clean abstractions
    for multimodal video analysis and structured scene querying.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini service with LangChain.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        # Initialize LangChain ChatGoogleGenerativeAI
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.7,
        )

        # Create structured output version for queries
        self.structured_llm = self.llm.with_structured_output(
            SceneQueryResponse,
            method="json_mode"
        )

        logger.info(f"GeminiService initialized with model: {model_name}")

    def analyze_construction_scene(self, keyframes: List[Image.Image]) -> str:
        """
        Analyze construction site video keyframes to generate scene summary.

        This is a multimodal operation that processes both text prompts
        and image data to understand the construction environment.

        Args:
            keyframes: List of PIL Image objects (typically 2-10 frames)

        Returns:
            Scene summary string describing objects, equipment, activities,
            and spatial relationships
        """
        if not keyframes:
            raise ValueError("At least one keyframe is required for analysis")

        logger.info(f"Analyzing {len(keyframes)} keyframes...")

        # Construct analysis prompt
        analysis_prompt = """
        This is a sequence of frames from a construction site 'helmet cam'.
        Analyze the scene and describe:
        1. Major objects/materials (e.g. lumber, concrete)
        2. Equipment present
        3. Activities being performed
        4. Spatial relationships

        Format as a concise summary for a 3D scene index.
        """

        # Build message content with text + images
        message_content = [{"type": "text", "text": analysis_prompt}]

        for img in keyframes:
            # Convert PIL Image to base64 data URL
            img_data_url = image_to_base64_url(img)
            message_content.append({
                "type": "image_url",
                "image_url": img_data_url
            })

        # Create HumanMessage with multimodal content
        message = HumanMessage(content=message_content)

        # Invoke LLM
        print(f"[GEMINI] Sending {len(keyframes)} keyframes to Gemini for analysis...")
        response = self.llm.invoke([message])

        print(f"[GEMINI] Analysis complete. Response length: {len(response.content)} characters")
        print(f"[GEMINI] Response preview: {response.content[:300]}...")
        logger.info("Scene analysis complete")
        return response.content

    def query_scene(self, user_query: str, scene_context: str) -> SceneQueryResponse:
        """
        Query construction site scenes with structured output.

        Uses Pydantic schema to ensure consistent, validated JSON responses
        containing analysis, location data, confidence, and worker information.

        Args:
            user_query: Natural language question from user
            scene_context: Combined scene summaries to query against

        Returns:
            SceneQueryResponse with validated structured data
        """
        logger.info(f"Processing query: {user_query[:50]}...")
        print(f"\n[GEMINI] Query scene called")
        print(f"[GEMINI] User query: {user_query}")
        print(f"[GEMINI] Scene context length: {len(scene_context)} characters")

        prompt = f"""
        Based on the following construction site scene analysis:
        {scene_context}

        User Query: {user_query}

        Respond with:
        1. 'analysis': A detailed natural language answer.
        2. 'location': Specific site location if mentioned (e.g. "South wall").
        3. 'coordinates': Mock coordinates if possible (e.g. "X:10, Y:20").
        4. 'confidence': 'High', 'Medium', or 'Low'.
        5. 'hotspots': A list of matching hotspot IDs (h1, h2, h3, h4, h5, h6) if applicable.
        6. 'worker': The name of a worker if seen or inferred.
        7. 'workerRole': Their role.
        """

        print(f"[GEMINI] Sending structured query to Gemini...")
        # Use structured output - automatically validates against Pydantic model
        response: SceneQueryResponse = self.structured_llm.invoke(prompt)

        print(f"[GEMINI] Structured response received:")
        print(f"  - Analysis: {response.analysis[:100] if response.analysis else 'None'}...")
        print(f"  - Confidence: {response.confidence}")
        print(f"  - Hotspots: {response.hotspots}")
        print(f"  - Location: {response.location}")
        print(f"  - Coordinates: {response.coordinates}")
        print(f"  - Worker: {response.worker}")
        print(f"  - Worker Role: {response.workerRole}")

        logger.info(f"Query complete with confidence: {response.confidence}")
        return response

    def analyze_video_comprehensive(
        self,
        keyframes: List[Image.Image],
        video_id: str,
        filename: str,
        duration_seconds: float
    ) -> VideoDescriptor:
        """
        Comprehensive video analysis generating full VideoDescriptor.

        Analyzes video keyframes to extract:
        - People and their activities
        - Spatial zones and organization
        - Scene characteristics (lighting, atmosphere, setting)
        - Frame snapshots at key moments
        - Object timelines

        Args:
            keyframes: List of PIL Image objects from the video
            video_id: Unique video identifier
            filename: Original video filename
            duration_seconds: Video duration

        Returns:
            VideoDescriptor with comprehensive analysis
        """
        if not keyframes:
            raise ValueError("At least one keyframe is required for analysis")

        logger.info(f"Starting comprehensive analysis of {filename} with {len(keyframes)} keyframes...")
        print(f"\n[GEMINI] Comprehensive video analysis starting...")
        print(f"[GEMINI] Video: {filename}")
        print(f"[GEMINI] Keyframes: {len(keyframes)}")
        print(f"[GEMINI] Duration: {duration_seconds:.1f}s")

        # Construct detailed analysis prompt
        analysis_prompt = f"""
        Analyze this video footage comprehensively. The video is {duration_seconds:.1f} seconds long.
        You are viewing {len(keyframes)} keyframes sampled evenly throughout the video.

        Provide a detailed analysis covering:

        1. PEOPLE TRACKING:
           - Identify all people visible in the video
           - Describe their appearance, roles, and activities
           - Note what objects they use and who they interact with
           - Track their locations and when they appear

        2. SPATIAL UNDERSTANDING:
           - Identify distinct zones/areas in the scene
           - Describe what happens in each zone
           - Note the spatial organization and layout

        3. SCENE CHARACTERISTICS:
           - Describe the setting (indoor/outdoor, type of location)
           - Analyze lighting, visibility, camera angle
           - Assess the overall activity level and mood
           - Note distinctive features of the environment

        4. FRAME SNAPSHOTS:
           - Create 3-5 snapshots at key moments throughout the video
           - For each snapshot, document:
             * All visible objects (furniture, equipment, materials, personal items)
             * People present and their activities
             * Active zones
             * Scene state (lighting, activity level, organization)
             * A one-sentence summary of that moment
           - Include timestamps in seconds

        5. ADDITIONAL INSIGHTS:
           - Generate descriptive tags
           - Identify notable moments
           - Detect any anomalies or unusual observations

        Be specific and detailed. Use timestamps when describing events.
        Focus on observable facts rather than speculation.
        """

        # Build message content with text + images
        message_content = [{"type": "text", "text": analysis_prompt}]

        for idx, img in enumerate(keyframes):
            img_data_url = image_to_base64_url(img)
            message_content.append({
                "type": "image_url",
                "image_url": img_data_url
            })
            print(f"[GEMINI] Added keyframe {idx + 1}/{len(keyframes)}")

        # Create structured LLM for VideoAnalysisModel
        structured_llm = self.llm.with_structured_output(
            VideoAnalysisModel,
            method="json_mode"
        )

        # Create HumanMessage
        message = HumanMessage(content=message_content)

        print(f"[GEMINI] Sending analysis request to Gemini...")
        # Invoke LLM with structured output
        analysis: VideoAnalysisModel = structured_llm.invoke([message])

        print(f"[GEMINI] Analysis complete!")
        print(f"[GEMINI] Found {len(analysis.people)} people")
        print(f"[GEMINI] Found {len(analysis.spatial_zones)} spatial zones")
        print(f"[GEMINI] Generated {len(analysis.frame_snapshots)} frame snapshots")

        # Convert Pydantic model to VideoDescriptor dataclass
        descriptor = VideoDescriptor(
            video_id=video_id,
            filename=filename,
            duration_seconds=duration_seconds,
            upload_timestamp=datetime.now(),
            analysis_timestamp=datetime.now(),
            overall_summary=analysis.overall_summary,
            scene_type=analysis.scene_type,
            primary_activities=analysis.primary_activities,
            key_locations=analysis.key_locations,
            num_people_estimate=analysis.num_people_estimate,
            spatial_organization=analysis.spatial_organization,
            tags=analysis.tags,
            notable_moments=[(0.0, moment) for moment in analysis.notable_moments],  # Parse timestamps if needed
            anomalies_detected=analysis.anomalies_detected,
            gemini_model_used=self.llm.model_name if hasattr(self.llm, 'model_name') else "gemini-2.5-flash"
        )

        # Convert people
        for person in analysis.people:
            descriptor.people.append(PersonAppearance(
                person_id=person.person_id,
                role_inferred=person.role_inferred,
                description=person.description,
                activities=person.activities,
                interactions=person.interactions,
                location_description=person.location_description,
                timestamp_range=(person.timestamp_start, person.timestamp_end),
                notable_behaviors=person.notable_behaviors,
                objects_used=person.objects_used,
                confidence=person.confidence
            ))

        # Convert spatial zones
        for zone in analysis.spatial_zones:
            descriptor.spatial_zones.append(SpatialZone(
                zone_name=zone.zone_name,
                zone_description=zone.zone_description,
                visible_timeranges=[(0.0, duration_seconds)],  # Simplified for now
                people_in_zone=zone.people_in_zone,
                objects_in_zone=zone.objects_in_zone,
                activities_in_zone=zone.activities_in_zone,
                zone_type=zone.zone_type,
                importance=zone.importance,
                accessibility=zone.accessibility,
                state_changes=zone.state_changes
            ))

        # Convert scene characteristics
        sc = analysis.scene_characteristics
        descriptor.scene_characteristics = SceneCharacteristics(
            setting_type=sc.setting_type,
            location_description=sc.location_description,
            space_characteristics=sc.space_characteristics,
            lighting_type=sc.lighting_type,
            lighting_quality=sc.lighting_quality,
            lighting_changes=sc.lighting_changes,
            visibility=sc.visibility,
            camera_quality=sc.camera_quality,
            view_angle=sc.view_angle,
            activity_level=sc.activity_level,
            noise_level_inferred=sc.noise_level_inferred,
            overall_mood=sc.overall_mood,
            weather=sc.weather,
            temperature_inferred=sc.temperature_inferred,
            time_of_day=sc.time_of_day,
            season_inferred=sc.season_inferred,
            distinctive_features=sc.distinctive_features,
            background_elements=sc.background_elements,
            environmental_impact=sc.environmental_impact
        )

        # Convert frame snapshots
        for snapshot in analysis.frame_snapshots:
            descriptor.frame_snapshots.append(FrameSnapshot(
                timestamp=snapshot.timestamp,
                furniture=snapshot.furniture,
                equipment=snapshot.equipment,
                materials=snapshot.materials,
                personal_items=snapshot.personal_items,
                decorative_items=snapshot.decorative_items,
                other_objects=snapshot.other_objects,
                object_states=snapshot.object_states,
                object_locations=snapshot.object_locations,
                recently_moved=snapshot.recently_moved,
                in_use=snapshot.in_use,
                newly_appeared=snapshot.newly_appeared,
                disappeared=snapshot.disappeared,
                organization_level=snapshot.organization_level,
                focal_objects=snapshot.focal_objects,
                people_visible=snapshot.people_visible,
                people_descriptions=snapshot.people_descriptions,
                people_locations=snapshot.people_locations,
                primary_activity_by_person=snapshot.primary_activity_by_person,
                zones_visible=snapshot.zones_visible,
                zone_descriptions=snapshot.zone_descriptions,
                primary_zone=snapshot.primary_zone,
                lighting_state=snapshot.lighting_state,
                visibility_state=snapshot.visibility_state,
                activity_level=snapshot.activity_level,
                camera_state=snapshot.camera_state,
                scene_summary=snapshot.scene_summary
            ))

        logger.info(f"Comprehensive analysis complete for {filename}")
        return descriptor


# Singleton instance for application use
_gemini_service = None


def get_gemini_service() -> GeminiService:
    """
    Get or create singleton GeminiService instance.

    This pattern ensures we reuse the same LLM instance across requests,
    avoiding unnecessary re-initialization overhead.
    """
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
